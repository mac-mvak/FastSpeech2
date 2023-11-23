import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np





def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat

class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)




class FactorPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self,
                encoder_dim, var_predictor_filter_size,
                var_predictor_kernel_size, var_dropout, **kwargs):
        super(FactorPredictor, self).__init__()

        self.input_size = encoder_dim
        self.filter_size = var_predictor_filter_size
        self.kernel = var_predictor_kernel_size
        self.conv_output_size = var_predictor_filter_size
        self.dropout = var_dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        for block in self.conv_net:
            encoder_output = block(encoder_output)
            
        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out
    

class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, **params):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = FactorPredictor(**params)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predicted = self.duration_predictor(x)

        if target is not None:
            out = self.LR(x, target, mel_max_length)
            return out, duration_predicted
        else:
            duration_predicted = (
                (torch.exp(duration_predicted) - 1)* alpha + 0.5
            ).int()
            duration_predicted[duration_predicted < 0] = 0
            out = self.LR(x, duration_predicted)
            mel_pos = torch.stack(
                [torch.Tensor([i+1  for i in range(out.size(1))])]
            ).long().to(x.device)
            return out, mel_pos


class VarianceAdapter(nn.Module):
    """ Variance Analyzer for Energy, Duration, Pitch """
    def __init__(self, **params):
        super(VarianceAdapter, self).__init__()
        self.duration_regulator = LengthRegulator(**params)
        self.energy_predictor = FactorPredictor(**params)
        self.pitch_predictor = FactorPredictor(**params)

        energies = [params['energy_min'], params['energy_max']]
        ener_bins = torch.linspace(np.log(energies[0] +1), np.log(energies[1]+2), params['num_bins'])
        self.register_buffer('ener_bins', ener_bins)
        self.ener_embeds = nn.Embedding(params['num_bins'], params['encoder_dim'])

        pitches = [params['min_pitch'], params['max_pitch']]
        pitch_bins = torch.linspace(np.log(pitches[0] +1), np.log(pitches[1]+2), params['num_bins'])
        self.register_buffer('pitch_bins', pitch_bins)
        self.pitch_embeds = nn.Embedding(params['num_bins'], params['encoder_dim'])
    
    def get_discr(self, factor_type, x, target=None, coeff=1.):
        if factor_type == 'energy':
            pred = self.energy_predictor(x)
            bins = self.ener_bins

        if factor_type == 'pitch':
            pred = self.pitch_predictor(x)
            bins = self.pitch_bins
        
        if target is not None:
            buckets = torch.bucketize(torch.log1p(target), bins)
        else:
            estimated = (torch.exp(pred) - 1) * coeff
            buckets = torch.bucketize(torch.log1p(estimated), bins)
        
        if factor_type=='energy':
            return self.ener_embeds(buckets), pred
        if factor_type == 'pitch':
            return self.pitch_embeds(buckets), pred

    def forward(self, enc_output, length_target=None, energy_target=None, pitch_target=None,
                 mel_max_length=None, length_coeff=1, energy_coeff=1, pitch_coeff=1):
        out, duration_predicted = self.duration_regulator(enc_output, target=length_target, mel_max_length=mel_max_length, alpha=length_coeff)

        ener_embeds, ener_pred = self.get_discr('energy', out, energy_target, energy_coeff)
        pitch_embeds, pitch_pred = self.get_discr('pitch', out, pitch_target, pitch_coeff)

        combined_embeds = out + ener_embeds + pitch_embeds
        return combined_embeds, duration_predicted, ener_pred, pitch_pred

 



