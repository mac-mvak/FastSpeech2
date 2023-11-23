import torch
import torch.nn as nn
import torch.nn.functional as F

from .trasformers import Encoder, Decoder
from .predictors import VarianceAdapter



def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self, **params):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(**params)
        self.variance_adapter = VarianceAdapter(**params)
        self.decoder = Decoder(**params)

        self.mel_linear = nn.Linear(params['decoder_dim'], params['num_mels'])

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, text, src_pos, mel_pos=None, mel_max_length=None, duration=None, 
                energy_target=None, energy_coeff=1.0, length_coeff=1.0, **kwargs):        
        x, mask = self.encoder(text, src_pos)
        if self.training:
            output, duration_predicted, energy_pred = self.variance_adapter(x, 
                                                        length_coeff=length_coeff, 
                                                        length_target=duration, mel_max_length=mel_max_length,
                                                        energy_target=energy_target,
                                                        energy_coeff=energy_coeff)
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            return {
                "mel_output": output,
                "duration_predicted": duration_predicted,
                "energy_predicted":energy_pred
            }
        else:
            output, mel_pos = self.variance_adapter(x, length_coeff=length_coeff, energy_coeff=energy_coeff)
            output = self.decoder(output, mel_pos)
            output = self.mel_linear(output)
            return {"mel_output":output}













