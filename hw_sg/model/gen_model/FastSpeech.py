import torch
import torch.nn as nn
import torch.nn.functional as F

from .trasformers import Encoder, Decoder
from .predictors import LengthRegulator



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
        self.length_regulator = LengthRegulator(**params)
        self.decoder = Decoder(**params)

        self.mel_linear = nn.Linear(params['decoder_dim'], params['num_mels'])

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0):
        x, mask = self.encoder(src_seq, src_pos)
        if self.training:
            output, duration_predicted = self.length_regulator(x, alpha, length_target, mel_max_length)
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            return {
                "mel_output": output,
                "duration_predicted": duration_predicted
            }
        else:
            output, mel_pos = self.length_regulator(x, alpha)
            output = self.decoder(output, mel_pos)
            output = self.mel_linear(output)
            return {"mel_output":output}













