import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel_output, duration_predicted, mel_target, duration, **kwargs):
        mel_loss = self.mse_loss(mel_output, mel_target)

        duration_predictor_loss = self.l1_loss(duration_predicted,
                                               torch.log(duration.float() + 1))

        loss = mel_loss + duration_predictor_loss

        return loss