import torch
import torch.nn as nn
import numpy as np
import pdb
import torch.nn.init as init


class FourierLayer(nn.Module):
    def __init__(self, num_channels, initial_amplitude=None, initial_shift=None, initial_beta=None, phase=1,
                 cutoff_ratio=1):
        super(FourierLayer, self).__init__()
        self.num_channels = num_channels
        self.cutoff_ratio = cutoff_ratio
        self.L = 1


        if phase == 1:
            if initial_amplitude is None:
                self.amplitude = nn.Parameter(torch.randn(num_channels))
            else:

                self.amplitude = nn.Parameter(initial_amplitude.clone().detach().float())
        else:
            if initial_amplitude is None:
                self.amplitude = torch.randn(num_channels, dtype=torch.float32)
            else:

                self.amplitude = initial_amplitude.clone().detach().float().requires_grad_(False)


        if phase == 2:
            if initial_beta is None:
                self.beta = nn.Parameter(torch.randn(num_channels))
            else:
                self.beta = nn.Parameter(initial_beta.clone().detach().float())
        else:
            if initial_beta is None:
                self.beta = torch.randn(num_channels, dtype=torch.float32)
            else:
                self.beta = initial_beta.clone().detach().float().requires_grad_(False)


        if phase == 3:
            if initial_shift is None:
                self.shift = nn.Parameter(torch.randn(num_channels))
            else:
                self.shift = nn.Parameter(initial_shift.clone().detach().float())
        else:
            if initial_shift is None:
                self.shift = torch.randn(num_channels, dtype=torch.float32)
            else:
                self.shift = initial_shift.clone().detach().float().requires_grad_(False)


        if phase == 4:
            self.amplitude = nn.Parameter(
                torch.randn(num_channels) if initial_amplitude is None else initial_amplitude.clone().detach().float()
            )
            self.beta = nn.Parameter(
                torch.randn(num_channels) if initial_beta is None else initial_beta.clone().detach().float()
            )
            self.shift = nn.Parameter(
                torch.randn(num_channels) if initial_shift is None else initial_shift.clone().detach().float()
            )

        self.hypera = nn.Parameter(torch.rand(num_channels))



    def forward(self, x):

        # Perform Fourier transform and center it
        x_fft = torch.fft.fft2(x)
        x_fft = torch.fft.fftshift(x_fft)

        N = x.shape[2]
        kGrid = (2 * np.pi / self.L) * torch.fft.fftshift(torch.fft.fftfreq(N, d=1 / N)).to(x.device)
        kx_grid, ky_grid = torch.meshgrid(kGrid, kGrid, indexing='ij')

        k_sqrt = torch.sqrt(kx_grid ** 2 + ky_grid ** 2).to(x.device) / x.shape[2]

        multipliers = []
        for i in range(self.num_channels):
            part1 = self.amplitude[i] / ((k_sqrt + self.shift[i]) ** 2 + self.amplitude[i] ** 2)
            part2 = self.amplitude[i] / ((k_sqrt - self.shift[i]) ** 2 + self.amplitude[i] ** 2)
            filter_k = k_sqrt ** 2 / (k_sqrt ** 2 + self.hypera[i])

            multiplier = self.beta[i] * (part1 + part2) * filter_k

            # multiplier = self.beta[i] * (part1 + part2)
            multipliers.append(multiplier.unsqueeze(0))
        multipliers_sum = torch.sum(torch.stack(multipliers), dim=0)
        x_ifft = torch.fft.ifft2(x_fft * multipliers_sum).real
        x_ifft = torch.fft.ifftshift(x_ifft)

        return x_ifft, x_fft, self.amplitude, self.shift, self.beta, multipliers, self.hypera


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        # init.kaiming_uniform_(m.weight)
        # init.xavier_uniform_(m.weight)
        init.xavier_normal_(m.weight)
        # if m.bias is not None:
        #     nn.init.zeros_(m.bias)   # bias 通常初始化为


class SpectralLayer(nn.Module):
    def __init__(self, in_channels, out_channels, channel_feat, kernel_size, initial_amplitude=None, initial_shift=None,
                 initial_beta=None, phase=1):
        super(SpectralLayer, self).__init__()
        self.fu = FourierLayer(channel_feat, initial_amplitude, initial_shift, initial_beta, phase=phase)
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels, channel_feat, kernel_size, padding=pad, padding_mode='circular', bias=False)
        self.conv2 = nn.Conv2d(channel_feat, out_channels, kernel_size, padding=pad, padding_mode='circular',
                               bias=False)
        k2 = 3
        p2 = (k2 - 1) // 2
        self.conv10 = nn.Conv2d(channel_feat, channel_feat, k2, padding=p2, padding_mode='circular',
                                bias=False)

        self.tanh = nn.Tanh()

        padding_s = (kernel_size - 2) // 2
        padding_up = (kernel_size - 2) // 2
        self.stride_conv_d5 = nn.Conv2d(channel_feat, channel_feat, kernel_size=kernel_size, stride=5,
                                        padding=padding_s)
        self.transconv_d5 = nn.ConvTranspose2d(channel_feat, channel_feat, kernel_size=kernel_size, stride=5,
                                               padding=padding_up)

        self.apply(init_weights)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.tanh(x1)
        x1 = self.stride_conv_d5(x1)

        x1 = self.conv10(x1)

        x1 = self.transconv_d5(x1)

        x1 = self.conv2(x1)
        x1 = self.tanh(x1)
        iffx, ffx, amplitude, shift, beta, multiplier, a_par = self.fu(x)
        iffx = self.tanh(iffx)
        output = x1 + iffx
        # output = x1

        return output, x1, iffx, output, ffx, amplitude, shift, beta, multiplier, a_par


class Wise_SpectralTransform(nn.Module):
    def __init__(self, num_layers, channel_feat, kernel_size, phase=1, initial_parameters=None):
        super(Wise_SpectralTransform, self).__init__()
        layers = []

        for i in range(num_layers):
            initial_amplitude = (
                initial_parameters['amplitudes'][i] if initial_parameters and initial_parameters[
                    'amplitudes'] is not None else None
            )
            initial_shift = (
                initial_parameters['shifts'][i] if initial_parameters and initial_parameters[
                    'shifts'] is not None else None
            )
            initial_beta = (
                initial_parameters['betas'][i] if initial_parameters and initial_parameters[
                    'betas'] is not None else None
            )

            spindle = SpectralLayer(
                in_channels=1,
                out_channels=1,
                channel_feat=channel_feat,
                kernel_size=kernel_size,
                initial_amplitude=initial_amplitude,
                initial_shift=initial_shift,
                initial_beta=initial_beta,
                phase=phase
            )
            layers.append(spindle)

        self.snet = nn.Sequential(*layers)

    def forward(self, input):
        x = input
        seq = []
        mid = {}
        fourier = {}
        amplitudes = []
        shifts = []
        betas = []
        multipliers = []
        a_pars = []
        layer_number = 0

        seq.append(input)
        fourier[layer_number] = [input]

        for layer in self.snet:
            x, mid1, iffx, output, ffx, amplitude, shift, beta, multiplier, a_par = layer(x)
            layer_number += 1

            seq.append(x)
            amplitudes.append(amplitude)
            shifts.append(shift)
            betas.append(beta)
            multipliers.append(multiplier)
            a_pars.append(a_par)

            mid[layer_number] = [mid1, iffx, output]
            fourier[layer_number] = [ffx, iffx]

        mid[-1] = [input]
        mid[0] = [input]

        return x, seq, mid, fourier, amplitudes, shifts, betas, multipliers, a_pars
