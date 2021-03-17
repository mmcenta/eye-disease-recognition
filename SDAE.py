import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


def add_gaussian_noise(x, std):
    return x + Variable(x.data.new(x.size()).normal_(0, std))


class CDAE(nn.Module):
    r"""
    Convolutional denoising autoencoder layer for stacked autoencoders.

    Args:
        in_channels: the number of channels in the input.
        out_channels: the number of channels in the output.
        stride: stride of the convolutional layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, noise_std=0.1, **kwargs):
        super(CDAE, self).__init__(**kwargs)

        self.std = noise_std

        self.encoder = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        self.decoder = nn.ConvTranspose2d(out_channels, in_channels, kernel_size, stride=stride, padding=0)

    def forward(self, x):
        if self.training:
            x += add_gaussian_noise(x, self.std)
        
        emb = torch.relu(self.encoder(x))
        return emb, torch.relu(self.decoder(emb))

    def reconstruct(self, emb):
        return self.decoder(emb)


class SCDAE(nn.Module):
    """
    Stacked convolutional denoising auto-encoder.
    Gradients are detached between layers for independent training.
    """
    def __init__(self, in_channels, emb_channels=[200, 100], noise_std=0.1, **kwargs):
        super(SCDAE, self).__init__(**kwargs)

        self.stack = nn.ModuleList([CDAE(in_channels, emb_channels[0], 2, 
            noise_std=noise_std)])
        for i in range(len(emb_channels) - 1):
            self.stack.append(CDAE(emb_channels[i], emb_channels[i+1], 2,
                noise_std=noise_std))

    def forward(self, x):
        embs, recons = [x], []
        for m in self.stack:
            emb, recon = m(embs[-1].detach().clone())
            embs.append(emb)
            recons.append(recon)
        return embs, recons

    def reconstruct(self, emb):
        for m in reversed(self.stack):
            emb = m.reconstruct(emb)
        return emb


class TwinSDAE(nn.Module):
    """
    Twin stacked denoising auto-encoder.
    """
    def __init__(self, in_channels, emb_channels=[200, 100], noise_std=0.1,
        share_weights=False, **kwargs):
        super(TwinSDAE, self).__init__(**kwargs)

        self.left_sdae = SCDAE(in_channels, emb_channels=emb_channels,
            noise_std=noise_std)
        if share_weights:
            self.right_sdae = self.left_sdae
        else:
            self.right_sdae = SCDAE(in_channels, emb_channels=emb_channels,
            noise_std=noise_std)

    def forward(self, left_images, right_images):
        left_embs, left_recons = self.left_sdae(left_images)
        right_embs, right_recons = self.right_sdae(right_images)
        return {
            'left': (left_embs, left_recons),
            'right': (right_embs, right_recons),
        }
        