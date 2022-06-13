""" Pytorch implementation of HighRes-net, a neural network for multi-frame super resolution (MFSR) by recursive fusion. """

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

import src.lanczos as lanczos



class ResidualBlock(nn.Module):

#     def __init__(self, in_channels : int = 64, kernel_size : int = 3) -> None:
    def __init__(self, in_channels : int, kernel_size : int) -> None:
        '''
        Args:
            in_channels : int, number of hidden channels.
            kernel_size : int, shape of a 2D kernel.
        '''
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding),
            nn.PReLU()
        )


    def forward(self, x : Tensor) -> Tensor:
        '''
        Args:
            x : tensor (B, C, H, W), encoding.
        Returns:
            x + residual: tensor (B, C, H, W).
        '''
        return x + self.block(x)



class Encoder(nn.Module):
    '''
    Encodes an input tensor x.
    '''

    def __init__(self, in_channels : int, out_channels : int, num_layers : int, kernel_size : int) -> None:
        '''
        Args:
            in_channels : int, number of input channels.
            out_channels : int, number of output channels.
            num_layers : int, number of residual block layers.
            kernel_size : int, shape of a 2D kernel.
        '''

        super(Encoder, self).__init__()
        padding = kernel_size // 2

        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.PReLU()
        )

        res_layers = [ResidualBlock(out_channels, kernel_size) for _ in range(num_layers)]
        self.res_layers = nn.Sequential(*res_layers)

        ## TODO: is this necessary?
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        )


    def forward(self, x : Tensor) -> Tensor:
        '''
        Args:
            x : tensor (B, C, H, W), batch.
        Returns:
            out: tensor (B, C, H, W), batch encodings.
        '''
        x = self.init_layer(x)
        x = self.res_layers(x)
        x = self.final(x)
        return x



class RecursiveFusion(nn.Module):
    '''
    Recursive fusion of the encodings of multiple revisits.
    '''

    def __init__(self, in_channels : int, kernel_size : int, revisits_residual : bool) -> None:
        '''
        Fusion operator; reducing the number of input channels by a factor of 2.
        Args:
            in_channels : int, number of input channels.
            kernel_size : int, shape of a 2D kernel.
            revisits_residual : bool, include padded views when fusing.
        '''
        super(RecursiveFusion, self).__init__()
        self.revisits_residual = revisits_residual
        padding = kernel_size // 2
        self.fuse = nn.Sequential(
            ResidualBlock(2 * in_channels, kernel_size),  # 2 * in_channels from the two input images
            nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding),
            nn.PReLU()
        )


    def forward(self, x : Tensor, r : Tensor) -> Tensor:
        '''
        Args:
            x : tensor (B, T, C, H, W), batch encodings, of T revisits per instance.
            r : tensor (B, T, 1, 1, 1), binary revisits indicator of:
                 zeros if the revisit is a padding; ones otherwise.
        Returns:
            tensor (B, C, H, W), batch fused encodings.
        '''

        B, T, C, H, W = x.shape
        parity = T % 2
        half_T = T // 2

        while half_T > 0:
            x1 = x[:, :half_T]  # first half of revisit encodings (B, T/2, C, H, W)
            x2 = x[:, half_T:T - parity]  # second half
            x2 = torch.flip(x2, [1])

            x12 = torch.cat([x1, x2], 2)  # concat on channel dim (B, T/2 pairs, 2*C, H, W)
            x12 = x12.view(-1, 2 * C, H, W)  # revisits as batch instances
            x = self.fuse(x12)  # fuse channels per revisit pair (B * T/2, C, H, W)
            x = x.view(B, half_T, C, H, W)  # fused encodings (B, T/2, C, H, W)

            if self.revisits_residual:  # skip connect padded views (r_B = 0)
                r1 = r[:, :half_T]
                r2 = r[:, half_T:T - parity]
                r2 = torch.flip(r2, [1])
                x = x1 + (r2 * x)  # (B, T/2 pairs, C, H, W)
                r = r1

            T = half_T
            parity = T % 2
            half_T = T // 2

        ## TODO: What is this for??
        ## By this point x should be (B, 1, C, H, W).
        ## It should suffice to say x.squeeze(1), or x[:, 0].
        return torch.mean(x, 1)



class Decoder(nn.Module):
    '''
    Decodes an fused encoding x.
    '''

    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        upsample_channels : int,
        deconv_kernel_size : int,
        final_kernel_size : int,
        deconv_stride : int,
        additional_scale_factor : Optional[float] = None,
    ) -> None:
        '''
        Args:
            in_channels : int, number of input channels.
            out_channels: int, number of output channels.
            upsample_channels : int, number of channels in the upsampled encoding.
            deconv_kernel_size : int, shape of a 2D kernel in the Upsample layer.
            final_kernel_size : int, shape of a 2D kernel in the final Conv2d layer.
            deconv_stride : int, stride of ConvTranspose2d.
            additional_scale_factor : float, scaling factor for additional upscaling.
        '''

        super(Decoder, self).__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=upsample_channels, kernel_size=deconv_kernel_size, stride=deconv_stride),
            nn.PReLU()
        )

        self.scale = False  # no additional scaling by default
        if additional_scale_factor is not None:
            self.scale = True
            ## TODO: look into upscaling with Lanczos
            self.additional_scaling = nn.Sequential(nn.Upsample(mode='bicubic', scale_factor=additional_scale_factor, align_corners=False))

        self.final = nn.Conv2d(in_channels=upsample_channels,
                               out_channels=out_channels,
                               kernel_size=final_kernel_size,
                               padding=final_kernel_size//2)


    def forward(self, x : Tensor) -> Tensor:
        '''
        Args:
            x : tensor (B, C, H, W), batch fused encodings.
        Returns:
            out: tensor (B, C_out, s*H, s*W), batch encodings upscaled by a factor of s.
        '''
        x = self.deconv(x)
        if self.scale:
            x = self.additional_scaling(x)
        x = self.final(x)
        return x



class HighResNet(nn.Module):
    ''' 
    HighResNet, a neural network for multi-frame super resolution (MFSR) by recursive fusion.
    Super-resolves by fusing many revisits of low-resolution images.
    '''

    def __init__(
        self,
        in_channels : int,
        num_channels : int,
        out_channels : int,
        revisits_residual : bool,
        num_encoder_layers : int = 2,
        additional_scale_factor : Optional[float] = None,
    ) -> None:
        '''
        Args:
            in_channels : int, number of input channels.
            num_channels : int, number intermediate channels for representational learning.
            out_channels: int, number of output channels.
            revisits_residual : bool, include padded views when fusing.
            num_encoder_layers : int, number of residual layers in encoder.
            additional_scale_factor : float, scaling factor for additional upscaling.
        '''

        super(HighResNet, self).__init__()

        ## in_channels * 2 due to appending the reference (anchor) image.
        self.encode = Encoder(in_channels=in_channels*2, out_channels=num_channels, num_layers=num_encoder_layers, kernel_size=3)

        self.fuse = RecursiveFusion(in_channels=num_channels, kernel_size=3, revisits_residual=True)

        self.decode = Decoder(in_channels=num_channels, out_channels=out_channels, upsample_channels=num_channels, deconv_kernel_size=3,
                              final_kernel_size=1, deconv_stride=2, additional_scale_factor=additional_scale_factor)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x, r):
        '''
        Args:
            x : tensor (B, T, C, H, W), batch of low-resolution images, with T revisits per instance.
            r : tensor (B, T), binary revisits indicator of zeros if the revisit is a padding; ones otherwise.
        Returns:
            tensor (B, C, H, W), batch of super-resolved images.
        '''

        B, T, C, H, W = x.shape
        min_revisits = torch.count_nonzero(r, dim=1).min()
        r = r.view(B, T, 1, 1, 1)
        ## Reference image (aka anchor), shared across multiple revisits.
        refs, _ = torch.median(x[:, :min_revisits], dim=1, keepdim=True)  # (B, 1, C, H, W)
        refs = refs.expand_as(x)  # (B, T, C, H, W)
        x2 = torch.cat([x, refs], 2)  # Concat with copies of the anchor (B, T, C * 2, H, W)
        x2 = x2.view(B * T, C * 2, H, W)

        z = self.encode(x2)  # Encode revisits appended with the anchor (B * T, num_channels, H, W)
        z = z.view(B, T, -1, H, W)  # (B, T, num_channels, H, W)

        ## Fuse and upsample
        z_fused = self.fuse(z, r.float())  # Fuse encodings (B, num_channels, H, W)
        srs = self.decode(z_fused)  # Decode final encoding (B, out_channels, s*H, s*W)
        srs = self.sigmoid(srs)
        return srs



class ShiftNet(nn.Module):
    '''
    ShiftNet, a neural network for sub-pixel registration and interpolation with lanczos kernel.
    Registers pairs of images through sub-pixel shift parameters.

    Based on a Pytorch implementation of HomographyNet.
    Reference: https://arxiv.org/pdf/1606.03798.pdf and https://github.com/mazenmel/Deep-homography-estimation-Pytorch
    
    Currently supports translations (2 params).
    Inputs pairs of images (B, 2 * C, H, W) and outputs translation parameters (B, 2).
    '''

    def __init__(self, in_channel : int = 3) -> None:
        '''
        Args:
            in_channel : int, number of input channels.
        '''

        super(ShiftNet, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(2 * in_channel, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))

        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))

        self.layer5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))

        self.layer7 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())

        self.drop1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        self.activ1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 2, bias=False)
        self.fc2.weight.data.zero_()  # Init the weights with the identity transformation


    def forward(self, x : Tensor) -> Tensor:
        '''
        Args:
            x : tensor (B, 2, C, H, W), batch of pairs of images.
        Returns:
            tensor (B, 2), translation parameters.
        '''

        ## Center each image with its own mean.
        C = self.in_channel  # this must be added to the constructor
        x[:, :C] = x[:, :C] - torch.mean(x[:, :C], dim=(2, 3), keepdim=True)
        x[:, C:] = x[:, C:] - torch.mean(x[:, C:], dim=(2, 3), keepdim=True)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        out = out.view(-1, 128 * 16 * 16)
        out = self.drop1(out)  # dropout on spatial tensor (C*W*H)

        out = self.fc1(out)
        out = self.activ1(out)
        out = self.fc2(out)
        return out


    def transform(self, theta : Tensor, I : Tensor) -> Tensor:
        '''
        Shifts images I by theta with Lanczos interpolation.
        Args:
            theta : tensor (B, 2), translation params
            I : tensor (B, C_in, H, W), input images
        Returns:
            out: tensor (B, C_in, W, H), shifted images
        '''
        new_I = lanczos.lanczos_shift(img=I,
                                      shift=theta.flip(-1),
                                      a=3, p=5)

        return new_I
