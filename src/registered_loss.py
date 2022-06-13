from typing import Any, Callable, Tuple, Optional

from pytorch_lightning import LightningModule
import torch
from torch import Tensor
from torch.nn import Conv3d

from src.lanczos import lanczos_kernel



class ShiftConv2d(LightningModule):
    '''
    A Conv2d layer and generates are shifted versions of x, with shifts
    from `start` to `end` with stepsize `step` on the last two dimensions.
    '''

    def __init__(self, start : float, end : float, step : float) -> None:
        """
        Args:
            start (float): relative start shift (in pixels).
            end (float): relative end shift (in pixels).
            step (float): (sub-)pixel shift of each step from start to end.
        """

        super().__init__()

        self.start = float(start)
        self.end = float(end)
        self.step = float(step)

#         if (step == 1) and (start == -end) and ((end - start) % 1 == 0):
#             K_y, K_x = self.separable_shift_kernels(w=w)
#             K_y, K_x = K_y[:, None, None], K_x[:, None, None]
#         else:
        K_y, K_x = self.separable_lanczos_kernels(self.start, self.end, self.step)

        o, _, _, h, _ = K_y.shape
        self.conv2d_yshift = Conv3d(in_channels=1, out_channels=o, kernel_size=(1, h, 1),
                                    padding=(0, h//2, 0), bias=False, padding_mode='zeros')
        self.conv2d_yshift.weight.data = K_y  # Fix and freeze the shift-kernel
        self.conv2d_yshift.requires_grad_(False)
        self.register_buffer("K_y", K_y)

        o, _, _, _, w = K_x.shape
        self.conv2d_xshift = Conv3d(in_channels=1, out_channels=o, kernel_size=(1, 1, w),
                                    padding=(0, 0, w//2), bias=False, padding_mode='zeros')
        self.conv2d_xshift.weight.data = K_x
        self.conv2d_xshift.requires_grad_(False)
        self.register_buffer("K_x", K_x)


#     ## This functionality is now covered by `separable_lanczos_kernels` and hence it is deprecated.
#     def separable_shift_kernels(self, w : int) -> Tuple[Tensor, Tensor]:
#         '''
#         Makes 2*w 1D convolution kernels (w, 1) for discrete shifts.

#         Args:
#             w (int): shift in number of pixels.

#         Returns:
#             A 2-tuple of tensors ((w, 1), (1, w))
#         '''

#         K_y = torch.zeros(w, w, 1)  # (num_kernels, H, W)
#         K_x = torch.zeros(w, 1, w)
#         for i in range(w):
#             K_y[i, i, 0] = 1
#             K_x[i, 0, i] = 1

#         return K_y, K_x


    def separable_lanczos_kernels(
        self,
        start : float,
        end : float,
        step : float,
    ) -> Tuple[Tensor, Tensor]:
        '''
        Makes two sets of 1D convolution kernels for y- and x-axis shifts.

        Args:
            start (float): relative start shift (in pixels).
            end (float): relative end shift (in pixels).
            step (float): (sub-)pixel shift of each step from start to end.

        Returns:
            A 2-tuple of tensors ((k, 1), (1, k))
        '''

        shift = torch.arange(start, end + 1e-3, step)[:, None]
        K_ = lanczos_kernel(shift, a=3)
        K_y = K_[:, None, None, :, None]
        K_x = K_[:, None, None, None, :]

        return K_y, K_x


    def forward(self, x : Tensor) -> Tensor:
        x = x[:, None]
        xs = self.conv2d_yshift(x)
        B, S, C, H, W = xs.shape
        xs = xs.view(B * S, 1, C, H, W)
        xs = self.conv2d_xshift(xs)
        _, _, _, H, W = xs.shape
        xs = xs.view(B, -1, C, H, W)
        return xs



class RegisteredLoss(LightningModule):
    '''
    Applies a loss func to shifted versions of y and forwards the min of the shifted losses.
    Initial version: https://gitlab.com/frontierdevelopmentlab/fdl-us-2020-droughts/xstream/-/blob/registered-loss/ml/src/loss.py#L73-130
    '''

    def __init__(
        self,
        start : float,
        end : float,
        step : float,
        loss_func : Callable,
        reduction : str = 'mean',
        **loss_kwargs : Any,
    ) -> None:
        """
        Args:
            start (float): relative start shift (in pixels).
            end (float): relative end shift (in pixels).
            step (float): (sub-)pixel shift of each step from start to end.
            loss_func (callable): loss function to apply at each pixel of each channel.
                Hint: use the `reduction='none'` option if the loss supports it.
            reduction (str, default='mean'): Reduction to apply along the batch dimension.
                One of ['mean'|'sum'|'none']. 'none' applies no reduction.
            **loss_kwargs (dict): arguments passed to `loss_func`.
        """

        super().__init__()

        self._shiftconv2d = ShiftConv2d(start, end, step)
        self.start = float(start)
        self.end = float(end)
        self.step = float(step)
        self.loss_func = loss_func
        self.loss_kwargs = loss_kwargs
        self.reduction = reduction


    def _shifted_loss(self, y_pred : Tensor, y : Tensor) -> torch.float32:
        '''
        Shifted versions of the loss.

        Args:
            y_pred : torch.Tensor (B, C, H, W).
            y : torch.Tensor (B, C, H, W).

        Returns:
            torch.Tensor (B, num_shifts)
        '''

        wy = self._shiftconv2d.conv2d_yshift.weight.shape[-2] // 2
        wx = self._shiftconv2d.conv2d_xshift.weight.shape[-1] // 2

        ## Shifted versions of y: (B, num_shifts, C, H, W).
        y_pred_shifted = self._shiftconv2d(y_pred)
        ## Do not evaluate loss at the border strips.
        y_pred_shifted = y_pred_shifted[..., wy:-wy, wx:-wx]

        ## Broadcastable view for loss_func. expand_as creates a view (no copying).
        _y = y[:, None, :, wy:-wy, wx:-wx]
        _y = _y.expand_as(y_pred_shifted)

        ## Element-wise loss.
        loss = self.loss_func(y_pred_shifted, _y, **self.loss_kwargs)

        return loss.mean(dim=(-3, -2, -1))  # Reduce along C, H, W dims.


    def registered_loss(self, y_pred : Tensor, y : Tensor) -> torch.float32:
        '''
        Version of the loss where only the min of each input in the mini-batch is forwarded.
        '''

        B = y.shape[0]  # Batch size
        loss_all_shifts = self._shifted_loss(y_pred, y)
        i_min_loss = loss_all_shifts.argmin(dim=1)  # Argmin loss
        min_loss = loss_all_shifts[range(B), i_min_loss]
        if self.reduction == 'mean':
            loss = min_loss.mean()
        elif self.reduction == 'sum':
            loss = min_loss.sum()
        elif self.reduction == 'none':
            loss = min_loss
        else:
            raise NotImplementedError(f"Expected `reduction` values: ['mean'|'sum'|'none']. Got {reduction}.")
        return loss


    def forward(self, y_pred : Tensor, y : Tensor) -> torch.float32:
        return self.registered_loss(y_pred, y)


    @staticmethod
    def register(start : float, end : float, step : float, reduction : str = 'mean', **loss_kws : Any) -> Callable:
        def _register_decorator(loss_func : Callable) -> Callable:
            return RegisteredLoss(start, end, step, loss_func=loss_func, reduction=reduction, **loss_kws)
        return _register_decorator
