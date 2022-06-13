import sys
sys.path.append('..')

from argparse import ArgumentParser, Namespace
import pytest
import torch

from src.FSRCNN import FSRCNNModule


net_args = Namespace(in_channels=3, out_channels=3, upscale_factor=2, additional_scaling=1.0469)
args = Namespace(batch_size = 16, lr = 0.001, lr_decay=0.95, loss ='MSE')
model = FSRCNNModule(net_args, lr= args.lr, lr_decay=args.lr_decay, loss=args.loss)


@pytest.mark.filterwarnings("ignore:The default behavior for interpolate/upsample")
def test_fsrcnn():
    lowres = torch.zeros([8,3,64,64])
    highres = torch.zeros([8,3,134,134])
    superres = model(lowres)
    assert superres.shape == highres.shape
