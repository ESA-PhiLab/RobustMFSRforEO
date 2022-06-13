import sys
sys.path.append('..')

from argparse import ArgumentParser, Namespace
import pytest
import torch

from src.SRGAN import SRGAN


model = SRGAN()


@pytest.mark.filterwarnings("ignore:The default behavior for interpolate/upsample")
def test_srgan():
    lowres = torch.zeros([8,3,64,64])
    highres = torch.zeros([8,3,134,134])
    superres = model(lowres)
    assert superres.shape == highres.shape
