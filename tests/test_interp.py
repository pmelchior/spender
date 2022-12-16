import numpy as np
import torch
from torchinterp1d import interp1d as torchinterp1d
from tqdm.auto import tqdm
from spender.util import interp1d as spender_interp1d

rtol = 1e-3
atol = 1e-6
n = 100
runs = 1000

def test_interp1d_torchinterp1d():
    for _ in tqdm(range(runs)):
        x = torch.rand(n)
        x = torch.sort(x).values
        y = torch.rand(n)
        xq = torch.rand(n)
        xq = torch.sort(xq).values

        yq = torchinterp1d(x, y, xq).numpy()
        yq2 = spender_interp1d(x, y, xq, mask=False).numpy()

        assert np.allclose(yq[1:-1], yq2[1:-1], rtol=rtol, atol=atol), (yq, yq2)


def test_interp1d_numpy():
    for _ in tqdm(range(runs)):
        x = torch.rand(n)
        x = torch.sort(x).values
        y = torch.rand(n)
        xq = torch.rand(n)
        xq = torch.sort(xq).values

        yq3 = spender_interp1d(x, y, xq, mask=True).numpy()
        yq4 = np.interp(xq.numpy(), x.numpy(), y.numpy())

        assert np.allclose(yq3, yq4, rtol=rtol, atol=atol), (yq3, yq4)


if __name__ == "__main__":
    test_interp1d_torchinterp1d()
    test_interp1d_numpy()
