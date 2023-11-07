import numpy as np
import torch

from torch import nn
from nflows import transforms, flows
from nflows import distributions as distributions_

class NeuralDensityEstimator(flows.Flow):
    """
    Neural density estimator class of the MAF kind.
    """

    def __init__(
            self,
            dim: int = 1,
            initial_pos: dict = None,
            hidden_features: int = 50,
            num_transforms: int = 5,
            num_bins: int = 10,
            embedding_net = None,
            device=None,
            **kwargs):
        """
        Initialize neural density estimator.

        Parameters
        ----------
        dim: int
            Dimension of sample space
        initial_pos: dict.
            Initial position of the density, 
            e.g., ``{'bounds': [[1, 2], [0, 1]], 'std': [1, .05]}``.
            It includes the bounds for sampling the means of Gaussians, 
            and the standard deviations of the Gaussians.
        method: str.
            Method to use for density estimation, either ``'nsf'`` or ``'maf'``.
        hidden_features: int. 
            Number of hidden features.
        num_transforms: int. 
            Number of transforms.
        num_bins: int.
            Number of bins used for the splines.
        embedding_net: torch.nn.Module. 
            Optional embedding network for y.
        kwargs: dict.
            Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.
        """

        # build the NDE
        if initial_pos is None:
            raise ValueError(
                "initial_pos must be specified. Please see the documentation.")
        assert len(initial_pos['bounds']) == len(
            initial_pos['std']), "The length of bounds and std must be the same."

        transform, distribution, embedding_net = build_maf(
            dim=dim,
            initial_pos=initial_pos,
            hidden_features=hidden_features,
            num_transforms=num_transforms,
            embedding_net=embedding_net,
            **kwargs
        )

        super().__init__(transform, distribution, embedding_net=embedding_net)

def build_maf(
    dim: int = 1,
    z_score_x: bool = True,
    hidden_features: int = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    initial_pos: dict = {'bounds': [[1, 2], [0, 1]], 'std': [1, .05]},
    **kwargs,
):
    """Builds MAF to describe p(x).
    Args:
        dim: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms.
        embedding_net: Optional embedding network for y.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.
    Returns:
        Neural network.
    """
    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    transforms.MaskedAffineAutoregressiveTransform(
                        features=dim,
                        hidden_features=hidden_features,
                        num_blocks=2,
                        use_residual_blocks=False,
                        random_mask=False,
                        activation=torch.tanh,
                        dropout_probability=0.0,
                        use_batch_norm=False,
                    ),
                    transforms.RandomPermutation(features=dim),
                ]
            )
            for _ in range(num_transforms)
        ]
    )

    # if z_score_x:
    #     transform_zx = standardizing_transform(batch_x)
    #     transform = transforms.CompositeTransform([transform_zx, transform])
    #
    if initial_pos is not None:
        _mean = np.random.uniform(
            low=np.array(initial_pos['bounds'])[:, 0], high=np.array(initial_pos['bounds'])[:, 1])
        transform_init = transforms.AffineTransform(shift=torch.Tensor(-_mean) / torch.Tensor(initial_pos['std']),
                                                    scale=1.0 / torch.Tensor(initial_pos['std']))
        transform = transforms.CompositeTransform([transform_init, transform])

    distribution = distributions_.StandardNormal((dim,))
    return transform, distribution, embedding_net