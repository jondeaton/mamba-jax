import jax
import jax.numpy as jnp

import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray


class BidirectionalConv(eqx.Module):
    conv_left: eqx.nn.Conv
    conv_right: eqx.nn.Conv

    def __init__(self, dim: int, kernel_size: int, key: PRNGKeyArray):
        pad_size = kernel_size - 1
        keys = jax.random.split(key)
        self.conv_left = eqx.nn.Conv(
            num_spatial_dims=1,
            in_channels=dim // 2,
            out_channels=dim // 2,
            groups=dim // 2,
            kernel_size=kernel_size,
            padding=((pad_size, 0),),
            use_bias=False,
            key=keys[0],
        )
        self.conv_right = eqx.nn.Conv(
            num_spatial_dims=1,
            in_channels=dim // 2,
            out_channels=dim // 2,
            groups=dim // 2,
            kernel_size=kernel_size,
            padding=((0, pad_size),),
            use_bias=False,
            key=keys[1],
        )

    def __call__(self, x: Float[Array, "d l"]) -> Float[Array, "d l"]:
        # assert x.shape[1] == self.conv_left.in_channels, x.shape
        _, d = x.shape
        x_left = self.conv_left(x[: d // 2, :])
        x_right = self.conv_right(x[d // 2 :, :])
        # assert False, (x_left.shape, x_right.shape)
        return jnp.concatenate([x_left, x_right])
