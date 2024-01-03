"""Mamba implementaiton in JAX / equinox.

References
    1. Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    2. minimal-mamba: Simple, minimal implementation of the Mamba SSM in one file of PyTorch
        John (Zhiyao) Ma https://github.com/johnma2006/mamba-minimal
"""

from typing import Callable
from jaxtyping import Array, Integer, Float, Bool, PRNGKeyArray

import math

import jax
import jax.numpy as jnp
import einops
import equinox as eqx

import mamba.bidirectional


class Mamba(eqx.Module):
    """Top-level mamba module."""

    embed: eqx.nn.Embedding
    layers: Callable
    bidirectional: bool = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        d_inner: int,
        d_state: int,
        key: PRNGKeyArray,
        bidirectional: bool = False,
    ):
        self.bidirectional = bidirectional
        embed_key, layer_key = jax.random.split(key)
        self.embed = eqx.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_size=d_model,
            key=embed_key,
        )

        def block(key):
            return ResidualBlock(MambaBlock(d_model, d_inner, d_state, key=key))

        keys = jax.random.split(layer_key, n_layer)
        self.layers = jax.vmap(block)(keys)

    def __call__(self, token_ids: Integer[Array, "l"]) -> Float[Array, "l d"]:
        x = jax.vmap(self.embed)(token_ids)
        resets = token_ids <= 2  # reset at pad, <cls> or <eos> token.
        x, _ = jax.lax.scan(lambda x, layer: (layer(x, resets), None), x, self.layers)
        return x @ self.embed.weight.T


class ResidualBlock(eqx.Module):
    layer: Callable

    def __init__(self, layer: Callable):
        self.layer = layer

    def __call__(
        self,
        x: Float[Array, "l d"],
        *args,
        **kwargs,
    ) -> Float[Array, "l d"]:
        x_ = jax.vmap(lambda x: x * jax.lax.rsqrt(jnp.mean(x**2) + 1e-6))(x)
        x_ = self.layer(x_, *args, **kwargs)
        return x + x_


class MambaBlock(eqx.Module):
    conv: eqx.nn.Conv | mamba.bidirectional.BidirectionalConv
    res_proj: eqx.nn.Linear
    in_proj: eqx.nn.Linear

    B_proj: eqx.nn.Linear
    C_proj: eqx.nn.Linear
    dt_proj: eqx.nn.Sequential

    log_A: Float[Array, "d_inner d_state"]
    D: Float[Array, "d_inner"]
    out_proj: eqx.nn.Linear
    bidirectional: bool = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        d_state: int,
        key: PRNGKeyArray,
        dt_rank: int | None = None,
        conv_kernel_size: int = 4,
        bidirectional: bool = False,
    ):
        self.bidirectional = bidirectional
        keys = jax.random.split(key, 10)
        self.res_proj = eqx.nn.Linear(d_model, d_inner, use_bias=False, key=keys[0])
        self.in_proj = eqx.nn.Linear(d_model, d_inner, use_bias=False, key=keys[1])

        if self.bidirectional:
            self.conv = mamba.bidirectional.BidirectionalConv(
                dim=d_inner,
                kernel_size=conv_kernel_size,
                key=keys[2],
            )
        else:
            pad_size = conv_kernel_size - 1
            self.conv = eqx.nn.Conv(
                num_spatial_dims=1,
                in_channels=d_inner,
                out_channels=d_inner,
                kernel_size=conv_kernel_size,
                padding=((pad_size, 0)),
                groups=d_inner,
                use_bias=False,
                key=keys[2],
            )

        self.B_proj = eqx.nn.Linear(d_inner, d_state, use_bias=False, key=keys[3])
        self.C_proj = eqx.nn.Linear(d_inner, d_state, use_bias=False, key=keys[4])

        dt_rank = dt_rank or math.ceil(d_model / 16)
        dt_proj = eqx.nn.Sequential(
            [
                eqx.nn.Linear(d_inner, dt_rank, use_bias=False, key=keys[5]),
                eqx.nn.Linear(dt_rank, d_inner, use_bias=True, key=keys[6]),
            ]
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_scale = 1.0
        dt_init_std = jax.lax.rsqrt(float(dt_rank)) * dt_scale
        weight = jax.random.uniform(
            keys[8],
            shape=dt_proj.layers[1].weight.shape,
            minval=-dt_init_std,
            maxval=dt_init_std,
        )
        dt_proj = eqx.tree_at(lambda l: l.layers[1].weight, dt_proj, weight)

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = jnp.clip(
            jnp.exp(
                jax.random.uniform(keys[9], shape=(d_inner,))
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ),
            a_min=dt_init_floor,
        )
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        bias = dt + jnp.log(-jnp.expm1(-dt))
        self.dt_proj = eqx.tree_at(lambda l: l.layers[1].bias, dt_proj, bias)

        A = einops.repeat(jnp.arange(1, d_state + 1), "n -> d n", d=d_inner)
        self.log_A = jnp.log(A)
        self.D = jnp.ones(shape=(d_inner,))
        self.out_proj = eqx.nn.Linear(d_inner, d_model, use_bias=False, key=keys[7])

    def __call__(
        self,
        x: Float[Array, "l d"],
        resets: Bool[Array, "l"] | None = None,
    ) -> Float[Array, "l d"]:
        res = jax.vmap(self.res_proj)(x)
        x = jax.vmap(self.in_proj)(x)

        x = self.conv(x.T).T
        assert x.shape == res.shape, (x.shape, res.shape)
        x = jax.nn.silu(x)

        y = self.ssm(x, resets)
        y *= jax.nn.silu(res)

        out = jax.vmap(self.out_proj)(y)
        return out

    def ssm(
        self,
        x: Float[Array, "l d_inner"],
        resets: Bool[Array, "l"] | None = None,
    ) -> Float[Array, "l d_inner"]:
        A = -jnp.exp(self.log_A)

        B = jax.vmap(self.B_proj)(x)
        C = jax.vmap(self.C_proj)(x)
        # log(1+e^x) keeps time-step, dt, positive.
        dt = jax.nn.softplus(jax.vmap(self.dt_proj)(x))

        if not self.bidirectional:
            return selective_scan(x, dt, A, B, C, self.D)
        else:
            xf, xb = jnp.split(x, 2, axis=1)
            dtf, dtb = jnp.split(dt, 2, axis=1)
            Af, Ab = jnp.split(A, 2, axis=0)
            Df, Db = jnp.split(self.D, 2)
            fwd = selective_scan(xf, dtf, Af, B, C, Df, resets=resets)
            bwd = selective_scan(xb, dtb, Ab, B, C, Db, resets=resets, reverse=True)
            return jnp.concatenate([fwd, bwd], axis=1)


def selective_scan(
    x: Float[Array, "l d_in"],
    dt: Float[Array, "l d_in"],
    A: Float[Array, "d_in n"],
    B: Float[Array, "l n"],
    C: Float[Array, "l n"],
    D: Float[Array, "d_in"],
    resets: Bool[Array, "l"] | None = None,
    discretization: str = "bilinear",
    reverse: bool = False,
) -> Float[Array, "l d_in"]:
    """Selective Scan Algorithm.

    Reset enables
        1. sequence packing <seq1>!<seq2>
        2. padding invariance for bidirectional.
    """
    l, d_in = x.shape
    _, n = A.shape

    if resets is None:
        resets = jnp.zeros(shape=(l,), dtype=bool)

    dt_A = dt[:, :, None] * A[None, :, :]
    dt_B = dt[:, :, None] * B[:, None, :]
    if discretization == "bilinear":
        # Following kernel() of Listing 1 in https://arxiv.org/pdf/2206.11893.pdf
        dA = (1 + dt_A / 2) / (1 - dt_A / 2)
        dB = dt_B / (1 - dt_A / 2)
    elif discretization == "zoh":
        # TODO: review this.
        dA = jnp.exp(dt_A)
        dB = dt_B
    else:
        raise NotImplementedError()

    assert isinstance(dA, Float[Array, f"{l} {d_in} {n}"])
    assert isinstance(dB, Float[Array, f"{l} {d_in} {n}"])

    def f(
        h: Float[Array, "d_in n"],
        params: tuple[
            Float[Array, "d_in"],  # x
            Float[Array, "d_in n"],  # dA
            Float[Array, "d_in n"],  # dB
            Float[Array, "n"],  # C
            Bool,  # reset
        ],
    ):
        xi, dAi, dBi, Ci, reset = params

        h_: Float[Array, "d_in n"] = jax.lax.cond(
            reset,
            lambda _: jnp.zeros_like(h),
            lambda _: dAi * h + dBi * xi[:, None],
            None,
        )
        y: Float[Array, "d_in"] = h_ @ Ci

        return h_, y

    h0 = jnp.zeros(shape=(d_in, n), dtype=x.dtype)
    _, y = jax.lax.scan(f, h0, (x, dA, dB, C, resets), reverse=reverse)
    return y + x * D
