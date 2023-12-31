import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx

import mamba


def test_mamba_forward():
    l = 16
    d_model = 128

    key = jax.random.PRNGKey(0)
    model = mamba.Mamba(
        d_model=d_model, n_layer=2, vocab_size=10, d_inner=32, d_state=16, key=key
    )

    token_ids = jax.random.randint(key, shape=(l,), minval=0, maxval=10)
    logits = eqx.filter_jit(model)(token_ids)
    assert logits.shape == (l, 10)
    assert not np.isnan(logits).any()


def test_mamba_grad():
    b = 4
    l = 16
    d_model = 128

    key = jax.random.PRNGKey(0)
    model = mamba.Mamba(
        d_model=d_model, n_layer=2, vocab_size=10, d_inner=32, d_state=16, key=key
    )

    def loss_fn(model, x):
        y = jax.vmap(model)(x)
        return jnp.sum(jnp.square(y))

    token_ids = jax.random.randint(key, shape=(b, l), minval=0, maxval=10)

    loss, grads = jax.jit(jax.value_and_grad(loss_fn))(model, token_ids)
    assert not jnp.isnan(loss)
    assert isinstance(grads, mamba.Mamba)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda x: ~jnp.isnan(x).any(), grads)
    )


def test_mamba_block_causality():
    l = 16
    d_model = 128

    key = jax.random.PRNGKey(0)
    block = mamba.MambaBlock(d_model=d_model, d_inner=32, d_state=16, key=key)
    fwd = eqx.filter_jit(block)

    x = jax.random.uniform(key, shape=(l, d_model))
    y = fwd(x)

    for i in range(l):
        x_ = x.at[i, :].set(jnp.zeros(d_model))
        y_ = fwd(x_)
        np.testing.assert_allclose(y[:i], y_[:i])
        assert jnp.isclose(y[i:], y_[i:]).all(axis=1).mean() < 0.1


def test_mamba_causality():
    l = 16
    d_model = 128

    key = jax.random.PRNGKey(0)
    model = mamba.Mamba(
        d_model=d_model, n_layer=2, vocab_size=10, d_inner=32, d_state=16, key=key
    )
    fwd = eqx.filter_jit(model)

    token_ids = jax.random.randint(key, shape=(l,), minval=0, maxval=10)
    y = fwd(token_ids)

    for i in range(l):
        t = token_ids.at[i].set((token_ids[i] + 1) % 10)
        y_ = fwd(t)
        np.testing.assert_allclose(y[:i], y_[:i])


def test_mamba_block_bidirectional():
    l = 16
    d_model = 128

    key = jax.random.PRNGKey(0)
    block = mamba.MambaBlock(
        d_model=d_model,
        d_inner=32,
        d_state=16,
        key=key,
        bidirectional=True,
    )
    fwd = eqx.filter_jit(block)

    x = jax.random.uniform(key, shape=(l, d_model))
    y = fwd(x)

    for i in range(l):
        x_ = x.at[i, :].set(jnp.zeros(d_model))
        y_ = fwd(x_)
        assert np.isclose(y, y_).all(axis=1).sum() == 0


def test_mamba_bidirectional():
    l = 64
    d_model = 1024

    key = jax.random.PRNGKey(0)
    model = mamba.Mamba(
        d_model=d_model,
        n_layer=4,
        vocab_size=10,
        d_inner=256,
        d_state=128,
        bidirectional=True,
        key=key,
    )
    fwd = eqx.filter_jit(model)

    token_ids = jax.random.randint(key, shape=(l,), minval=0, maxval=10)
    y = fwd(token_ids)
    assert y.shape == (l, 10)
    assert not np.isnan(y).any()

    import random

    for _ in range(10):
        t = token_ids
        for i in random.choices(range(l), k=l // 4):
            t = t.at[i].set(random.choice(range(10)))
        y_ = fwd(t)
        assert np.isclose(y, y_).all(axis=1).mean() < 0.5
