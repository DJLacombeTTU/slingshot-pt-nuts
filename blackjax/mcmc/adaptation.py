import jax
import jax.numpy as jnp
from typing import NamedTuple

class MassMatrixState(NamedTuple):
    mean: jax.Array
    m2: jax.Array
    count: int

def init_mass_matrix_adapter(dim: int):
    return MassMatrixState(
        mean=jnp.zeros(dim),
        m2=jnp.zeros(dim),
        count=0
    )

def update_mass_matrix(state: MassMatrixState, position: jax.Array) -> MassMatrixState:
    """Welford's online variance algorithm for stable estimation."""
    new_count = state.count + 1
    delta = position - state.mean
    new_mean = state.mean + delta / new_count
    delta2 = position - new_mean
    new_m2 = state.m2 + delta * delta2
    return MassMatrixState(new_mean, new_m2, new_count)

def get_inverse_mass_matrix(state: MassMatrixState, regularization=1e-3) -> jax.Array:
    """Calculates the diagonal inverse mass matrix from the variance."""
    variance = state.m2 / jnp.maximum(state.count - 1, 1)
    return 1.0 / (variance + regularization)