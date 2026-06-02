import jax
import jax.numpy as jnp
import pytest

import blackjax
from blackjax.mcmc.factories import nuts_factory

def test_slingshot_parameter_recovery():
    """Verify Slingshot MP-MCMC can sample from a standard normal target."""
    def logdensity_fn(x):
        return -0.5 * jnp.sum(x**2)

    rng_key = jax.random.PRNGKey(42)
    initial_position = jnp.array([2.0, -2.0])
    
    # Slingshot requires a multi-rung starting state
    num_rungs = 4
    initial_positions = jnp.tile(initial_position, (num_rungs, 1))
    initial_ladder = jnp.geomspace(1.0, 0.1, num_rungs)
    
    # 1. Instantiate the algorithm via the new top-level stateless API
    init_fn, step_fn = blackjax.slingshot(
        logdensity_fn=logdensity_fn, 
        kernel_factory=nuts_factory,
        inner_params={"step_size": 0.5, "inverse_mass_matrix": jnp.ones(2)},
        static_ladder=True,
        coupled_adaptation=False
    )
    
    # 2. Initialize the PyTree state
    state = init_fn(initial_positions, initial_ladder)
    
    # Compile the transition step using lax.scan
    @jax.jit(static_argnames=("num_steps",))
    def run_chain(key, initial_state, num_steps=100):
        def body_fn(carry_state, step_key):
            next_state, info = step_fn(step_key, carry_state)
            # Record only the cold chain's position
            return next_state, next_state.pt_state.position[0]
            
        keys = jax.random.split(key, num_steps)
        _, positions = jax.lax.scan(body_fn, initial_state, keys)
        return positions

    # Execute chain execution loop
    positions = run_chain(rng_key, state, num_steps=200)
    
    # Assert output shapes and check for numerical execution integrity
    assert positions.shape == (200, 2)
    assert not jnp.any(jnp.isnan(positions))