import jax
import jax.numpy as jnp

# Import the new BlackJax-native module
from blackjax.mcmc import slingshot
from blackjax.mcmc.factories import nuts_factory

# 1. Define a pure JAX target (e.g., a simple 2D Gaussian)
def target_logdensity(position):
    return -0.5 * jnp.sum(position ** 2)

def run_native_test():
    print("1. Initializing Native BlackJax PT-NUTS Kernel...")
    num_rungs = 8
    num_steps = 1000
    
    # Setup initial states
    initial_ladder = jnp.geomspace(1.0, 0.05, num_rungs)
    initial_positions = jnp.zeros((num_rungs, 2))
    rng_key = jax.random.PRNGKey(42)

    # 2. Build the stateless kernel (Pass configuration here)
    init_fn, step_fn = slingshot.build_kernel(
        logdensity_fn=target_logdensity,
        kernel_factory=nuts_factory,
        inner_params={"step_size": 0.01, "inverse_mass_matrix": jnp.ones(2)},
        static_ladder=False,
        coupled_adaptation=True
    )

    # 3. Initialize the Universe (Pass the positions and ladder here)
    state = init_fn(initial_positions, initial_ladder)

    print("2. Executing JAX Scan Loop...")
    
    # 4. Define the single-step driver loop
    def one_step(current_state, current_key):
        new_state, info = step_fn(current_key, current_state)
        # Record the cold chain's position
        return new_state, new_state.pt_state.position[0]

    # 5. Run the XLA compilation and execution
    keys = jax.random.split(rng_key, num_steps)
    final_state, cold_chain_history = jax.lax.scan(one_step, state, keys)

    print("\n--- Sampling Complete! ---")
    print(f"Final Cold Chain Shape: {cold_chain_history.shape}")
    print(f"Final Adapted Ladder: {final_state.tune_state}")

if __name__ == "__main__":
    run_native_test()