import jax
import jax.numpy as jnp
import numpy as np
import arviz as az
import time
import matplotlib.pyplot as plt
import seaborn as sns
import blackjax

# Import your new stateless factory
from blackjax.mcmc import slingshot
from blackjax.mcmc.factories import nuts_factory

def run_native_jax_comparison():
    print("\n" + "="*60)
    print("Benchmarking: Standard BlackJax NUTS vs. Slingshot (PT-NUTS)")
    print("The 2D Rastrigin Multimodal Torture Test (Pure JAX)")
    print("="*60)
    
    # -----------------------------------------------------------------
    # 1. Define the 2D Rastrigin Target in Pure JAX
    # -----------------------------------------------------------------
    def rastrigin_logprob(position):
        """Native JAX log-density for the 2D Rastrigin function."""
        x, y = position[0], position[1]
        
        # Prior: Normal(0, 5)
        logprior = -0.5 * ((x / 5.0)**2 + (y / 5.0)**2) - jnp.log(25.0 * 2 * jnp.pi)
        
        # Energy surface
        energy = 20.0 + (x**2 - 10.0 * jnp.cos(2.0 * jnp.pi * x)) + \
                        (y**2 - 10.0 * jnp.cos(2.0 * jnp.pi * y))
                        
        return logprior - energy

    # Common Configuration
    num_chains = 4
    num_warmup = 3000
    num_samples = 6000
    rng_key = jax.random.PRNGKey(42)
    chain_keys = jax.random.split(rng_key, num_chains)

    # -----------------------------------------------------------------
    # 2. BASELINE: Standard BlackJax NUTS (Will get trapped)
    # -----------------------------------------------------------------
    print("\nStarting Standard NUTS (Baseline)...")
    
    def run_standard_nuts(key):
        init_key, warmup_key, sample_key = jax.random.split(key, 3)
        init_pos = jax.random.normal(init_key, (2,)) * 5.0
        
        # Native BlackJax Window Adaptation
        warmup = blackjax.window_adaptation(blackjax.nuts, rastrigin_logprob)
        
        # Correctly unpack the state and adapted parameters
        (state, parameters), _ = warmup.run(warmup_key, init_pos, num_steps=num_warmup)
        
        # Build the kernel using the newly tuned parameters
        kernel = blackjax.nuts(rastrigin_logprob, **parameters)
        
        def one_step(current_state, current_key):
            new_state, _ = kernel.step(current_key, current_state)
            return new_state, new_state.position
            
        keys = jax.random.split(sample_key, num_samples)
        _, positions = jax.lax.scan(one_step, state, keys)
        return positions

    t0_nuts = time.time()
    # JIT compile and map across 4 chains
    nuts_samples = jax.jit(jax.vmap(run_standard_nuts))(chain_keys)
    time_nuts = time.time() - t0_nuts
    print(f"Standard NUTS completed in {time_nuts:.2f} seconds.")

    # -----------------------------------------------------------------
    # 3. CHALLENGER: Slingshot PT-NUTS (Your Engine)
    # -----------------------------------------------------------------
    print("\nStarting Slingshot PT-NUTS...")
    num_rungs = 16
    initial_ladder = jnp.geomspace(1.0, 0.01, num_rungs)
    
    init_fn, step_fn = slingshot.build_kernel(
        logdensity_fn=rastrigin_logprob,
        kernel_factory=nuts_factory,
        inner_params={"step_size": 0.05, "inverse_mass_matrix": jnp.ones(2)},
        static_ladder=False,
        coupled_adaptation=True
    )

    def run_slingshot(key):
        init_key, sample_key = jax.random.split(key)
        # Initialize particles across all rungs
        init_pos = jax.random.normal(init_key, (num_rungs, 2)) * 5.0
        state = init_fn(init_pos, initial_ladder)
        
        def one_step(current_state, current_key):
            new_state, info = step_fn(current_key, current_state)
            # Only record the cold chain (index 0)
            return new_state, new_state.pt_state.position[0]
            
        total_steps = num_warmup + num_samples
        keys = jax.random.split(sample_key, total_steps)
        _, positions = jax.lax.scan(one_step, state, keys)
        
        # Discard warmup
        return positions[num_warmup:]

    t0_pt = time.time()
    # JIT compile and map across 4 chains
    pt_samples = jax.jit(jax.vmap(run_slingshot))(chain_keys)
    time_pt = time.time() - t0_pt
    print(f"Slingshot PT-NUTS completed in {time_pt:.2f} seconds.")

    # -----------------------------------------------------------------
    # 4. Packaging and Diagnostics
    # -----------------------------------------------------------------
    idata_nuts = az.from_dict(
        posterior={"x": np.array(nuts_samples[..., 0]), "y": np.array(nuts_samples[..., 1])}
    )
    idata_pt = az.from_dict(
        posterior={"x": np.array(pt_samples[..., 0]), "y": np.array(pt_samples[..., 1])}
    )

    print("\n--- STANDARD NUTS DIAGNOSTICS ---")
    print(az.summary(idata_nuts))
    
    print("\n--- SLINGSHOT PT-NUTS DIAGNOSTICS ---")
    print(az.summary(idata_pt))

    # -----------------------------------------------------------------
    # 5. Visualization
    # -----------------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # NUTS Plot
    sns.kdeplot(
        x=idata_nuts.posterior["x"].values.flatten(), 
        y=idata_nuts.posterior["y"].values.flatten(), 
        cmap="Blues", fill=True, thresh=0.05, ax=axes[0]
    )
    axes[0].set_title(f"Standard NUTS (Trapped)\nTime: {time_nuts:.1f}s")
    axes[0].set_xlim(-5, 5); axes[0].set_ylim(-5, 5)
    
    # PT-NUTS Plot
    sns.kdeplot(
        x=idata_pt.posterior["x"].values.flatten(), 
        y=idata_pt.posterior["y"].values.flatten(), 
        cmap="Reds", fill=True, thresh=0.05, ax=axes[1]
    )
    axes[1].set_title(f"Slingshot PT-NUTS (Full Exploration)\nTime: {time_pt:.1f}s")
    axes[1].set_xlim(-5, 5); axes[1].set_ylim(-5, 5)
    
    plt.tight_layout()
    plt.savefig("native_rastrigin_comparison.png", dpi=300)
    print("\nComparison complete. Plot saved as 'native_rastrigin_comparison.png'.")

if __name__ == "__main__":
    run_native_jax_comparison()