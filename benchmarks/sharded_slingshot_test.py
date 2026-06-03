import os
# Force JAX to ignore the GPU and use the CPU for this local simulation
os.environ["JAX_PLATFORMS"] = "cpu"
# Force JAX to slice that CPU into 4 virtual devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import time
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from blackjax.mcmc import slingshot
from blackjax.mcmc.factories import nuts_factory

def run_sharded_benchmark():
    print(f"JAX detects {jax.device_count()} devices: {jax.devices()}")
    
    def rastrigin_logprob(position):
        x, y = position[0], position[1]
        logprior = -0.5 * ((x / 5.0)**2 + (y / 5.0)**2) - jnp.log(25.0 * 2 * jnp.pi)
        energy = 20.0 + (x**2 - 10.0 * jnp.cos(2.0 * jnp.pi * x)) + \
                        (y**2 - 10.0 * jnp.cos(2.0 * jnp.pi * y))
        return logprior - energy

    num_chains = 4
    num_rungs = 16
    num_warmup = 1000
    num_samples = 2000
    rng_key = jax.random.PRNGKey(42)

    # =========================================================================
    # 1. DEFINE THE DEVICE MESH AND SHARDING STRATEGY
    # =========================================================================
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('rungs_axis',))
    
    # Ladder is 1D: (rungs,) -> Shard across devices
    rung_sharding = NamedSharding(mesh, P('rungs_axis'))
    
    # Positions are 3D: (chains, rungs, dimensions)
    # We only want to shard the middle axis!
    pos_sharding = NamedSharding(mesh, P(None, 'rungs_axis', None))

    print(f"Distributing {num_rungs} rungs for {num_chains} parallel chains across {len(devices)} devices...")

    # =========================================================================
    # 2. BUILD, VMAP, AND DISTRIBUTE THE INITIAL STATE
    # =========================================================================
    init_fn, step_fn = slingshot.build_kernel(
        logdensity_fn=rastrigin_logprob,
        kernel_factory=nuts_factory,
        inner_params={"step_size": 0.05, "inverse_mass_matrix": jnp.ones(2)},
        static_ladder=False,
        coupled_adaptation=True
    )
    
    # Vectorize the kernel to handle the `chains` dimension (axis 0)
    vmap_init = jax.vmap(init_fn, in_axes=(0, None))
    vmap_step = jax.vmap(step_fn, in_axes=(0, 0))

    init_key, sample_key = jax.random.split(rng_key)
    
    raw_ladder = jnp.geomspace(1.0, 0.01, num_rungs)
    raw_pos = jax.random.normal(init_key, (num_chains, num_rungs, 2)) * 5.0
    
    sharded_ladder = jax.device_put(raw_ladder, rung_sharding)
    sharded_pos = jax.device_put(raw_pos, pos_sharding)

    state = vmap_init(sharded_pos, sharded_ladder)

    # =========================================================================
    # 3. JIT COMPILE THE SPMD MULTI-CHAIN LOOP
    # =========================================================================
    @jax.jit
    def run_all_chains(current_state, key):
        def one_step(carry, step_key):
            # Split the step's random key into 4 unique keys for our chains
            chain_keys = jax.random.split(step_key, num_chains)
            
            new_state, info = vmap_step(chain_keys, carry)
            
            # Extract the cold chain (index 0 of the rungs) across all 4 chains
            # pt_state.position shape: (num_chains, num_rungs, dimensions)
            cold_positions = new_state.pt_state.position[:, 0, :]
            return new_state, cold_positions
            
        keys = jax.random.split(key, num_warmup + num_samples)
        final_state, positions = jax.lax.scan(one_step, current_state, keys)
        
        # positions shape out of scan: (num_steps, num_chains, dims)
        # Swap axes to match standard ArviZ (chains, steps, dims)
        return jnp.swapaxes(positions, 0, 1)

    # =========================================================================
    # 4. EXECUTE
    # =========================================================================
    print("Compiling and executing multi-chain SPMD parallel tempering...")
    t0 = time.time()
    
    positions = run_all_chains(state, sample_key)
    positions.block_until_ready() 
    
    t_total = time.time() - t0
    print(f"Sharded multi-chain execution completed in {t_total:.2f} seconds.")

    # =========================================================================
    # 5. DIAGNOSTICS & PLOTTING
    # =========================================================================
    print("\nPackaging and calculating diagnostics...")
    
    # Discard warmup
    samples = np.array(positions)[:, num_warmup:, :]
    
    # ArviZ directly accepts shape (chains, draws)
    idata = az.from_dict(
        posterior={
            "x": samples[:, :, 0], 
            "y": samples[:, :, 1]
        }
    )
    
    print("\n--- SHARDED MULTI-CHAIN PT-NUTS DIAGNOSTICS ---")
    print(az.summary(idata))
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 6))
    
    # Flatten across chains for the density plot
    sns.kdeplot(
        x=samples[:, :, 0].flatten(), 
        y=samples[:, :, 1].flatten(), 
        cmap="Reds", fill=True, thresh=0.05
    )
    plt.title(f"4-Chain Sharded PT-NUTS\nTime: {t_total:.1f}s")
    plt.xlim(-5, 5); plt.ylim(-5, 5)
    plt.tight_layout()
    plt.savefig("sharded_rastrigin_4chains.png", dpi=300)
    print("\nPlot saved as 'sharded_rastrigin_4chains.png'.")

if __name__ == "__main__":
    run_sharded_benchmark()