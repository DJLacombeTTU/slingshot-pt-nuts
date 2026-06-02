import jax
import jax.numpy as jnp
from blackjax import slingshot_warmup, create_thermodynamic_mesh, shard_pytree

def target_logdensity(position):
    return -0.5 * jnp.sum(position ** 2)

# 1. SETUP HARDWARE MESH
print(f"Detected {jax.device_count()} local device(s): {jax.local_devices()}")
mesh, sharding_spec = create_thermodynamic_mesh()

# 2. INITIALIZE ARRAYS ON THE HOST
# We will use 8 rungs this time to ensure there's enough data to split across GPUs
num_rungs = 8
initial_ladder = jnp.linspace(1.0, 0.05, num_rungs)
initial_positions = jnp.zeros((num_rungs, 2))
rng_key = jax.random.PRNGKey(42)

# 3. PHYSICALLY DISTRIBUTE THE DATA
# We wrap the context in `with mesh:` to tell JAX to use our hardware grid
with jax.set_mesh(mesh):
    # Push the initial inputs onto the distributed devices
    sharded_positions = shard_pytree(initial_positions, sharding_spec)
    sharded_ladder = shard_pytree(initial_ladder, sharding_spec)
    
    print("\nPre-Compilation Memory Map:")
    print(f"Positions sit on: {sharded_positions.sharding}")
    
    print("\nStarting Distributed XLA Compilation (8 Rungs)...")
    
    # 4. EXECUTE THE DISTRIBUTED LOOP
    final_pt_state, frozen_ladder, ladder_history = jax.jit(
        slingshot_warmup, 
        static_argnames=['logdensity_fn', 'num_tuning_steps']
    )(
        rng_key, 
        sharded_positions, 
        sharded_ladder, 
        target_logdensity,
        num_tuning_steps=500
    )

print("\nDistributed Warm-up Complete!")
print(f"Final Positions physically sit on: {final_pt_state.position.sharding}")
print("\nFrozen Ladder: \n", frozen_ladder)