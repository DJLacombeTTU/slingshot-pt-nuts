import jax
import jax.numpy as jnp
import pymc as pm
import numpy as np
import arviz as az
import warnings
import blackjax
from typing import Callable, Optional

# --- IMPORT THE NEW STATELESS BLACKJAX API ---
from blackjax.mcmc.slingshot import build_kernel
from blackjax.mcmc.factories import nuts_factory
from pymc.sampling.jax import get_jaxified_logp

def sample_slingshot(
    pymc_model: pm.Model, 
    rng_key: jax.Array,
    kernel_factory: Optional[Callable] = None, 
    num_chains: int = 4,
    num_warmup: int = 3000,
    num_samples: int = 6000,
    num_rungs: int = 64, 
    base_beta: float = 1.0, 
    min_beta: float = 0.001,
    static_ladder: bool = True,
    coupled_adaptation: bool = False,
    use_pathfinder_vi: bool = True  
) -> az.InferenceData:

    print("1. Compiling PyMC Model to pure XLA graph...")
    
    # 1A. Robust Variable Extraction
    value_vars = pymc_model.value_vars
    var_names = [v.name for v in value_vars]
    
    init_point = pymc_model.initial_point()
    var_shapes = [init_point[name].shape for name in var_names]
    var_sizes = [init_point[name].size for name in var_names]
    split_indices = np.cumsum(var_sizes)[:-1]
    
    # 1B. Official PyMC JAX compilation
    raw_logp_fn = get_jaxified_logp(pymc_model)
    
    # 1C. Wrap it so BlackJax receives a flat vector, and PyMC gets its separated arrays
    def pure_jax_logp_fn(flat_array):
        split_arrays = jnp.split(flat_array, split_indices)
        reshaped_arrays = [arr.reshape(shape) for arr, shape in zip(split_arrays, var_shapes)]
        return raw_logp_fn(reshaped_arrays)
    
    pathfinder_variance = None

    if use_pathfinder_vi:
        try:
            print(" -> Running Pathfinder VI for MAP Initialization...")
            # Use the modern top-level BlackJax VI API
            pathfinder = blackjax.pathfinder(pure_jax_logp_fn)
            
            # Flatten the initial point for Pathfinder
            flat_init = jnp.concatenate([jnp.array(init_point[name]).flatten() for name in var_names])
            
            # Call .init() instead of .run() to execute the L-BFGS optimization
            state, _ = pathfinder.init(rng_key, flat_init, num_samples=1000)
            
            # Extract position and generate approximate samples to compute the variance
            base_position = state.position
            pf_samples, _ = pathfinder.sample(rng_key, state, num_samples=1000)
            pathfinder_variance = jnp.var(pf_samples, axis=0)
            
            chain_inits = jnp.tile(base_position, (num_chains, num_rungs, 1))
            noise = jax.random.normal(rng_key, chain_inits.shape) * 0.1
            chain_inits = chain_inits + noise

        except Exception as e:
            print(f" -> Pathfinder VI failed ({e}). Falling back to prior sampling.")
            use_pathfinder_vi = False 

    if not use_pathfinder_vi:
        print(" -> Generating Initial Positions from PyMC Priors...")
        base_positions = []
        for i in range(num_chains):
            prior_draw = pymc_model.initial_point(random_seed=int(rng_key[0]) + i)
            flat_prior = jnp.concatenate([jnp.array(prior_draw[name]).flatten() for name in var_names])
            base_positions.append(flat_prior)
            
        base_positions = jnp.stack(base_positions)
        chain_inits = jnp.repeat(base_positions[:, None, :], num_rungs, axis=1)
        pathfinder_variance = None

    initial_ladder = jnp.geomspace(base_beta, min_beta, num_rungs)
    
    if kernel_factory is None:
        kernel_factory = nuts_factory

    # =========================================================================
    # THE NATIVE BLACKJAX API INTEGRATION
    # =========================================================================
    flat_size = sum(var_sizes)
    inv_mass_matrix = pathfinder_variance if pathfinder_variance is not None else jnp.ones(flat_size)
    
    inner_params = {
        "step_size": 0.01,
        "inverse_mass_matrix": inv_mass_matrix
    }
    
    init_fn, step_fn = build_kernel(
        logdensity_fn=pure_jax_logp_fn,
        kernel_factory=kernel_factory,
        inner_params=inner_params,
        static_ladder=static_ladder,
        coupled_adaptation=coupled_adaptation
    )

    def _run_single_ladder(key, init_positions):
        state = init_fn(init_positions, initial_ladder)
        
        def one_step(current_state, current_key):
            new_state, info = step_fn(current_key, current_state)
            cold_chain_position = new_state.pt_state.position[0]
            return new_state, cold_chain_position
        
        total_steps = num_warmup + num_samples
        keys = jax.random.split(key, total_steps)
        final_state, position_history = jax.lax.scan(one_step, state, keys)
        return position_history[num_warmup:]

    # =========================================================================

    chain_keys = jax.random.split(rng_key, num_chains)
    num_devices = jax.device_count()
    print(f"2. Adapting Engine Geometry ({num_chains} Parallel Chains across {num_devices} device(s))...")
    
    if num_devices > 1:
        if num_chains % num_devices != 0:
            warnings.warn(f"Chains ({num_chains}) is not divisible by devices ({num_devices}). Fallback to vmap.")
            vmapped_pipeline = jax.jit(jax.vmap(_run_single_ladder))
            raw_samples_chains = vmapped_pipeline(chain_keys, chain_inits)
        else:
            chains_per_dev = num_chains // num_devices
            sharded_keys = chain_keys.reshape(num_devices, chains_per_dev, -1)
            sharded_inits = chain_inits.reshape(num_devices, chains_per_dev, -1, chain_inits.shape[-1])
            
            @jax.pmap
            def execute_on_device(keys, inits):
                return jax.vmap(_run_single_ladder)(keys, inits)
                
            sharded_samples = execute_on_device(sharded_keys, sharded_inits)
            raw_samples_chains = sharded_samples.reshape((num_chains, num_samples, -1))
    else:
        vmapped_pipeline = jax.jit(jax.vmap(_run_single_ladder))
        raw_samples_chains = vmapped_pipeline(chain_keys, chain_inits)

    print("3. Packaging into ArviZ InferenceData...")
    raw_samples_np = np.asarray(raw_samples_chains)
    split_samples = np.split(raw_samples_np, split_indices, axis=2)
    
    posterior_dict = {}
    for var_name, shape, samples in zip(var_names, var_shapes, split_samples):
        reshaped_samples = samples.reshape((num_chains, num_samples) + tuple(shape))
        posterior_dict[var_name] = reshaped_samples

    # Removed the outdated dims and coords arguments
    idata = az.from_dict(
        posterior=posterior_dict
    )
    
    return idata