import jax
import jax.numpy as jnp
import pymc as pm
import numpy as np
import arviz as az
import warnings
import blackjax
from typing import Callable, Optional
from blackjax.mcmc.slingshot import slingshot_warmup
from blackjax.mcmc.factories import nuts_factory

# --- THE NATIVE BLACKJAX ADAPTATIONS ---
from blackjax.adaptation.pathfinder_adaptation import pathfinder_adaptation
from blackjax.adaptation.window_adaptation import window_adaptation

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
    use_pathfinder_vi: bool = True  # <--- Now uses Native BlackJax Pathfinder!
) -> az.InferenceData:

    print("1. Compiling PyMC Model to pure XLA graph...")
    model_logp_wrapper = pymc_model.compile_fn(
        inputs=pymc_model.value_vars, outs=pymc_model.logp(), mode="JAX", point_fn=False
    )
    pure_jax_logp_fn = model_logp_wrapper.vm.jit_fn
    
    init_point = pymc_model.initial_point()
    var_names = [var.name for var in pymc_model.value_vars]
    var_shapes = [init_point[n].shape for n in var_names]
    var_sizes = [int(np.prod(s)) if len(s) > 0 else 1 for s in var_shapes]
    split_indices = tuple(np.cumsum(var_sizes)[:-1].tolist())
    
    if kernel_factory is None: kernel_factory = nuts_factory
    
    def flattened_logdensity(pos):
        splits = jnp.split(pos, split_indices)
        res = pure_jax_logp_fn(*[jnp.reshape(s, sh) for s, sh in zip(splits, var_shapes)])
        return res[0] if isinstance(res, (list, tuple)) else res

    flat_init = jnp.array(np.concatenate([init_point[n].flatten() for n in var_names]))
    
    # Symmetry breaker to prevent PyMC default saddle-point crashes
    chain_keys = jax.random.split(rng_key, num_chains)
    chain_inits = flat_init + jax.random.normal(rng_key, (num_chains, flat_init.shape[-1])) * 0.1

    initial_ladder = jnp.geomspace(base_beta, min_beta, num_rungs)

    def _run_single_ladder(chain_key, initial_pos):
        geo_key, thermo_key, sample_key = jax.random.split(chain_key, 3)
        
        # =====================================================================
        # PHASE 1: GEOMETRY INITIALIZATION (Pathfinder vs Window)
        # =====================================================================
        if use_pathfinder_vi:
            warmup = pathfinder_adaptation(blackjax.nuts, flattened_logdensity)
        else:
            warmup = window_adaptation(blackjax.nuts, flattened_logdensity)

        # Both algorithms return the exact same output structure!
        (adapted_state, adapted_params), _ = warmup.run(
            geo_key, initial_pos, num_steps=num_warmup // 2
        )
        
        geo_step_size = adapted_params["step_size"]
        geo_mass_matrix = adapted_params["inverse_mass_matrix"]
        
        # Tile the highly-optimized position across the temperature rungs
        pt_initial_positions = jnp.tile(adapted_state.position, (num_rungs, 1))

        # =====================================================================
        # PHASE 2: THERMODYNAMIC SLINGSHOT WARMUP
        # =====================================================================
        final_pt_state, _, final_mm, final_step_size = slingshot_warmup(
            rng_key=thermo_key,
            initial_positions=pt_initial_positions,
            initial_beta_ladder=initial_ladder,
            logdensity_fn=flattened_logdensity,
            kernel_factory=kernel_factory, 
            num_tuning_steps=num_warmup // 2,
            step_size=geo_step_size,
            inverse_mass_matrix=geo_mass_matrix,
            static_ladder=static_ladder,
            coupled_adaptation=coupled_adaptation 
        )
        
        # If Track Mode is on, capture the massive global matrix it learned
        if coupled_adaptation and final_mm is not None:
            geo_mass_matrix = final_mm
            geo_step_size = final_step_size
        
        # =====================================================================
        # PHASE 3: SAMPLING
        # =====================================================================
        def inner_factory(ld, **kwargs): return kernel_factory(ld, **kwargs)
        
        _, step_pt = blackjax.parallel_tempering(
            logdensity_fn=flattened_logdensity,
            inner_kernel=inner_factory, 
            inner_parameters={"step_size": geo_step_size, "inverse_mass_matrix": geo_mass_matrix}
        )
        
        def scan_fn(state, key):
            new_state, _ = step_pt(key, state)
            return new_state, new_state.inner_state.position[0]
            
        keys = jax.random.split(sample_key, num_samples)
        _, positions = jax.lax.scan(scan_fn, final_pt_state, keys)
        return positions

    # =========================================================================
    # HARDWARE POLYMORPHISM (Multi-GPU Distribution)
    # =========================================================================
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
            sharded_inits = chain_inits.reshape(num_devices, chains_per_dev, -1)
            
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
    
    trace_dict = {}
    for name, shape, samples in zip(var_names, var_shapes, split_samples):
        target_shape = (num_chains, num_samples) + shape
        trace_dict[name] = samples.reshape(target_shape)
        
    return az.from_dict(posterior=trace_dict)