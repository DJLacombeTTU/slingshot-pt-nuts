import jax
import jax.numpy as jnp
import blackjax
from typing import Callable, Any

# =============================================================================
# PURE JAX WELFORD ESTIMATOR (With Matrix Inheritance)
# =============================================================================
def init_welford(template, initial_variance=None):
    mean = template
    if initial_variance is not None:
        # Seed the accumulator with 100 "ghost steps" of the Pathfinder matrix
        # so it never forgets the global geometry!
        count = jnp.array(100, dtype=jnp.int32)
        m2 = initial_variance * 99.0
    else:
        count = jnp.array(0, dtype=jnp.int32)
        m2 = jnp.zeros_like(template)
    return mean, m2, count

def update_welford(state, value):
    mean, m2, count = state
    count = count + 1
    delta = value - mean
    mean = mean + delta / count
    delta2 = value - mean
    m2 = m2 + delta * delta2
    return mean, m2, count

def get_inverse_mass_matrix(state, regularizer=1e-3):
    mean, m2, count = state
    var = jnp.where(count > 1, m2 / (count - 1), jnp.ones_like(mean))
    return var + regularizer

# =============================================================================
# PURE JAX DUAL AVERAGING (Armored)
# =============================================================================
def init_dual_averaging(initial_step_size=0.01):
    return {
        "log_step_size": jnp.log(initial_step_size),
        "log_step_size_avg": jnp.log(initial_step_size),
        "t": 0.0,
        "mu": jnp.log(10.0 * initial_step_size),
        "error_sum": 0.0
    }

def update_dual_averaging(state, acceptance_rate, target_rate=0.8):
    gamma = 0.05
    t0 = 10.0
    kappa = 0.75
    
    t = state["t"] + 1.0
    error = target_rate - acceptance_rate
    error_sum = state["error_sum"] + error
    
    log_step_size = state["mu"] - (error_sum * jnp.sqrt(t)) / (gamma * (t + t0))
    log_step_size = jnp.clip(log_step_size, jnp.log(1e-5), jnp.log(10.0))
    
    eta = t ** -kappa
    log_step_size_avg = eta * log_step_size + (1.0 - eta) * state["log_step_size_avg"]
    
    return {
        "log_step_size": log_step_size,
        "log_step_size_avg": log_step_size_avg,
        "t": t,
        "mu": state["mu"],
        "error_sum": error_sum
    }

def reset_dual_averaging(state):
    current_step_size = jnp.exp(state["log_step_size"])
    return {
        "log_step_size": state["log_step_size"],
        "log_step_size_avg": state["log_step_size_avg"],
        "t": 0.0,
        "mu": jnp.log(10.0 * current_step_size),
        "error_sum": 0.0
    }

# =============================================================================
# SLINGSHOT ENGINE
# =============================================================================
def slingshot_warmup(
    rng_key: jax.Array,
    initial_positions: jax.Array,
    initial_beta_ladder: jax.Array,
    logdensity_fn: Callable,
    kernel_factory: Callable, 
    num_tuning_steps: int = 1000,
    inner_params: dict = None,
    static_ladder: bool = False,
    coupled_adaptation: bool = False,
    **kwargs
):
    if inner_params is None: inner_params = {}
    inner_params.update(kwargs)

    init_pt, step_pt = blackjax.parallel_tempering(
        logdensity_fn=logdensity_fn, inner_kernel=kernel_factory, inner_parameters=inner_params 
    )
    pt_state = init_pt(initial_positions, initial_beta_ladder)
    
    if not static_ladder:
        init_tune, update_tune, get_beta = blackjax.parallel_tempering_adaptation()
        tune_state = init_tune(initial_beta_ladder)
    else:
        tune_state = ()  

    if coupled_adaptation:
        # NEW: Extract the Pathfinder matrix and feed it directly into Welford!
        active_mm = inner_params.get("inverse_mass_matrix", jnp.ones_like(initial_positions[0]))
        mm_state = init_welford(initial_positions[0], initial_variance=active_mm)
        da_state = init_dual_averaging(inner_params.get("step_size", 0.01))
    else:
        mm_state = (); active_mm = (); da_state = ()
    
    def _warmup_step(carry, step_key):
        pt_state, tune_state, mm_state, active_mm, da_state = carry
        
        if coupled_adaptation:
            active_step_size = jnp.exp(da_state["log_step_size"])
            dynamic_params = inner_params.copy()
            dynamic_params["inverse_mass_matrix"] = active_mm
            dynamic_params["step_size"] = active_step_size
            
            _, dynamic_step_pt = blackjax.parallel_tempering(
                logdensity_fn=logdensity_fn, inner_kernel=kernel_factory, inner_parameters=dynamic_params
            )
        else:
            dynamic_step_pt = step_pt

        new_pt, pt_info = dynamic_step_pt(step_key, pt_state)
        
        if coupled_adaptation:
            cold_accept = jnp.nan_to_num(pt_info.inner_info.acceptance_rate[0], nan=0.0)
            updated_da_state = update_dual_averaging(da_state, cold_accept)
            
            updated_mm_state = update_welford(mm_state, new_pt.position[0])
            _, _, count = updated_mm_state
            
            window_size = 200
            is_window_end = (count > 0) & (count % window_size == 0)
            
            new_active_mm = jnp.where(is_window_end, get_inverse_mass_matrix(updated_mm_state), active_mm)
            
            # THE FIX: When resetting Welford, seed it with the CURRENT position, not the time-traveled start position!
            fresh_welford = init_welford(new_pt.position[0], initial_variance=new_active_mm)
            
            new_mm_state = jax.tree_util.tree_map(
                lambda u, f: jnp.where(is_window_end, f, u), updated_mm_state, fresh_welford
            )
            
            fresh_da = reset_dual_averaging(updated_da_state)
            new_da_state = jax.tree_util.tree_map(
                lambda u, f: jnp.where(is_window_end, f, u), updated_da_state, fresh_da
            )
        else:
            new_da_state = da_state; new_mm_state = mm_state; new_active_mm = active_mm
        
        if not static_ladder:
            new_tune = update_tune(tune_state, pt_info.swap_acceptance)
            new_beta = get_beta(new_tune)
            new_pt = init_pt(new_pt.position, new_beta)
        else:
            new_tune = tune_state; new_beta = pt_state.beta
            
        return (new_pt, new_tune, new_mm_state, new_active_mm, new_da_state), new_beta

    final_carry, beta_history = jax.lax.scan(
        _warmup_step, (pt_state, tune_state, mm_state, active_mm, da_state), jax.random.split(rng_key, num_tuning_steps)
    )
    
    final_pt = final_carry[0]
    final_beta = get_beta(final_carry[1]) if not static_ladder else initial_beta_ladder

    if coupled_adaptation:
        final_mm = final_carry[3]
        final_step_size = jnp.exp(final_carry[4]["log_step_size_avg"])
    else:
        final_mm = None; final_step_size = None
    
    return final_pt, final_beta, final_mm, final_step_size