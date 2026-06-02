import jax
import jax.numpy as jnp
import blackjax
from typing import Callable, NamedTuple, Tuple, Any

# =============================================================================
# PURE JAX WELFORD ESTIMATOR (With Matrix Inheritance)
# =============================================================================
def init_welford(template, initial_variance=None):
    mean = template
    if initial_variance is not None:
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
# STATE & INFO PYTREES
# =============================================================================
class SlingshotState(NamedTuple):
    """The complete stateless pytree for Slingshot PT-NUTS."""
    pt_state: Any
    tune_state: Any
    mm_state: Any
    active_mm: Any
    da_state: Any

class SlingshotInfo(NamedTuple):
    """Diagnostics tracking for a single step."""
    swap_acceptance: jnp.ndarray
    is_window_end: bool


# =============================================================================
# BLACKJAX KERNEL API: INIT
# =============================================================================
def init(
    initial_positions: jax.Array,
    initial_beta_ladder: jax.Array,
    logdensity_fn: Callable,
    kernel_factory: Callable,
    inner_params: dict,
    static_ladder: bool,
    coupled_adaptation: bool
) -> SlingshotState:
    
    init_pt, _ = blackjax.parallel_tempering(
        logdensity_fn=logdensity_fn, inner_kernel=kernel_factory, inner_parameters=inner_params 
    )
    pt_state = init_pt(initial_positions, initial_beta_ladder)
    
    if not static_ladder:
        init_tune, _, _ = blackjax.parallel_tempering_adaptation()
        tune_state = init_tune(initial_beta_ladder)
    else:
        tune_state = ()  

    if coupled_adaptation:
        active_mm = inner_params.get("inverse_mass_matrix", jnp.ones_like(initial_positions[0]))
        mm_state = init_welford(initial_positions[0], initial_variance=active_mm)
        da_state = init_dual_averaging(inner_params.get("step_size", 0.01))
    else:
        mm_state = ()
        active_mm = ()
        da_state = ()
        
    return SlingshotState(
        pt_state=pt_state,
        tune_state=tune_state,
        mm_state=mm_state,
        active_mm=active_mm,
        da_state=da_state
    )

# =============================================================================
# BLACKJAX KERNEL API: STEP
# =============================================================================
def step(
    rng_key: jax.Array,
    state: SlingshotState,
    logdensity_fn: Callable,
    kernel_factory: Callable,
    inner_params: dict,
    static_ladder: bool,
    coupled_adaptation: bool
) -> Tuple[SlingshotState, SlingshotInfo]:
    
    pt_state, tune_state, mm_state, active_mm, da_state = state
    
    # 1. Inject active adaptation states into the kernel
    if coupled_adaptation:
        active_step_size = jnp.exp(da_state["log_step_size"])
        dynamic_params = inner_params.copy()
        dynamic_params["inverse_mass_matrix"] = active_mm
        dynamic_params["step_size"] = active_step_size
        
        _, dynamic_step_pt = blackjax.parallel_tempering(
            logdensity_fn=logdensity_fn, inner_kernel=kernel_factory, inner_parameters=dynamic_params
        )
    else:
        _, dynamic_step_pt = blackjax.parallel_tempering(
            logdensity_fn=logdensity_fn, inner_kernel=kernel_factory, inner_parameters=inner_params
        )

    # 2. Execute ONE independent NUTS and Swap Step
    new_pt, pt_info = dynamic_step_pt(rng_key, pt_state)
    
    is_window_end_flag = False
    
    # 3. Update Welford and Dual Averaging states based on the cold chain
    if coupled_adaptation:
        cold_accept = jnp.nan_to_num(pt_info.inner_info.acceptance_rate[0], nan=0.0)
        updated_da_state = update_dual_averaging(da_state, cold_accept)
        
        updated_mm_state = update_welford(mm_state, new_pt.position[0])
        _, _, count = updated_mm_state
        
        window_size = 200
        is_window_end = (count > 0) & (count % window_size == 0)
        is_window_end_flag = is_window_end
        
        new_active_mm = jnp.where(is_window_end, get_inverse_mass_matrix(updated_mm_state), active_mm)
        
        fresh_welford = init_welford(new_pt.position[0], initial_variance=new_active_mm)
        
        new_mm_state = jax.tree_util.tree_map(
            lambda u, f: jnp.where(is_window_end, f, u), updated_mm_state, fresh_welford
        )
        
        fresh_da = reset_dual_averaging(updated_da_state)
        new_da_state = jax.tree_util.tree_map(
            lambda u, f: jnp.where(is_window_end, f, u), updated_da_state, fresh_da
        )
    else:
        new_da_state = da_state
        new_mm_state = mm_state
        new_active_mm = active_mm
    
    # 4. Update the Thermodynamic Grid Layout
    if not static_ladder:
        _, update_tune, get_beta = blackjax.parallel_tempering_adaptation()
        new_tune = update_tune(tune_state, pt_info.swap_acceptance)
        new_beta = get_beta(new_tune)
        
        init_pt, _ = blackjax.parallel_tempering(
            logdensity_fn=logdensity_fn, inner_kernel=kernel_factory, inner_parameters=inner_params 
        )
        new_pt = init_pt(new_pt.position, new_beta)
    else:
        new_tune = tune_state

    # 5. Repackage the new universe
    new_state = SlingshotState(
        pt_state=new_pt,
        tune_state=new_tune,
        mm_state=new_mm_state,
        active_mm=new_active_mm,
        da_state=new_da_state
    )
    
    info = SlingshotInfo(
        swap_acceptance=pt_info.swap_acceptance,
        is_window_end=is_window_end_flag
    )
    
    return new_state, info

# =============================================================================
# BLACKJAX KERNEL API: FACTORY
# =============================================================================
def build_kernel(
    logdensity_fn: Callable,
    kernel_factory: Callable,
    inner_params: dict = None,
    static_ladder: bool = False,
    coupled_adaptation: bool = False
):
    """
    Builds the Slingshot PT-NUTS kernel matching the standard BlackJax API pattern.
    """
    if inner_params is None: 
        inner_params = {}
        
    def init_fn(initial_positions, initial_beta_ladder):
        return init(
            initial_positions, initial_beta_ladder, logdensity_fn, 
            kernel_factory, inner_params, static_ladder, coupled_adaptation
        )
        
    def step_fn(rng_key, state):
        return step(
            rng_key, state, logdensity_fn, kernel_factory, 
            inner_params, static_ladder, coupled_adaptation
        )
        
    return init_fn, step_fn