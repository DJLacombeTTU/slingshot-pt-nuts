from typing import NamedTuple
import jax
import jax.numpy as jnp

class PTAdaptationState(NamedTuple):
    log_beta_diffs: jax.Array  
    step_count: int            

# 1. ADD THE TUNING RATE PARAMETER HERE
def parallel_tempering_adaptation(
    target_acceptance_rate: float = 0.65, 
    decay_rate: float = 0.6, 
    tuning_rate: float = 0.05  # Slows down early massive jumps
):
    def init(initial_beta_ladder: jax.Array) -> PTAdaptationState:
        diffs = initial_beta_ladder[:-1] - initial_beta_ladder[1:]
        log_diffs = jnp.log(jnp.maximum(diffs, 1e-10))
        return PTAdaptationState(log_beta_diffs=log_diffs, step_count=1)

    def update(state: PTAdaptationState, swap_acceptance_mask: jax.Array) -> PTAdaptationState:
        t = state.step_count
        
        # 2. APPLY THE TUNING RATE TO THE COOLING SCHEDULE
        gamma_t = tuning_rate * (1.0 / (t ** decay_rate))
        
        empirical_rate = swap_acceptance_mask.astype(jnp.float32)
        rate_difference = empirical_rate - target_acceptance_rate
        
        new_log_diffs = state.log_beta_diffs + gamma_t * rate_difference
        
        return PTAdaptationState(log_beta_diffs=new_log_diffs, step_count=t + 1)

    def get_beta(state: PTAdaptationState, base_beta: float = 1.0) -> jax.Array:
        diffs = jnp.exp(state.log_beta_diffs)
        cumulative_diffs = jnp.cumsum(diffs)
        lower_rungs = base_beta - cumulative_diffs
        beta_ladder = jnp.concatenate([jnp.array([base_beta]), lower_rungs])
        return jnp.maximum(beta_ladder, 1e-10)

    return init, update, get_beta