from typing import Callable, NamedTuple, Any
import jax
import jax.numpy as jnp

# =============================================================================
# 1. IMMUTABLE STATE AND INFO PYTREES
# =============================================================================
class ParallelTemperingState(NamedTuple):
    """Encapsulates the sharded state of the thermodynamic ladder."""
    position: jax.Array              # Shape: (N, D) - Parameter positions across rungs
    logdensity: jax.Array            # Shape: (N,)   - UNTEMPERED log-probability densities
    inner_state: Any                 # Pytree         - Sharded inner sampler states
    beta: jax.Array                  # Shape: (N,)   - Active inverse temperature ladder

class ParallelTemperingInfo(NamedTuple):
    """Diagnostic information for the Parallel Tempering transition."""
    inner_info: Any                  # Nested diagnostic info from the inner kernels
    swap_acceptance: jax.Array       # Shape: (N-1,) - Boolean indicators of swap success


# =============================================================================
# 2. THE FACTORY INTERFACE
# =============================================================================
def parallel_tempering(
    logdensity_fn: Callable[[jax.Array], float],
    inner_kernel: Any,  # A BlackJax kernel factory (e.g., blackjax.nuts)
    inner_parameters: dict, # Hyperparameters for the inner kernel
):
    """Factory that builds a pure, framework-agnostic Parallel Tempering kernel."""

    # -------------------------------------------------------------------------
    # COMPONENT 2: THE STATE INITIALIZATION FUNCTION
    # -------------------------------------------------------------------------
    def init(initial_positions: jax.Array, beta: jax.Array) -> ParallelTemperingState:
        """Initializes the collective state across all temperature rungs."""
        
        # Helper function for a single rung to enforce a unary log-density signature
        def one_rung_init(pos, beta_val):
            def local_logdensity(p):
                return beta_val * logdensity_fn(p)
            kernel_instance = inner_kernel(local_logdensity, **inner_parameters)
            return kernel_instance.init(pos)
        
        # Vectorize initialization across all temperature rungs simultaneously
        sharded_inner_states = jax.vmap(one_rung_init)(initial_positions, beta)
        
        # Compute baseline untempered log-densities across the input coordinates
        base_logdensities = jax.vmap(logdensity_fn)(initial_positions)
        
        return ParallelTemperingState(
            position=initial_positions,
            logdensity=base_logdensities,
            inner_state=sharded_inner_states,
            beta=beta
        )

    # -------------------------------------------------------------------------
    # COMPONENT 3: THE STEP FUNCTION (THE EXECUTION CORE)
    # -------------------------------------------------------------------------
    def step(rng_key: jax.Array, state: ParallelTemperingState) -> tuple[ParallelTemperingState, ParallelTemperingInfo]:
        """Executes one complete iteration: Local Mutation followed by Global Swap."""
        key_mutation, key_swap = jax.random.split(rng_key)
        num_rungs = state.beta.shape[0]
        
        # --- STAGE 1: LOCAL MUTATION (Parallel Trajectories) ---
        mutation_keys = jax.random.split(key_mutation, num_rungs)
        
        # Helper function for a single rung mutation step
        def one_rung_step(key, inner_state, beta_val):
            def local_logdensity(p):
                return beta_val * logdensity_fn(p)
            kernel_instance = inner_kernel(local_logdensity, **inner_parameters)
            return kernel_instance.step(key, inner_state)
        
        # Vectorize the mutation step across all parallel chains
        new_inner_states, inner_info = jax.vmap(one_rung_step)(mutation_keys, state.inner_state, state.beta)
        
        # Extract post-mutation coordinates and update base log-densities
        new_positions = new_inner_states.position
        new_base_logdensities = jax.vmap(logdensity_fn)(new_positions)
        
        # --- STAGE 2: GLOBAL SWAP (Vectorized Even/Odd Exchange Gates) ---
        key_even, key_odd = jax.random.split(key_swap)
        
        def execute_swap_phase(rng_key, current_positions, current_logdensities, current_inner, is_even):
            start_idx = 0 if is_even else 1
            idx_i = jnp.arange(start_idx, num_rungs - 1, 2)
            idx_j = idx_i + 1
            
            # Guard against empty slices
            if idx_i.shape[0] == 0:
                # Return empty arrays of the correct dimension for the mask/indices
                empty_mask = jnp.zeros((0,), dtype=bool)
                return current_positions, current_logdensities, current_inner, empty_mask, idx_i
            
            # Extract adjacent pairs
            pos_i, pos_j = current_positions[idx_i], current_positions[idx_j]
            logdeb_i, logdeb_j = current_logdensities[idx_i], current_logdensities[idx_j]
            beta_i, beta_j = state.beta[idx_i], state.beta[idx_j]
            
            # Calculate Metropolis-Hastings swap acceptance probability
            delta_beta = beta_i - beta_j
            delta_logdensity = logdeb_j - logdeb_i
            swap_log_prob = delta_beta * delta_logdensity
            
            random_draws = jax.random.uniform(rng_key, shape=(idx_i.shape[0],))
            accept_mask = jnp.log(random_draws) < swap_log_prob
            
            # Expand mask dimensions to broadcast cleanly across position arrays
            mask_expanded = jnp.expand_dims(accept_mask, axis=-1)
            
            # Calculate conditional swaps
            swapped_pos_i = jnp.where(mask_expanded, pos_j, pos_i)
            swapped_pos_j = jnp.where(mask_expanded, pos_i, pos_j)
            
            swapped_log_i = jnp.where(accept_mask, logdeb_j, logdeb_i)
            swapped_log_j = jnp.where(accept_mask, logdeb_i, logdeb_j)
            
            # Inject updates back into the global arrays using XLA-compatible scatter methods
            updated_positions = current_positions.at[idx_i].set(swapped_pos_i).at[idx_j].set(swapped_pos_j)
            updated_logdensities = current_logdensities.at[idx_i].set(swapped_log_i).at[idx_j].set(swapped_log_j)
            
            # Dynamically reshape the boolean mask to broadcast over arbitrary inner kernel leaves
            def swap_inner_tree(tree_i, tree_j):
                def leaf_swap(l_i, l_j):
                    broadcast_shape = [-1] + [1] * (l_i.ndim - 1)
                    m = accept_mask.reshape(broadcast_shape)
                    return jnp.where(m, l_j, l_i)
                
                swapped_tree_i = jax.tree_util.tree_map(leaf_swap, tree_i, tree_j)
                swapped_tree_j = jax.tree_util.tree_map(leaf_swap, tree_j, tree_i)
                return swapped_tree_i, swapped_tree_j
            
            # Slice, swap, and re-insert the inner kernel sub-pytrees
            inner_i = jax.tree_util.tree_map(lambda x: x[idx_i], current_inner)
            inner_j = jax.tree_util.tree_map(lambda x: x[idx_j], current_inner)
            new_inner_i, new_inner_j = swap_inner_tree(inner_i, inner_j)
            
            updated_inner = jax.tree_util.tree_map(
                lambda global_arr, arr_i, arr_j: global_arr.at[idx_i].set(arr_i).at[idx_j].set(arr_j),
                current_inner, new_inner_i, new_inner_j
            )
            
            return updated_positions, updated_logdensities, updated_inner, accept_mask, idx_i

        # Initialize a global array to track which swaps succeeded across all gaps
        global_swap_record = jnp.zeros(num_rungs - 1, dtype=bool)

        # Execute Phase 1: Even-indexed rungs swap upward (0 <-> 1, 2 <-> 3)
        pos_after_even, log_after_even, inner_after_even, mask_even, idx_even = execute_swap_phase(
            key_even, new_positions, new_base_logdensities, new_inner_states, is_even=True
        )
        global_swap_record = global_swap_record.at[idx_even].set(mask_even)
        
        # Execute Phase 2: Odd-indexed rungs swap upward (1 <-> 2, 3 <-> 4)
        final_positions, final_logdensities, final_inner, mask_odd, idx_odd = execute_swap_phase(
            key_odd, pos_after_even, log_after_even, inner_after_even, is_even=False
        )
        if idx_odd.shape[0] > 0:
            global_swap_record = global_swap_record.at[idx_odd].set(mask_odd)
        
        # Synchronize coordinates inside the final structural inner sampler state
        final_inner_sampler_state = final_inner._replace(position=final_positions)

        final_state = ParallelTemperingState(
            position=final_positions,
            logdensity=final_logdensities,
            inner_state=final_inner_sampler_state,
            beta=state.beta
        )
        
        info = ParallelTemperingInfo(
            inner_info=inner_info,
            swap_acceptance=global_swap_record
        )

        return final_state, info

    return init, step