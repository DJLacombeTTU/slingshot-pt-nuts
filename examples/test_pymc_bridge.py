import jax
import jax.numpy as jnp
import pymc as pm
import numpy as np
from blackjax import build_slingshot_from_pymc

# =============================================================================
# 1. GENERATE SYNTHETIC DATA
# =============================================================================
np.random.seed(42)
N = 100

# Synthetic explanatory variable and true parameters
log_income = np.random.normal(10, 1.5, size=N)
true_alpha = 1.5
true_beta = 0.8
true_sigma = 1.0

# True Data Generating Process
y = true_alpha + true_beta * log_income + np.random.normal(0, true_sigma, size=N)

print("1. Building PyMC Model...")
# =============================================================================
# 2. DEFINE THE PYMC MODEL
# =============================================================================
with pm.Model() as toy_model:
    # Priors
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)
    
    # Likelihood
    mu = alpha + beta * log_income
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

print("2. Bridging PyMC to Slingshot XLA Engine...")
# =============================================================================
# 3. ENGAGE THE BRIDGE
# =============================================================================
# This extracts the JAX graph and structures it for our parallel tempering kernel
sample_warmup, initial_ladder = build_slingshot_from_pymc(
    toy_model, 
    num_rungs=4, 
    base_beta=1.0, 
    min_beta=0.01
)

print("3. Starting XLA Compilation and Sampling (1,000 Steps)...")
# =============================================================================
# 4. EXECUTE THE ACCELERATED SAMPLER
# =============================================================================
rng_key = jax.random.PRNGKey(42)

# JIT compile the bridged warmup function
compiled_sampler = jax.jit(sample_warmup, static_argnames=['num_steps'])

# Fire the XLA graph
final_pt_state, frozen_ladder, ladder_history = compiled_sampler(rng_key, num_steps=1000)

print("\n--- Sampling Complete! ---")
print(f"Initial Ladder: {initial_ladder}")
print(f"Frozen Ladder:  {frozen_ladder}")

# The final positions shape will be (Num Rungs, Total Dimensions)
print(f"\nFinal Parameter Matrix Shape: {final_pt_state.position.shape}")
print("Cold Chain (Rung 0) Final Position [alpha, beta, sigma_unconstrained]:")
print(final_pt_state.position[0])