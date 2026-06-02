import jax
import pymc as pm
import numpy as np
import arviz as az
from blackjax import sample_slingshot

# =============================================================================
# 1. GENERATE SYNTHETIC ECONOMETRIC DATA
# =============================================================================
np.random.seed(42)
N = 250

log_income_raw = np.random.normal(0, 1.5, size=N)
true_alpha = 1.5
true_beta = 0.8
true_sigma = 1.0

y_raw = true_alpha + true_beta * log_income_raw + np.random.normal(0, true_sigma, size=N)

# THE FIX: Standardize the data so the posterior geometry matches our Identity Mass Matrix!
log_income = (log_income_raw - np.mean(log_income_raw)) / np.std(log_income_raw)
y = (y_raw - np.mean(y_raw)) / np.std(y_raw)

# =============================================================================
# 2. DEFINE THE PYMC REGRESSION
# =============================================================================
with pm.Model() as reg_model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)
    
    mu = alpha + beta * log_income
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

# =============================================================================
# 3. ENGAGE SLINGSHOT
# =============================================================================
rng_key = jax.random.PRNGKey(99)

idata = sample_slingshot(
    pymc_model=reg_model,
    rng_key=rng_key,
    num_chains=4,
    num_warmup=1000,
    num_samples=2000,
    num_rungs=4,
    step_size=5e-3  # A balanced step size for standardized geometry
)

# =============================================================================
# 4. VIEW THE RESULTS
# =============================================================================
print("\n=== SLINGSHOT POSTERIOR SUMMARY ===")
print(az.summary(idata))