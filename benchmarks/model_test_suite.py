import os
import multiprocessing
import jax
import pymc as pm
import numpy as np
import arviz as az

from blackjax.mcmc.pymc_bridge import sample_slingshot

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass 

# Unlock 64-bit precision for Bayesian Computation
jax.config.update("jax_enable_x64", True)

# =============================================================================
# ENGINE CONFIGURATION (PR STANDARD)
# =============================================================================
def get_slingshot_config(model, rng_key, coupled=False, use_pathfinder=True, static_ladder=True):
    return {
        "pymc_model": model,
        "rng_key": rng_key,
        "num_chains": 4,
        "num_warmup": 3000,
        "num_samples": 6000,
        "num_rungs": 64,
        "min_beta": 0.001,
        "static_ladder": static_ladder,
        "coupled_adaptation": coupled,
        "use_pathfinder_vi": use_pathfinder
    }

# =============================================================================
# BENCHMARK 1: ROSENBROCK TWISTED BANANA (Curvature Stress Test)
# =============================================================================
def run_rosenbrock():
    print("\n" + "="*70); print("Benchmarking 1: Rosenbrock Twisted Banana"); print("="*70)
    with pm.Model() as model:
        theta = pm.Flat("theta", shape=10)
        pm.Potential("rosenbrock", -pm.math.sum(100.0 * (theta[1:] - theta[:-1]**2)**2 + (1.0 - theta[:-1])**2))
    
    idata = sample_slingshot(**get_slingshot_config(model, jax.random.PRNGKey(501)))
    print(az.summary(idata))

# =============================================================================
# BENCHMARK 2: NEAL'S FUNNEL (Scale Shift Stress Test)
# =============================================================================
def run_neals_funnel():
    print("\n" + "="*70); print("Benchmarking 2: Neal's Funnel (Non-Centered)"); print("="*70)
    with pm.Model() as model:
        v = pm.Normal("v", 0, 3.0)
        x = pm.Deterministic("x", pm.Normal("x_raw", 0, 1.0, shape=9) * pm.math.exp(v / 2.0))
    
    idata = sample_slingshot(**get_slingshot_config(model, jax.random.PRNGKey(301)))
    print(az.summary(idata))

# =============================================================================
# BENCHMARK 3: WIDELY SEPARATED GMM (Desert Crossing Test)
# =============================================================================
def run_separated_gmm():
    print("\n" + "="*70); print("Benchmarking 3: Widely Separated GMM"); print("="*70)
    np.random.seed(901)
    y_data = np.concatenate([np.random.normal(-10, 1, 100), np.random.normal(10, 1, 100)])
    
    with pm.Model() as model:
        w = pm.Dirichlet('w', a=np.ones(2))
        import pymc.distributions.transforms as tr
        mu = pm.Normal('mu', mu=0, sigma=15, shape=2, transform=tr.ordered, initval=np.array([-10.0, 10.0]))
        
        pm.Deterministic("True_Cluster_Centers", mu)
        pm.Deterministic("True_Mixture_Weights", w)
        pm.NormalMixture('y_obs', w=w, mu=mu, sigma=np.array([1.0, 1.0]), observed=y_data)
        
    idata = sample_slingshot(**get_slingshot_config(model, jax.random.PRNGKey(902), coupled=True))
    print(az.summary(idata))

# =============================================================================
# BENCHMARK 4: 15D RASTRIGIN (Global Multimodality Test)
# =============================================================================
def run_15d_rastrigin():
    print("\n" + "="*70); print("Benchmarking 4: 15D Rastrigin (The PR Standard)"); print("="*70)
    n_dims = 15
    with pm.Model() as model:
        theta = pm.Normal("theta", mu=0.0, sigma=5.0, shape=n_dims)
        pm.Potential("rastrigin_landscape", -(10.0 * n_dims + pm.math.sum(theta**2 - 10.0 * pm.math.cos(2 * np.pi * theta))))
    
    # Pathfinder OFF (prevents variance collapse), Static Ladder OFF (repairs 15D gaps)
    idata = sample_slingshot(**get_slingshot_config(
        model, jax.random.PRNGKey(801), coupled=True, static_ladder=False, use_pathfinder=False
    ))
    print(az.summary(idata))


# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("INITIALIZING SLINGSHOT PR SHOWCASE SUITE...")
    run_rosenbrock()         
    run_neals_funnel()       
    run_separated_gmm()      
    run_15d_rastrigin()      
    