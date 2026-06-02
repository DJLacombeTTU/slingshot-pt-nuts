import os
import jax

# =========================================================================
# CRITICAL FIX: UNLOCK 64-BIT PRECISION FOR BAYESIAN COMPUTATION
# =========================================================================
jax.config.update("jax_enable_x64", True)

import pymc as pm
import numpy as np
import arviz as az

from blackjax.mcmc.pymc_bridge import sample_slingshot

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def build_toy_model():
    """A simple 2D bimodal distribution."""
    print("Building 2D Bimodal Toy Model...")
    with pm.Model() as model:
        w = pm.Dirichlet('w', a=np.ones(2))
        
        import pymc.distributions.transforms as tr
        
        # --- THE FIX: Provide an ordered initval so PyMC doesn't calculate log(0) ---
        mu = pm.Normal('mu', mu=0, sigma=10, shape=2, transform=tr.ordered, initval=np.array([-2.0, 2.0]))
        
        y_data = np.concatenate([
            np.random.normal(-5, 1, 50), 
            np.random.normal(5, 1, 50)
        ])
        
        pm.NormalMixture('y_obs', w=w, mu=mu, sigma=np.array([1.0, 1.0]), observed=y_data)
        
    return model

if __name__ == "__main__":
    toy_model = build_toy_model()
    
    print("\nStarting Slingshot (Toy Configuration)...")
    
    idata = sample_slingshot(
        pymc_model=toy_model,
        rng_key=jax.random.PRNGKey(123),
        num_chains=4,
        num_warmup=3000,         
        num_samples=3000,        
        num_rungs=32,             
        coupled_adaptation=True, 
        use_pathfinder_vi=True  
    )
    
    print("\n" + "="*50)
    print("TOY MODEL RESULTS (ArviZ Summary)")
    print("="*50)
    print(az.summary(idata))