import os
import multiprocessing

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================
# 1. Stop JAX from aggressively pre-allocating 90% of GPU memory per process
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# 2. Force Python to use 'spawn' to prevent OS-level deadlocks with SMC
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass 
# =============================================================================

import jax
import pymc as pm
import numpy as np
import arviz as az
import time
import matplotlib.pyplot as plt
import seaborn as sns
from blackjax.mcmc.pymc_bridge import sample_slingshot

def run_high_dim_benchmark():
    n_dims = 40
    print("\n" + "="*85)
    print(f"Benchmarking: Sequential Monte Carlo (SMC) vs. Slingshot (PT)")
    print(f"The {n_dims}D High-Dimensional Rastrigin Multimodal Stress Test")
    print("="*85)
    
    # -----------------------------------------------------------------
    # Define the N-Dimensional Rastrigin Model using PyTensor Vectors
    # -----------------------------------------------------------------
    with pm.Model() as model_40d:
        # A vectorized continuous parameter space spanning 40 dimensions
        theta = pm.Normal("theta", mu=0.0, sigma=5.0, shape=n_dims)
        
        # Vectorized math: pm.math.cos and squaring operate element-wise.
        # Summing elements across the axis builds the total non-convex energy surface.
        energy = (10.0 * n_dims) + pm.math.sum(
            theta**2 - 10.0 * pm.math.cos(2.0 * np.pi * theta)
        )
        pm.Potential("rastrigin_potential", -energy)

    # Initialize tracking variables to guarantee safe scoping for the summary table
    idata_smc = None
    time_smc = np.nan
    max_rhat_smc, min_ess_smc = np.nan, np.nan
    global_mean_smc = np.nan

    # -----------------------------------------------------------------
    # EXPERIMENT 1: Sequential Monte Carlo (SMC)
    # -----------------------------------------------------------------
    print(f"\nRunning Experiment 1: Adaptive SMC in {n_dims} Dimensions...")
    print("Evaluating particle swarm dynamics against high-dimensional volume...")
    start_smc = time.time()
    try:
        idata_smc = pm.sample_smc(
            draws=4000, chains=4, random_seed=42,
            model=model_40d, progressbar=False
        )
        time_smc = time.time() - start_smc
        
        summary_smc = az.summary(idata_smc, var_names=["theta"])
        max_rhat_smc = summary_smc["r_hat"].max()
        min_ess_smc = summary_smc["ess_bulk"].min()
        
        # Calculate the absolute average distance from the true global origin (0,0,...,0)
        mean_coords = idata_smc.posterior["theta"].mean(dim=["chain", "draw"]).values
        global_mean_smc = np.mean(np.abs(mean_coords))
    except Exception as e:
        print(f"SMC sampler failed or encountered degeneracy: {e}")

    # -----------------------------------------------------------------
    # EXPERIMENT 2: Multi-Temperature Slingshot Engine
    # -----------------------------------------------------------------
    print(f"\nRunning Experiment 2: Slingshot (PT) in {n_dims} Dimensions...")
    print("Deploying gradient-driven NUTS chains across parallel XLA temperature rungs...")
    
    idata_sling = None
    time_slingshot = np.nan
    max_rhat_sling, min_ess_sling = np.nan, np.nan
    global_mean_sling = np.nan
    
    start_slingshot = time.time()
    try:
        # Generate the JAX PRNG Key
        rng_key = jax.random.PRNGKey(42)
        
        # Modernized API call with the explicit static_ladder lock
        idata_sling = sample_slingshot(
            pymc_model=model_40d,
            rng_key=rng_key,
            num_chains=4,
            num_warmup=3000,
            num_samples=6000,
            num_rungs=64,
            min_beta=0.001,
            static_ladder=True  # <--- CRITICAL FIX: High-Dimensional Override
        )
        time_slingshot = time.time() - start_slingshot
        
        summary_sling = az.summary(idata_sling, var_names=["theta"])
        max_rhat_sling = summary_sling["r_hat"].max()
        min_ess_sling = summary_sling["ess_bulk"].min()
        
        mean_coords_sling = idata_sling.posterior["theta"].mean(dim=["chain", "draw"]).values
        global_mean_sling = np.mean(np.abs(mean_coords_sling))
    except Exception as e:
        print(f"Slingshot engine failed to execute: {e}")
        return

    # -----------------------------------------------------------------
    # PRINT HIGH-DIMENSIONAL RESULTS TABLE
    # -----------------------------------------------------------------
    print("\n" + "="*85)
    print("                     HIGH-DIMENSIONAL BENCHMARK RESULTS                      ")
    print("="*85)
    print(f"Engine          | Runtime  | Avg Abs Coordinate Error | Max R_hat | Min Bulk ESS")
    print("-" * 85)
    
    smc_time_str = f"{time_smc:.2f}s" if not np.isnan(time_smc) else "N/A"
    smc_err_str = f"{global_mean_smc:.4f}" if not np.isnan(global_mean_smc) else "N/A"
    smc_rhat_str = f"{max_rhat_smc:.3f}" if not np.isnan(max_rhat_smc) else "N/A"
    smc_ess_str = f"{min_ess_smc:.1f}" if not np.isnan(min_ess_smc) else "N/A"

    sling_time_str = f"{time_slingshot:.2f}s"
    sling_err_str = f"{global_mean_sling:.4f}"
    sling_rhat_str = f"{max_rhat_sling:.3f}"
    sling_ess_str = f"{min_ess_sling:.1f}"

    print(f"Adaptive SMC    | {smc_time_str:<8} | {smc_err_str:<24} | {smc_rhat_str:<9} | {smc_ess_str:<12}")
    print(f"Slingshot (PT)  | {sling_time_str:<8} | {sling_err_str:<24} | {sling_rhat_str:<9} | {sling_ess_str:<12}")
    print("="*85)
    print("True Global Minimum Coordinate Value is exactly 0.0000 across all 40 axes.\n")

    # -----------------------------------------------------------------
    # VISUALIZATION: DIAGNOSING DEGENERACY ACROSS PARAMETER AXES
    # -----------------------------------------------------------------
    if idata_smc is not None and idata_sling is not None:
        print("Generating diagnostic comparison plots across dimensions...")
        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            dimensions = np.arange(n_dims)
            
            # Subplot 1: Convergence Check (R_hat across dimensions)
            axes[0].plot(dimensions, summary_smc["r_hat"], 'o--', color='blue', label='Adaptive SMC', alpha=0.7)
            axes[0].plot(dimensions, summary_sling["r_hat"], 's-', color='red', label='Slingshot (PT)', alpha=0.7)
            axes[0].axhline(1.05, color='black', linestyle=':', label='Convergence Threshold (1.05)')
            axes[0].set_ylabel(r"Potential Scale Reduction ($\hat{R}$)")
            axes[0].set_title(f"Convergence Performance Across All {n_dims} Dimensions")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Subplot 2: Effective Sample Size (ESS Bulk across dimensions)
            axes[1].plot(dimensions, summary_smc["ess_bulk"], 'o--', color='blue', label='Adaptive SMC', alpha=0.7)
            axes[1].plot(dimensions, summary_sling["ess_bulk"], 's-', color='red', label='Slingshot (PT)', alpha=0.7)
            axes[1].set_ylabel("Bulk Effective Sample Size (ESS)")
            axes[1].set_xlabel("Dimension Index (0 to 39)")
            axes[1].set_title("Sample Diversity (ESS) Across All Dimensions")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig.savefig("high_dim_convergence_diagnostic.png", dpi=300)
            plt.close(fig)
            print(" -> Diagnostic plot saved as 'high_dim_convergence_diagnostic.png'.")
            
        except Exception as e:
            print(f" -> Could not generate diagnostic graphics: {e}")

if __name__ == "__main__":
    run_high_dim_benchmark()