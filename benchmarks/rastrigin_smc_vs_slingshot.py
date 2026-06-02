import pymc as pm
import numpy as np
import arviz as az
import time
import matplotlib.pyplot as plt
import seaborn as sns
from blackjax.mcmc.pymc_bridge import sample_slingshot

def run_smc_vs_slingshot_comparison():
    print("\n" + "="*60)
    print("Benchmarking: Sequential Monte Carlo (SMC) vs. Slingshot (PT)")
    print("The 2D Rastrigin Multimodal Torture Test")
    print("="*60)
    
    # -----------------------------------------------------------------
    # Define the 2D Rastrigin Model
    # -----------------------------------------------------------------
    with pm.Model() as rastrigin_model:
        # Wide priors to allow broad exploration space
        x = pm.Normal("x", mu=0.0, sigma=5.0)
        y = pm.Normal("y", mu=0.0, sigma=5.0)
        
        # Mathematical definition of the Rastrigin energy surface
        energy = 20.0 + (x**2 - 10.0 * pm.math.cos(2.0 * np.pi * x)) \
                      + (y**2 - 10.0 * pm.math.cos(2.0 * np.pi * y))
        
        pm.Potential("rastrigin_potential", -energy)

    # Initialize tracking variables to guarantee safe scoping for the table
    idata_smc = None
    time_smc, rhat_x_smc, ess_x_smc = np.nan, np.nan, np.nan
    mean_x_smc, mean_y_smc = np.nan, np.nan

    # -----------------------------------------------------------------
    # EXPERIMENT 1: Sequential Monte Carlo (SMC)
    # -----------------------------------------------------------------
    print("\nRunning Experiment 1: Sequential Monte Carlo (SMC)...")
    print("Simulating particles moving through an adaptive temperature schedule...")
    start_smc = time.time()
    try:
        # PyMC v5 API uses pm.sample_smc directly
        idata_smc = pm.sample_smc(
            draws=1000, chains=4, random_seed=42,
            model=rastrigin_model, progressbar=False
        )
        time_smc = time.time() - start_smc
        summary_smc = az.summary(idata_smc, var_names=["x", "y"])
        rhat_x_smc = summary_smc.loc["x", "r_hat"]
        ess_x_smc = summary_smc.loc["x", "ess_bulk"]
        mean_x_smc = idata_smc.posterior["x"].mean().values
        mean_y_smc = idata_smc.posterior["y"].mean().values
    except Exception as e:
        print(f"SMC sampler crashed: {e}")

    # -----------------------------------------------------------------
    # EXPERIMENT 2: Parallel Tempered Slingshot Engine
    # -----------------------------------------------------------------
    print("\nRunning Experiment 2: Multi-Temperature Slingshot Engine...")
    print("Melting energy barriers... Sharding chains via JAX.")
    
    idata_sling = None
    time_slingshot, rhat_x_sling, ess_x_sling = np.nan, np.nan, np.nan
    mean_x_sling, mean_y_sling = np.nan, np.nan
    
    start_slingshot = time.time()
    try:
        idata_sling = sample_slingshot(
            rastrigin_model, draws=1000, tune=1000, chains=4, proposals=800, 
            num_temperatures=24, target_swap_accept=0.60, random_seed=42
        )
        time_slingshot = time.time() - start_slingshot
        summary_sling = az.summary(idata_sling, var_names=["x", "y"])
        rhat_x_sling = summary_sling.loc["x", "r_hat"]
        ess_x_sling = summary_sling.loc["x", "ess_bulk"]
        mean_x_sling = idata_sling.posterior["x"].mean().values
        mean_y_sling = idata_sling.posterior["y"].mean().values
    except Exception as e:
        print(f"Slingshot engine crashed: {e}")
        return

    # -----------------------------------------------------------------
    # PRINT RESULTS TABLE
    # -----------------------------------------------------------------
    print("\n" + "="*75)
    print("                     BENCHMARK COMPARISON RESULTS                    ")
    print("="*75)
    print(f"Engine          | Execution Time | Mean (X, Y)       | R_hat (X) | Bulk ESS")
    print("-" * 75)
    
    smc_time_str = f"{time_smc:.2f}s" if not np.isnan(time_smc) else "N/A"
    smc_mean_str = f"({mean_x_smc:>5.3f}, {mean_y_smc:>5.3f})" if not np.isnan(mean_x_smc) else "N/A"
    smc_rhat_str = f"{rhat_x_smc:.3f}" if not np.isnan(rhat_x_smc) else "N/A"
    smc_ess_str = f"{ess_x_smc:.1f}" if not np.isnan(ess_x_smc) else "N/A"

    print(f"Adaptive SMC    | {smc_time_str:<14} | {smc_mean_str:<17} | {smc_rhat_str:<9} | {smc_ess_str:<8}")
    print(f"Slingshot (PT)  | {time_slingshot:<14.2f}s | ({mean_x_sling:>5.3f}, {mean_y_sling:>5.3f}) | {rhat_x_sling:<9.3f} | {ess_x_sling:<8.1f}")
    print("="*75)
    print("True Theoretical Global Mean: (0.000, 0.000)\n")

    # -----------------------------------------------------------------
    # VISUALIZATION & DENSITY COMPARISON PLOTS
    # -----------------------------------------------------------------
    if idata_smc is not None and idata_sling is not None:
        print("Generating and saving empirical posterior density comparison...")
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            x_smc_samples = idata_smc.posterior["x"].values.flatten()
            y_smc_samples = idata_smc.posterior["y"].values.flatten()
            
            x_sling_samples = idata_sling.posterior["x"].values.flatten()
            y_sling_samples = idata_sling.posterior["y"].values.flatten()
            
            # Using fill=True to avoid deprecation warnings from old 'shade' parameter
            sns.kdeplot(x=x_smc_samples, y=y_smc_samples, cmap="Blues", fill=True, thresh=0.05, ax=axes[0])
            axes[0].set_title("Posterior Space Discovered by SMC")
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("y")
            axes[0].set_xlim(-5, 5)
            axes[0].set_ylim(-5, 5)
            
            sns.kdeplot(x=x_sling_samples, y=y_sling_samples, cmap="Reds", fill=True, thresh=0.05, ax=axes[1])
            axes[1].set_title("Posterior Space Discovered by Slingshot (PT)")
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("y")
            axes[1].set_xlim(-5, 5)
            axes[1].set_ylim(-5, 5)
            
            plt.tight_layout()
            fig.savefig("smc_vs_slingshot_density.png", dpi=300)
            plt.close(fig)
            print(" -> Saved 'smc_vs_slingshot_density.png' successfully.")
            
        except Exception as e:
            print(f" -> Could not generate comparison plots: {e}")

if __name__ == "__main__":
    run_smc_vs_slingshot_comparison()