import blackjax
import jax

# Use **kwargs to capture whatever parameters parallel_tempering sends
def nuts_factory(logdensity_fn, step_size, inverse_mass_matrix, **kwargs):
    return blackjax.nuts(
        logdensity_fn, 
        step_size=step_size, 
        inverse_mass_matrix=inverse_mass_matrix
    )

def hmc_factory(logdensity_fn, step_size, inverse_mass_matrix, num_steps=10, **kwargs):
    return blackjax.hmc(
        logdensity_fn, 
        step_size=step_size, 
        inverse_mass_matrix=inverse_mass_matrix,
        num_integration_steps=num_steps
    )

def rwmh_factory(logdensity_fn, step_size=None, inverse_mass_matrix=None):
    """Factory for RWMH (step_size/mass ignored for this simple kernel)."""
    return blackjax.random_walk_metropolis(logdensity_fn, sigma=0.1)            