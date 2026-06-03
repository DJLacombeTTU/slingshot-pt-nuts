import jax, jax.numpy as jnp, blackjax
from blackjax.mcmc.parallel_tempering import parallel_tempering

logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)        # untempered target
betas = jnp.array([1.0, 0.5])                         # cold, hot

init, step = parallel_tempering(
    logdensity_fn, blackjax.nuts,
    {"step_size": 0.5, "inverse_mass_matrix": jnp.ones(1)},
)
state = init(jnp.array([[3.0], [-3.0]]), betas)       # far apart -> swap near-certain
state, info = jax.jit(step)(jax.random.PRNGKey(0), state)

print("swap accepted:", bool(info.swap_acceptance[0]))
for k in range(2):
    pos     = state.inner_state.position[k]
    cached  = state.inner_state.logdensity_grad[k]
    correct = betas[k] * jax.grad(logdensity_fn)(pos)
    tag = "STALE" if not jnp.allclose(cached, correct, atol=1e-5) else "ok"
    print(f"rung {k} (beta={float(betas[k])}): cached={float(cached[0]):+.4f}  "
          f"correct={float(correct[0]):+.4f}  {tag}")