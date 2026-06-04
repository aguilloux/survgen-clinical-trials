# diffuse_model.py  (new file)
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from diffuse.diffusion.sde import LinearSchedule, SDE
from diffuse.timer import VpTimer
from diffuse.integrator.deterministic import DDIMIntegrator
from diffuse.integrator.stochastic import EulerMaruyamaIntegrator
from diffuse.denoisers.denoiser import Denoiser
from diffuse.predictor import Predictor


class LatentDiffusionNetwork(nnx.Module):
    """Score network for denoising latent (s, z) concatenations."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.linear4 = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, x, t):
        t_col = t.reshape((-1, 1)) if x.ndim > 1 else jnp.atleast_1d(t)
        z = jnp.concatenate((x, t_col), axis=-1)
        z = nnx.relu(self.linear1(z))
        z = nnx.relu(self.linear2(z))
        z = nnx.relu(self.linear3(z))
        return self.linear4(z)


class LatentDiffusion:
    """
    Wraps SDE definition, score-network training, and DDIM sampling
    for a fixed latent dimensionality.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent vector (s_dim + z_dim).
    hidden_dim : int
        Width of the score network. Default 256.
    n_steps : int
        Number of diffusion steps. Default 100.
    n_epochs : int
        Training epochs for the score network. Default 50.
    batch_size : int
        Mini-batch size during score-network training. Default 512.
    lr : float
        Adam learning-rate. Default 1e-3.
    seed : int
        JAX PRNG seed. Default 0.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        n_steps: int = 100,
        n_epochs: int = 50,
        batch_size: int = 512,
        lr: float = 1e-3,
        seed: int = 0,
    ):
        self.latent_dim = latent_dim
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        beta = LinearSchedule(b_min=0.02, b_max=1.0, t0=0.0, T=10.0)
        self.sde = SDE(beta=beta)
        self.timer = VpTimer(eps=1e-5, tf=self.sde.tf, n_steps=n_steps)

        self.network = LatentDiffusionNetwork(
            in_dim=latent_dim + 1,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            rngs=nnx.Rngs(seed),
        )
        self.optimizer = nnx.Optimizer(self.network, optax.adam(lr), wrt=nnx.Param)
        self.key = jax.random.PRNGKey(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_training_data(self, latents: jnp.ndarray) -> tuple:
        """Forward-diffuse `latents` at every timer step → (x, t, noise)."""
        xs, ts, ys = [], [], []
        key = self.key
        for i in range(self.n_steps):
            key, noise_key = jax.random.split(key)
            t_i = self.timer(i)
            noise = jax.random.normal(noise_key, shape=latents.shape)
            x_t = self.sde.signal_level(t_i) * latents + self.sde.noise_level(t_i) * noise
            xs.append(x_t)
            ts.append(jnp.full((latents.shape[0],), t_i))
            ys.append(noise)
        self.key = key  # advance global key
        return (
            jnp.concatenate(xs, axis=0),
            jnp.concatenate(ts, axis=0),
            jnp.concatenate(ys, axis=0),
        )

    @staticmethod
    @nnx.value_and_grad
    def _loss_fn(network, x_batch, t_batch, y_batch):
        return jnp.mean((network(x_batch, t_batch) - y_batch) ** 2)

    def _train_step(self, x_batch, t_batch, y_batch) -> float:
        loss, grads = self._loss_fn(self.network, x_batch, t_batch, y_batch)
        self.optimizer.update(self.network, grads)
        return float(loss)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, latents: jnp.ndarray, verbose: bool = True) -> "LatentDiffusion":
        """Train the score network on `latents` (shape [N, latent_dim])."""
        x, t, y = self._make_training_data(latents)
        n_total = x.shape[0]

        for epoch in range(self.n_epochs):
            self.key, perm_key = jax.random.split(self.key)
            perm = jax.random.permutation(perm_key, n_total)
            epoch_loss, n_batches = 0.0, 0

            for start in range(0, n_total, self.batch_size):
                idx = perm[start : start + self.batch_size]
                epoch_loss += self._train_step(x[idx], t[idx], y[idx])
                n_batches += 1

            if verbose and epoch % 10 == 0:
                print(f"  [diffusion] epoch {epoch:3d}  loss={epoch_loss / n_batches:.6f}")

        return self

    def sample(self, n_samples: int) -> jnp.ndarray:
        """Draw `n_samples` latent vectors via DDIM."""
        key = jax.random.PRNGKey(42)
        predictor = Predictor(
            model=self.sde, network=self.network.__call__, prediction_type="noise"
        )
        denoiser = Denoiser(
            integrator=DDIMIntegrator(model=self.sde, timer=self.timer),
            # integrator=EulerMaruyamaIntegrator(model=self.sde, timer=self.timer),
            model=self.sde,
            predictor=predictor,
            x0_shape=(self.latent_dim,),
        )
        final_state, _ = denoiser.generate(key, self.n_steps, n_samples, keep_history=True)
        return final_state.integrator_state.position