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
        self.linear3 = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, x, t):
        t_col = t.reshape((-1, 1)) if x.ndim > 1 else jnp.atleast_1d(t)
        z = jnp.concatenate((x, t_col), axis=-1)
        z = nnx.relu(self.linear1(z))
        z = nnx.relu(self.linear2(z))
        return self.linear3(z)


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
        hidden_dim: int = 32,
        n_steps: int = 100,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 1e-3,
        seed: int = 0,
        patience: int = 20,
        min_delta: float = 1e-6,
    ):
        self.latent_dim = latent_dim
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.min_delta = min_delta

        beta = LinearSchedule(b_min=0.02, b_max=1.0, t0=0.0, T=1.0)
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

    @staticmethod
    def _loss_only(network, x_batch, t_batch, y_batch):
        """Same score-matching MSE as `_loss_fn`, but no gradient tape.

        Used for validation passes where we want the loss value without
        paying for (or risking misuse of) a gradient computation.
        """
        return jnp.mean((network(x_batch, t_batch) - y_batch) ** 2)

    def _train_step(self, x_batch, t_batch, y_batch) -> float:
        loss, grads = self._loss_fn(self.network, x_batch, t_batch, y_batch)
        self.optimizer.update(self.network, grads)
        return float(loss)

    def _eval_step(self, x_batch, t_batch, y_batch) -> float:
        """Forward pass only — no optimizer update, no gradient computation."""
        loss = self._loss_only(self.network, x_batch, t_batch, y_batch)
        return float(loss)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        latents: jnp.ndarray,
        val_split: float = 0.2,
        verbose: bool = True,
    ) -> "LatentDiffusion":
        """Train the score network on `latents` (shape [N, latent_dim]).

        Parameters
        ----------
        latents   : array of shape [N, latent_dim]
        val_split : fraction of *original latent rows* held out for
                    validation before the forward-diffusion data is built.
                    Early stopping is driven by validation loss, not
                    training loss.
        verbose   : print per-epoch progress
        """
        n_latents = latents.shape[0]

        # ---- split the raw latents BEFORE forward-diffusion ----
        # (so train/val never share the same underlying (s, z) row,
        #  even though each row is expanded into n_steps noised copies)
        self.key, split_key = jax.random.split(self.key)
        perm = jax.random.permutation(split_key, n_latents)
        n_val = max(1, int(round(val_split * n_latents)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        latents_tr = latents[train_idx]
        latents_va = latents[val_idx]

        if verbose:
            print(f"  [diffusion] train/val split: {latents_tr.shape[0]} / {latents_va.shape[0]} latent rows")

        # Build forward-diffusion training data for each split.
        # Each call advances self.key, so train and val get independent noise draws.
        x_tr, t_tr, y_tr = self._make_training_data(latents_tr)
        x_va, t_va, y_va = self._make_training_data(latents_va)
        n_train = x_tr.shape[0]
        n_val_rows = x_va.shape[0]

        # Early stopping state — now tracked on VALIDATION loss
        best_val_loss = float('inf')
        best_params = None
        epochs_without_improvement = 0

        # History for convergence diagnostics
        self.history_ = {"train_loss": [], "val_loss": []}

        for epoch in range(self.n_epochs):
            # ---- training pass ----
            self.key, perm_key = jax.random.split(self.key)
            perm_tr = jax.random.permutation(perm_key, n_train)
            epoch_train_loss, n_batches = 0.0, 0

            for start in range(0, n_train, self.batch_size):
                idx = perm_tr[start : start + self.batch_size]
                epoch_train_loss += self._train_step(x_tr[idx], t_tr[idx], y_tr[idx])
                n_batches += 1

            avg_train_loss = epoch_train_loss / max(n_batches, 1)

            # ---- validation pass (no gradient updates) ----
            epoch_val_loss, n_val_batches = 0.0, 0
            for start in range(0, n_val_rows, self.batch_size):
                epoch_val_loss += self._eval_step(
                    x_va[start : start + self.batch_size],
                    t_va[start : start + self.batch_size],
                    y_va[start : start + self.batch_size],
                )
                n_val_batches += 1
            avg_val_loss = epoch_val_loss / max(n_val_batches, 1)

            self.history_["train_loss"].append(avg_train_loss)
            self.history_["val_loss"].append(avg_val_loss)

            if verbose and epoch % 10 == 0:
                print(
                    f"  [diffusion] epoch {epoch:3d}  "
                    f"train_loss={avg_train_loss:.6f}  val_loss={avg_val_loss:.6f}  "
                    f"best_val={best_val_loss:.6f}  patience={epochs_without_improvement}/{self.patience}"
                )

            # ---- early stopping on VALIDATION loss ----
            if avg_val_loss < best_val_loss - self.min_delta:
                best_val_loss = avg_val_loss
                best_params = nnx.state(self.network)   # snapshot best weights
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    if verbose:
                        print(
                            f"  [diffusion] early stopping at epoch {epoch}  "
                            f"best_val_loss={best_val_loss:.6f}"
                        )
                    nnx.update(self.network, best_params)  # restore best weights
                    break
        else:
            # Loop completed all n_epochs without early stopping —
            # still restore the best validation checkpoint rather than
            # whatever the final epoch happened to land on.
            if best_params is not None:
                if verbose:
                    print(
                        f"  [diffusion] reached n_epochs={self.n_epochs}; "
                        f"restoring best_val_loss={best_val_loss:.6f}"
                    )
                nnx.update(self.network, best_params)

        return self

    def plot_convergence(self, ax=None):
        """Plot train vs. validation loss curves from the last `fit()` call."""
        import matplotlib.pyplot as plt

        if not hasattr(self, "history_"):
            raise RuntimeError("Call `fit()` before `plot_convergence()`.")

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        epochs = range(len(self.history_["train_loss"]))
        ax.plot(epochs, self.history_["train_loss"], label="train loss")
        ax.plot(epochs, self.history_["val_loss"], label="val loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("score-matching MSE")
        ax.set_title("Diffusion training convergence")
        ax.legend()
        ax.grid(alpha=0.3)
        return ax

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


class ConditionalLatentDiffusion:
    """
    Conditional latent diffusion model for HI-VAE latents.
    Learns p(z | s) rather than p(z)

    Parameters
    ----------
    latent_dim : int
        Dimension of continuous latent z.
    s_dim : int
        Number of HI-VAE mixture components / dimension of s.
    hidden_dim : int
        Width of score network.
    n_steps : int
        Number of diffusion steps.
    n_epochs : int
        Maximum number of score-network training epochs.
    batch_size : int
        Diffusion training batch size.
    lr : float
        Adam learning rate.
    """

    def __init__(
        self,
        latent_dim: int,
        s_dim: int,
        hidden_dim: int = 32,
        n_steps: int = 100,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 1e-3,
        seed: int = 0,
        patience: int = 20,
        min_delta: float = 1e-6,
    ):
        self.latent_dim = latent_dim
        self.s_dim = s_dim
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.min_delta = min_delta

        beta = LinearSchedule(b_min=0.02, b_max=1.0, t0=0.0, T=1.0)
        self.sde = SDE(beta=beta)
        self.timer = VpTimer(eps=1e-5, tf=self.sde.tf, n_steps=n_steps)

        self.network = LatentDiffusionNetwork(
            in_dim=latent_dim + s_dim + 1,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            rngs=nnx.Rngs(seed),
        )
        self.optimizer = nnx.Optimizer(self.network, optax.adam(lr), wrt=nnx.Param)
        self.key = jax.random.PRNGKey(seed)

        # Will be estimated during fit().
        self.s_probs_ = None
        
    # ================================================================
    # Helpers
    # ================================================================

    def _prepare_s(self, s: jnp.ndarray) -> jnp.ndarray:
        """
        Convert s into shape [N, s_dim].
        Accepted formats
        ----------------
        1. Integer component labels, shape [N]: e.g. [0, 2, 1, 0, ...]
        2. Integer labels, shape [N, 1]
        3. One-hot / soft / Gumbel-softmax, shape [N, s_dim]:
           e.g. [[1., 0., 0.],
                 [0., 0., 1.],
                 ...]
           or soft:[[0.90, 0.08, 0.02],
                    [0.10, 0.80, 0.10],
                    ...]
        """
        s = jnp.asarray(s)
        if s.ndim == 1:
            s = jax.nn.one_hot(s.astype(jnp.int32), self.s_dim,)
        elif s.ndim == 2 and s.shape[1] == 1:
            s = jax.nn.one_hot(s[:, 0].astype(jnp.int32), self.s_dim)
        elif s.ndim == 2 and s.shape[1] == self.s_dim:
            # Already one-hot or soft assignments.
            s = s.astype(jnp.float32)
        else:
            raise ValueError(
                f"Unexpected shape for s: {s.shape}. "
                f"Expected [N], [N, 1], or [N, {self.s_dim}]."
            )
        return s

    # ================================================================
    # Forward diffusion
    # ================================================================
    
    def _make_training_data(self, latents: jnp.ndarray, s: jnp.ndarray) -> tuple:
        """ Forward diffuse z while keeping s fixed.
        For every original pair (z_i, s_i) create (z_i(t), s_i, t, epsilon) where only z is diffused.
        IMPORTANT: s is NOT noised.
        """
        xs, ss, ts, ys = [], [], [], []
        key = self.key
        for i in range(self.n_steps):
            key, noise_key = jax.random.split(key)
            t_i = self.timer(i)
            noise = jax.random.normal(noise_key, shape=latents.shape)
            z_t = (self.sde.signal_level(t_i) * latents + self.sde.noise_level(t_i) * noise)
            xs.append(z_t)
            # Same s for every diffusion time.
            ss.append(s)
            ts.append(jnp.full((latents.shape[0],),t_i,))
            ys.append(noise)
        self.key = key
        return (
            jnp.concatenate(xs, axis=0),
            jnp.concatenate(ss, axis=0),
            jnp.concatenate(ts, axis=0),
            jnp.concatenate(ys, axis=0),
        )

    # ================================================================
    # Conditional score loss
    # ================================================================

    @staticmethod
    @nnx.value_and_grad
    def _loss_fn(network, z_batch, s_batch, t_batch, noise_batch):
        """
        Noise-prediction objective: eps_hat = eps_theta(z_t, t, s)
        implemented by feeding concat(z_t, s) as x to the existing LatentDiffusionNetwork.
        """
        conditional_input = jnp.concatenate([z_batch, s_batch], axis=-1)
        noise_hat = network(conditional_input, t_batch)
        return jnp.mean((noise_hat - noise_batch) ** 2)

    @staticmethod
    def _loss_only(network, z_batch, s_batch, t_batch, noise_batch):
        conditional_input = jnp.concatenate([z_batch, s_batch], axis=-1)
        noise_hat = network(conditional_input, t_batch)
        return jnp.mean((noise_hat - noise_batch) ** 2)

    def _train_step(self, z_batch, s_batch, t_batch, noise_batch) -> float:
        loss, grads = self._loss_fn(self.network, z_batch, s_batch, t_batch, noise_batch)
        self.optimizer.update(self.network, grads)
        return float(loss)

    def _eval_step(self, z_batch, s_batch, t_batch, noise_batch) -> float:
        loss = self._loss_only(self.network, z_batch, s_batch, t_batch, noise_batch)
        return float(loss)

    # ================================================================
    # Training
    # ================================================================

    def fit(self,
        latents: jnp.ndarray,
        s: jnp.ndarray,
        val_split: float = 0.2,
        verbose: bool = True,
    ) -> "ConditionalLatentDiffusion":
        """
        Train conditional score model on paired (z_i, s_i).
        Parameters
        ----------
        latents : Shape [N, latent_dim].
        s : Either:[N], [N, 1], [N, s_dim]

        For HI-VAE, [N, s_dim] can directly contain the Gumbel-softmax / one-hot representation.
        """
        latents = jnp.asarray(latents)
        s = self._prepare_s(s)
        if latents.shape[0] != s.shape[0]:
            raise ValueError(f"latents and s must contain the same number of observations, got {latents.shape[0]} and {s.shape[0]}.")
        if latents.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent dimension {self.latent_dim}, got {latents.shape[1]}.")

        # ------------------------------------------------------------
        # Estimate empirical p(s)
        # For hard one-hot s: this gives component frequencies.
        # For soft q(s|x): this gives average posterior responsibilities.
        # ------------------------------------------------------------

        self.s_probs_ = jnp.mean(s, axis=0)
        self.s_probs_ = (self.s_probs_ / jnp.sum(self.s_probs_))

        # ------------------------------------------------------------
        # Train / validation split
        # IMPORTANT: z and s must use the SAME permutation.
        # ------------------------------------------------------------

        n_latents = latents.shape[0]
        self.key, split_key = jax.random.split(self.key)
        perm = jax.random.permutation(split_key, n_latents,)
        n_val = max(1, int(round(val_split * n_latents)),)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        z_tr = latents[train_idx]
        z_va = latents[val_idx]
        s_tr = s[train_idx]
        s_va = s[val_idx]

        if verbose:
            print(f"  [conditional diffusion] train/val split: {z_tr.shape[0]} / {z_va.shape[0]} latent rows")
            print(f"  [conditional diffusion] estimated p(s): {self.s_probs_}")

        # ------------------------------------------------------------
        # Forward diffuse z. s remains unchanged.
        # ------------------------------------------------------------

        z_tr_t, s_tr_t, t_tr, y_tr = (self._make_training_data(z_tr, s_tr,))
        z_va_t, s_va_t, t_va, y_va = (self._make_training_data(z_va, s_va))
        n_train = z_tr_t.shape[0]
        n_val_rows = z_va_t.shape[0]

        # ------------------------------------------------------------
        # Early stopping
        # ------------------------------------------------------------

        best_val_loss = float("inf")
        best_params = None
        epochs_without_improvement = 0
        self.history_ = {"train_loss": [], "val_loss": [],}

        # ------------------------------------------------------------
        # Optimization
        # ------------------------------------------------------------

        for epoch in range(self.n_epochs):

            # Training permutation
            self.key, perm_key = jax.random.split(self.key)
            perm_tr = jax.random.permutation(perm_key, n_train,)
            epoch_train_loss = 0.0
            n_batches = 0
            for start in range(0, n_train, self.batch_size):
                idx = perm_tr[start : start + self.batch_size]
                epoch_train_loss += self._train_step(z_tr_t[idx], s_tr_t[idx], t_tr[idx], y_tr[idx])
                n_batches += 1
            avg_train_loss = (epoch_train_loss / max(n_batches, 1))

            # --------------------------------------------------------
            # Validation
            # --------------------------------------------------------
            epoch_val_loss = 0.0
            n_val_batches = 0
            for start in range(0, n_val_rows, self.batch_size):
                end = start + self.batch_size
                epoch_val_loss += self._eval_step(z_va_t[start:end], s_va_t[start:end], t_va[start:end], y_va[start:end])
                n_val_batches += 1
            avg_val_loss = (epoch_val_loss / max(n_val_batches, 1))
            self.history_["train_loss"].append(avg_train_loss)
            self.history_["val_loss"].append(avg_val_loss)

            if verbose and epoch % 10 == 0:
                print(
                    f"  [conditional diffusion] "
                    f"epoch {epoch:3d}  "
                    f"train_loss={avg_train_loss:.6f}  "
                    f"val_loss={avg_val_loss:.6f}  "
                    f"best_val={best_val_loss:.6f}  "
                    f"patience={epochs_without_improvement}/{self.patience}"
                )

            # --------------------------------------------------------
            # Early stopping
            # --------------------------------------------------------

            if (avg_val_loss < best_val_loss - self.min_delta):
                best_val_loss = avg_val_loss
                best_params = nnx.state(self.network)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if (epochs_without_improvement >= self.patience):
                    if verbose:
                        print(f"  [conditional diffusion] early stopping at epoch {epoch}; best_val_loss={best_val_loss:.6f}")
                    if best_params is not None:
                        nnx.update(self.network, best_params)
                    break
        else:
            if best_params is not None:
                if verbose:
                    print(f"  [conditional diffusion]  reached n_epochs={self.n_epochs}; restoring best_val_loss={best_val_loss:.6f}")
                nnx.update(self.network, best_params)
        return self

    # ================================================================
    # Sampling
    # ================================================================

    def sample(
        self,
        n_samples: int,
        s: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """
        Draw z ~ p_theta(z | s).
        If s is supplied: generate z conditioned on those components.
        If s is None: sample s from empirical p(s) estimated during fit().
        """
        # Use advancing RNG rather than resetting PRNGKey(42) on every call.
        self.key, sample_key = jax.random.split(self.key)

        # ------------------------------------------------------------
        # Choose s for generated rows
        # ------------------------------------------------------------
        if s is None:
            if self.s_probs_ is None:
                raise RuntimeError("Call fit() before sample(), or explicitly provide s.")
            sample_key, s_key = jax.random.split(sample_key)
            s_idx = jax.random.categorical(s_key, jnp.log(self.s_probs_ + 1e-12), shape=(n_samples,))
            s = jax.nn.one_hot(s_idx, self.s_dim)
        else:
            s = self._prepare_s(s)
            if s.shape[0] != n_samples:
                raise ValueError(f"When s is supplied, s.shape[0] must equal n_samples. Got {s.shape[0]} and {n_samples}.")

        # ------------------------------------------------------------
        # Sample, conditioning each particle on its own row of s.
        #
        # `Denoiser.generate()` only vmaps over its internal `state`
        # (position / rng_key / step); a closed-over `s` array would stay
        # at its full (n_samples, s_dim) shape inside every vmapped
        # instance instead of being sliced per particle. So we vmap the
        # per-particle DDIM loop ourselves over (rndm_start, key, s_row)
        # jointly, giving each particle its matching s row.
        # ------------------------------------------------------------
        integrator = DDIMIntegrator(model=self.sde, timer=self.timer)

        def run_single(rndm_start, key, s_row):
            def conditional_network(z_t, t):
                conditional_input = jnp.concatenate([z_t, s_row], axis=-1)
                return self.network(conditional_input, t)

            predictor = Predictor(model=self.sde, network=conditional_network, prediction_type="noise")
            state = integrator.init(rndm_start, key)

            def body_fun(state, _):
                return integrator(state, predictor), None

            final_state, _ = jax.lax.scan(body_fun, state, jnp.arange(self.n_steps))
            return final_state.position

        sample_key, key_start = jax.random.split(sample_key)
        rndm_start = jax.random.normal(key_start, shape=(n_samples, self.latent_dim))
        keys = jax.random.split(sample_key, n_samples)

        return jax.vmap(run_single)(rndm_start, keys, s)

    def plot_convergence(self, ax=None):
        """Plot train vs. validation loss curves from the last `fit()` call."""
        import matplotlib.pyplot as plt

        if not hasattr(self, "history_"):
            raise RuntimeError("Call `fit()` before `plot_convergence()`.")

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        epochs = range(len(self.history_["train_loss"]))
        ax.plot(epochs, self.history_["train_loss"], label="train loss")
        ax.plot(epochs, self.history_["val_loss"], label="val loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("score-matching MSE")
        ax.set_title("Diffusion training convergence")
        ax.legend()
        ax.grid(alpha=0.3)
        return ax