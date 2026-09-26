"""Rectified flow / conditional flow matching -- replaces TIDE's DDPM+DDIM core.

Why this is the central change over TIDE. TIDE's DDPM needs a variance
schedule, predicts eps, and recovers a clean-sequence estimate by

    x0_hat = (x_tau - sqrt(1-abar_tau) * eps_theta) / sqrt(abar_tau),

a division that blows up as abar_tau -> 0 (the high-noise end of the
schedule). TIDE's own code documents three separate mitigations this forces:
restricting the auxiliary-loss timestep to the low-noise 30% of the schedule
(`aux_tau_frac`), clamping the *value* of intermediate x0 estimates during
differentiable sampling, and truncating backprop to only the last few
(low-noise) DDIM steps (`grad_steps`) because the Jacobian of that division
is otherwise ~2-5x10^2 and swamps every other loss term.

Rectified flow (Liu et al., 2022; used for time series in FlowTS/FM-TS/
TimeFlow, 2024-25) removes the division entirely. With

    X_t = (1-t) X_data + t * X_noise,   t in [0, 1],   v = X_noise - X_data,

the network predicts the (constant, along a straight path) velocity
v_theta(X_t, t), and solving the interpolation for the endpoint gives

    X_data_hat = X_t - t * v_theta(X_t, t)

with NO division anywhere -- bounded, well-conditioned gradients at every
t, by construction. That is what lets `x1_hat` be evaluated at any t (not
just a restricted low-noise slice) and what lets the differentiable
multi-step sampler used by the distributional loss skip the elaborate
value-clamp + gradient-truncation machinery TIDE needed: a straight-line ODE
step here is `x - dt * v_theta(x, t)`, whose gradient magnitude is directly
set by dt and the learned field's smoothness, not by a schedule-dependent
blow-up. `grad_steps` is kept only as a memory/compute knob, not a stability
requirement.

Generation integrates the reverse ODE from t=1 (pure noise) to t=0 (data)
with a handful of Euler steps; because the target field is (in the
population limit) already straight, RecFlowTime needs far fewer steps than
TIDE's 50-step DDIM sampler to reach a comparable sample (Section on
sampling budget in the paper).
"""
import torch

from .config import CoreConfig


class RectifiedFlow:
    """Not an nn.Module -- holds no parameters, just the flow mechanics."""

    def __init__(self, cfg: CoreConfig, device="cpu"):
        self.cfg = cfg
        self.device = device

    def to(self, device):
        self.device = device
        return self

    # ---- forward interpolation --------------------------------------------
    def sample_t(self, B, device, low=0.0, high=1.0, generator=None):
        return torch.rand(B, device=device, generator=generator) * (high - low) + low

    def sample_aux_t(self, B, device, generator=None):
        """t biased toward the low-noise (near-data) end for the critic losses.

        Unlike TIDE's `aux_tau_frac`, this is a learning-difficulty curriculum,
        not a numerical necessity: `predict_clean` below is division-free and
        well-behaved at every t in [0, 1].
        """
        high = max(1e-3, self.cfg.aux_tau_frac)
        return self.sample_t(B, device, 0.0, high, generator)

    # ---- prior ---------------------------------------------------------------
    def fit_noise_spectrum(self, X, seed=0):
        """Fit a temporally correlated prior whose power spectrum matches the
        data, instead of white noise.

        Flow matching transports one distribution to another, and nothing
        requires the starting distribution to be white. Starting from noise
        that already has the data's autocovariance means the velocity field
        does not have to build that structure from nothing; it only has to
        transport what actually differs. The prior stays Gaussian, and it is
        fitted once on the training split before training, so there is no
        feedback from the model.

        Implementation: colour white noise by the per-feature root mean power
        spectrum of the data, then rescale so each feature still has unit
        variance, which keeps the noise level the schedule assumes.
        """
        spec = torch.fft.rfft(X.float(), dim=1)
        amp = (spec.real ** 2 + spec.imag ** 2).mean(0).sqrt()       # (Fq, F)
        self.noise_filter = amp / amp.mean(0, keepdim=True).clamp_min(1e-12)
        g = torch.Generator().manual_seed(seed)
        probe = self._colour(torch.randn(512, X.shape[1], X.shape[2], generator=g))
        self.noise_scale = 1.0 / probe.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)
        return self

    def _colour(self, z):
        f = self.noise_filter.to(z.device)
        Z = torch.fft.rfft(z, dim=1) * f
        return torch.fft.irfft(Z, n=z.shape[1], dim=1)

    def sample_prior(self, shape, device, generator=None):
        z = torch.randn(shape, device=device, generator=generator)
        if not self.cfg.colored_prior or getattr(self, "noise_filter", None) is None:
            return z
        return self._colour(z) * self.noise_scale.to(device)

    def corrupt(self, x_data, t, noise=None):
        noise = self.sample_prior(x_data.shape, x_data.device) if noise is None else noise
        tt = t.view(-1, *([1] * (x_data.dim() - 1)))
        x_t = (1 - tt) * x_data + tt * noise
        return x_t, noise

    def predict_clean(self, x_t, t, v_pred, clamp=None):
        tt = t.view(-1, *([1] * (x_t.dim() - 1)))
        x_data_hat = x_t - tt * v_pred
        if clamp is not None:
            x_data_hat = x_data_hat.clamp(-clamp, clamp)
        return x_data_hat

    def _snr_weight(self, t, eps=1e-4):
        """min-SNR-style weighting adapted to flow time.

        SNR(t) for the linear path (1-t)*x + t*eps has variance ratio
        ((1-t)/t)^2; the min-SNR trick (Hang et al., 2023) caps this at
        `min_snr_gamma` so very-low-noise steps (t->0, SNR->inf) don't
        dominate the loss average, which speeds convergence without
        changing what the optimum is (SNR unweighted is a strict special
        case). Disabled by `core.use_min_snr=False`.
        """
        snr = ((1 - t) / t.clamp_min(eps)) ** 2
        w = torch.clamp(snr, max=self.cfg.min_snr_gamma) / snr.clamp_min(eps)
        floor = getattr(self.cfg, "min_snr_floor", 0.0)
        if floor:
            # As t -> 0 the SNR diverges and w -> gamma/SNR -> 0, so the
            # low-noise end of the path receives almost no gradient. That is
            # where fine temporal detail is resolved, and leaving it
            # unsupervised produces a velocity field that is rough near the
            # data. A floor keeps min-SNR's damping of the mid-range while
            # retaining signal at t -> 0.
            w = w.clamp_min(floor)
        return w

    # ---- temporal preconditioner -------------------------------------------
    def precond_sq_error(self, r):
        """Squared flow-matching residual measured partly in difference space:

            r^T M r,    M = I + lambda * D^T D,

        where D is the forward first-difference along time, (Dr)_t = r_{t+1}
        - r_t with (Dr)_T = 0. Returned elementwise as r^2 + lambda*(Dr)^2 so
        the caller can still apply a per-sample weight.

        Why this cannot change what the model converges to. The flow-matching
        loss E||v - u||^2 is minimized pointwise by v*(x_t, t) = E[u | x_t, t].
        Replacing the squared norm by the quadratic form r^T M r leaves that
        minimizer unchanged for any constant positive definite M, because the
        stationary condition is M (v - E[u | x_t, t]) = 0. Here M = I +
        lambda*D^T D is positive definite for every lambda >= 0, since D^T D
        is positive semidefinite. So there is no excluded value of lambda and
        no risk of an ill-conditioned M.

        What it does change is the geometry of the gradient. An error in the
        step-to-step differences of the velocity now costs more than an error
        in its level. A sequence that matches the level but not the
        transitions, which is what over-smoothing produces, therefore
        receives a larger gradient.

        Note the sign convention. Blending instead of adding, W = (1-lambda)I
        + lambda*S, would be a moving average: it *attenuates* high-frequency
        residuals and would push samples towards being smoother, which is the
        opposite of the intent.

        Multiple lags. With `precond_lags = (1, 2, 4, ...)` the metric becomes
        M = I + sum_k (lambda/|lags|) * D_k^T D_k, where D_k r_t = r_{t+k} -
        r_t. Lag 1 only sees neighbouring steps, so an error that repeats
        with the period of the signal is almost invisible to it. Adding
        larger lags measures exactly those errors. The sum of positive
        semidefinite terms keeps M positive definite, so the argument above
        is unchanged, and dividing by the number of lags keeps the total
        weight comparable to the single-lag case.
        """
        se = r ** 2
        lam = self.cfg.precond_lambda
        if not lam:
            return se
        lags = [k for k in self.cfg.precond_lags if 0 < k < r.shape[1]]
        if not lags:
            return se
        w = lam / len(lags)
        for k in lags:
            d = torch.zeros_like(r)
            d[:, :-k] = r[:, k:] - r[:, :-k]
            se = se + w * d ** 2
        return se

    # ---- minibatch optimal-transport coupling -------------------------------
    def couple(self, x_data, noise):
        """Re-pair the noise within the batch to minimise total transport cost.

        Flow matching normally pairs each sequence with independent noise.
        The resulting conditional paths cross, so the marginal velocity field
        the model must fit is curved, which costs sampling steps and
        accuracy. Pairing by minibatch optimal transport shortens and
        straightens the paths (Tong et al., 2023; Pooladian et al., 2023).
        Both marginals are unchanged, because a permutation of the noise is
        still a sample of the same noise distribution, so this cannot bias
        what the model learns to generate; it is a variance/geometry change.

        The cost is the same temporal metric the loss uses, M = I +
        lambda*D^T D, since ||u||_M^2 = ||u||^2 + lambda*||D u||^2 is a plain
        squared distance between sequences augmented by their scaled
        differences. Pairs are therefore close in the same sense in which
        errors are measured.

        Known caveat: on a minibatch this is a biased estimate of the true
        optimal coupling, and the bias grows as the batch gets smaller.
        """
        if not self.cfg.ot_coupling:
            return noise
        from scipy.optimize import linear_sum_assignment
        with torch.no_grad():
            def feats(z):
                if not self.cfg.precond_lambda:
                    return z.flatten(1)
                d = torch.zeros_like(z)
                d[:, :-1] = z[:, 1:] - z[:, :-1]
                return torch.cat([z, self.cfg.precond_lambda ** 0.5 * d], dim=-1).flatten(1)
            cost = torch.cdist(feats(x_data), feats(noise)).pow(2).cpu().numpy()
            _, col = linear_sum_assignment(cost)
            return noise[torch.as_tensor(col, device=noise.device)]

    # ---- one-step clean estimate (-> critics), with self-conditioning ------
    def x1_hat(self, model, x_data, t=None, generator=None, clamp=None,
              self_cond_prob=0.5, use_self_cond=False):
        B = x_data.shape[0]
        if t is None:
            t = self.sample_aux_t(B, x_data.device, generator)
        x_t, noise = self.corrupt(x_data, t)

        sc = torch.zeros_like(x_data) if use_self_cond else None
        if use_self_cond and torch.rand((), generator=generator).item() < self_cond_prob:
            with torch.no_grad():
                v0 = model(x_t, t, sc)
                sc = self.predict_clean(x_t, t, v0, clamp=clamp).detach()
        v_pred = model(x_t, t, sc) if use_self_cond else model(x_t, t, None)
        x_data_hat = self.predict_clean(x_t, t, v_pred, clamp=clamp)
        return x_data_hat, v_pred, t, noise, x_t

    # ---- training loss ------------------------------------------------------
    def flow_loss(self, model, x_data, generator=None, use_self_cond=False, self_cond_prob=0.5):
        B = x_data.shape[0]
        t = self.sample_t(B, x_data.device, generator=generator)
        noise = self.couple(x_data, self.sample_prior(x_data.shape, x_data.device))
        x_t, noise = self.corrupt(x_data, t, noise)
        target = noise - x_data

        sc = torch.zeros_like(x_data) if use_self_cond else None
        if use_self_cond and torch.rand((), generator=generator).item() < self_cond_prob:
            with torch.no_grad():
                v0 = model(x_t, t, sc)
                sc = self.predict_clean(x_t, t, v0).detach()
        v_pred = model(x_t, t, sc) if use_self_cond else model(x_t, t, None)

        se = self.precond_sq_error(v_pred - target)
        if self.cfg.use_min_snr:
            w = self._snr_weight(t).view(-1, *([1] * (x_data.dim() - 1)))
            se = se * w
        return se.mean()

    # ---- sampling -----------------------------------------------------------
    @torch.no_grad()
    def sample(self, model, shape, n_steps=None, device=None, clamp=1.0,
              use_self_cond=False, x_init=None):
        return self._integrate(model, shape, n_steps, device, clamp,
                               use_self_cond, x_init, differentiable=False, grad_steps=None)

    def sample_differentiable(self, model, shape, n_steps, device=None, clamp=1.0,
                              use_self_cond=False, grad_steps=None, x_init=None):
        return self._integrate(model, shape, n_steps, device, clamp,
                               use_self_cond, x_init, differentiable=True, grad_steps=grad_steps)

    def _integrate(self, model, shape, n_steps, device, clamp, use_self_cond,
                   x_init, differentiable, grad_steps):
        device = device or self.device
        n_steps = n_steps or self.cfg.sampling_steps
        ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
        x = self.sample_prior(shape, device) if x_init is None else x_init
        sc = torch.zeros(shape, device=device) if use_self_cond else None
        grad_steps = n_steps if (grad_steps is None or not differentiable) else grad_steps
        cutover = max(0, n_steps - grad_steps)

        def _step(x, sc, i):
            t_cur, t_next = ts[i], ts[i + 1]
            t_vec = torch.full((shape[0],), t_cur.item(), device=device)
            v = model(x, t_vec, sc)
            x_clean = self.predict_clean(x, t_vec, v, clamp=clamp)
            dt = (t_cur - t_next)
            x_next = x - dt * v
            return x_next, (x_clean.detach() if not differentiable else x_clean)

        if cutover > 0:
            with torch.no_grad():
                for i in range(cutover):
                    x, sc_new = _step(x, sc, i)
                    if use_self_cond:
                        sc = sc_new
        ctx = torch.enable_grad() if differentiable else torch.no_grad()
        with ctx:
            for i in range(cutover, n_steps):
                x, sc_new = _step(x, sc, i)
                if use_self_cond:
                    sc = sc_new if differentiable else sc_new.detach()
        return x
