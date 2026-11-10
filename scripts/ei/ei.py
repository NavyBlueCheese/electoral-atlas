"""Two-stage Bayesian ecological inference for Thai constituencies.

Question: how did each population group vote, constituency by constituency?
Inputs are aggregates only: each constituency's result (votes per outcome, including
'did not vote') and the share of its eligible voters in each group.

Stage 1 (regional pattern, full Bayesian):
    theta0[r,g,:] = softmax(alpha[g] + u[r,g])      P(outcome | group g, region r)
    p[c,:]        = sum_g A[c,g] * theta0[region(c),g,:]
    s[c,:]        ~ Dirichlet(kappa * p[c,:])         kappa absorbs local variation
Stage 2 (constituency estimates), for every posterior draw:
    start from theta0 plus random local variation (sd sigma_v on the logit scale),
    then adjust by iterative proportional fitting so the estimates reproduce the
    constituency's actual result exactly and every group's shares sum to 1.
The spread across draws is the reported uncertainty. It covers uncertainty in the
regional pattern and a stated amount of local variation; it cannot reveal local
group behaviour that the aggregate results do not contain.
"""
import numpy as np
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(2)

# Calibration from simulation tests (see validate.py): widen local variation by 30%, and
# national/regional ranges by 50%, so that 90% ranges contain the truth about 90% of the time.
SV_MULT = 1.3
NAT_WIDEN = 1.5


def stage1_model(A, S, reg, R, prov=None, P=0, preg=None):
    """Pattern by group: nation + region + (optionally) province. Returns theta0 per province (or region)."""
    C, G = A.shape
    K = S.shape[1]
    alpha = numpyro.sample('alpha', dist.Normal(0, 2).expand([G, K - 1]).to_event(2))
    su = numpyro.sample('sigma_u', dist.HalfNormal(1.0))
    zu = numpyro.sample('zu', dist.Normal(0, 1).expand([R, G, K - 1]).to_event(3))
    eta_r = alpha[None] + su * zu
    if prov is not None:
        sw = numpyro.sample('sigma_w', dist.HalfNormal(0.7))
        zw = numpyro.sample('zw', dist.Normal(0, 1).expand([P, G, K - 1]).to_event(3))
        eta = eta_r[preg] + sw * zw
        idx, L = prov, P
    else:
        eta, idx, L = eta_r, reg, R
    eta = jnp.concatenate([eta, jnp.zeros((L, G, 1))], axis=-1)
    theta0 = jax.nn.softmax(eta, axis=-1)
    numpyro.deterministic('theta0', theta0)
    p = jnp.einsum('cg,cgk->ck', A, theta0[idx])
    kappa = numpyro.sample('kappa', dist.LogNormal(jnp.log(300.0), 1.5))
    numpyro.sample('obs', dist.Dirichlet(kappa * jnp.clip(p, 1e-6, 1.0)), obs=S)


def shares(Y):
    S = Y / Y.sum(1, keepdims=True)
    S = np.clip(S, 1e-5, None)
    return S / S.sum(1, keepdims=True)


def stage1(A, Y, reg, warmup=600, samples=500, chains=2, seed=0, prov=None):
    R = int(reg.max()) + 1
    mcmc = MCMC(NUTS(stage1_model, target_accept_prob=0.9), num_warmup=warmup, num_samples=samples,
                num_chains=chains, chain_method='parallel', progress_bar=False)
    if prov is None:
        mcmc.run(jax.random.PRNGKey(seed), jnp.array(A), jnp.array(shares(Y)), jnp.array(reg), R)
    else:
        P = int(prov.max()) + 1
        preg = np.zeros(P, int); preg[prov] = reg
        mcmc.run(jax.random.PRNGKey(seed), jnp.array(A), jnp.array(shares(Y)), jnp.array(reg), R, jnp.array(prov), P, jnp.array(preg))
    return mcmc


def ipf(theta, A, P, iters=200):
    """theta: C x G x K starting guess. Rescale so rows (groups) sum to 1 and sum_g A*theta = P."""
    n = A[:, :, None] * theta
    for _ in range(iters):
        n *= (A / np.maximum(n.sum(2), 1e-12))[:, :, None]
        n *= (P / np.maximum(n.sum(1), 1e-12))[:, None, :]
    return n / np.maximum(n.sum(2, keepdims=True), 1e-12)


def estimate_sigma_v(A, Y, reg, theta0_mean, rng, grid=np.linspace(0.05, 1.2, 24), reps=3):
    # reg here is the index into theta0 (region or province)
    """Choose the local-variation scale whose simulated constituency results scatter around the
    regional pattern as much as the real results do (moment matching on result shares)."""
    P = shares(Y)
    base = theta0_mean[reg]
    p0 = np.einsum('cg,cgk->ck', A, base)
    target = np.mean((P - p0) ** 2)
    logb = np.log(np.clip(base, 1e-9, None))
    best = None
    for sv in grid:
        m = 0
        for _ in range(reps):
            t = np.exp(logb + rng.normal(0, sv, base.shape)); t /= t.sum(-1, keepdims=True)
            m += np.mean((np.einsum('cg,cgk->ck', A, t) - p0) ** 2) / reps
        d = abs(np.log(m / target))
        if best is None or d < best[0]:
            best = (d, sv)
    return float(best[1])


def stage2(A, Y, reg, theta0_draws, sigma_v, rng, ndraw=400):
    """Constituency-level draws that exactly reproduce each constituency's result."""
    P = shares(Y)
    idx = rng.choice(theta0_draws.shape[0], ndraw, replace=theta0_draws.shape[0] < ndraw)
    out = np.empty((ndraw,) + (A.shape[0], A.shape[1], P.shape[1]), dtype=np.float32)
    for i, d in enumerate(idx):
        base = np.log(np.clip(theta0_draws[d][reg], 1e-9, None)) + rng.normal(0, sigma_v, (A.shape[0], A.shape[1], P.shape[1]))
        t = np.exp(base); t /= t.sum(-1, keepdims=True)
        out[i] = ipf(t, A, P)
    return out


def summarise(draws):
    t = np.asarray(draws)
    mean = t.mean(0)
    lo, hi = np.percentile(t, [5, 95], axis=0)
    top = t.argmax(-1)
    ptop = np.stack([(top == k).mean(0) for k in range(t.shape[-1])], -1)
    return mean, lo, hi, ptop


def run(A, Y, reg, seed=0, warmup=600, samples=500, ndraw=400, prov=None):
    rng = np.random.default_rng(seed)
    m = stage1(A, Y, reg, warmup=warmup, samples=samples, seed=seed, prov=prov)
    th0 = np.asarray(m.get_samples()['theta0'])
    idx = reg if prov is None else prov
    sv = estimate_sigma_v(A, Y, idx, th0.mean(0), rng) * SV_MULT
    draws = stage2(A, Y, idx, th0, sv, rng, ndraw=ndraw)
    return m, sv, draws


def stage1_model_v2(A, S, reg, R, prov, P, preg):
    """v2: smooth age profile + geography shared across ages + small age-by-region interaction.
    eta[p,g,k] = alpha[g,k] + u[r(p),k] + w[p,k] + i[r(p),g,k]
    alpha follows a random walk over ordered age groups (neighbouring ages behave alike)."""
    C, G = A.shape
    K = S.shape[1]
    a0 = numpyro.sample('a0', dist.Normal(0, 2).expand([K - 1]).to_event(1))
    tau = numpyro.sample('tau', dist.HalfNormal(0.6))
    za = numpyro.sample('za', dist.Normal(0, 1).expand([G - 1, K - 1]).to_event(2))
    alpha = jnp.concatenate([a0[None], a0[None] + tau * jnp.cumsum(za, 0)], 0)
    su = numpyro.sample('sigma_u', dist.HalfNormal(1.0))
    zu = numpyro.sample('zu', dist.Normal(0, 1).expand([R, K - 1]).to_event(2))
    sw = numpyro.sample('sigma_w', dist.HalfNormal(0.7))
    zw = numpyro.sample('zw', dist.Normal(0, 1).expand([P, K - 1]).to_event(2))
    si = numpyro.sample('sigma_i', dist.HalfNormal(0.25))
    zi = numpyro.sample('zi', dist.Normal(0, 1).expand([R, G, K - 1]).to_event(3))
    geo = (su * zu)[preg] + sw * zw                     # P x K-1
    eta = alpha[None] + geo[:, None, :] + (si * zi)[preg]
    eta = jnp.concatenate([eta, jnp.zeros((P, G, 1))], axis=-1)
    theta0 = jax.nn.softmax(eta, axis=-1)
    numpyro.deterministic('theta0', theta0)
    p = jnp.einsum('cg,cgk->ck', A, theta0[prov])
    kappa = numpyro.sample('kappa', dist.LogNormal(jnp.log(300.0), 1.5))
    numpyro.sample('obs', dist.Dirichlet(kappa * jnp.clip(p, 1e-6, 1.0)), obs=S)


def run_v2(A, Y, reg, prov, seed=0, warmup=800, samples=600, ndraw=500, chains=2):
    rng = np.random.default_rng(seed)
    R = int(reg.max()) + 1; P = int(prov.max()) + 1
    preg = np.zeros(P, int); preg[prov] = reg
    mcmc = MCMC(NUTS(stage1_model_v2, target_accept_prob=0.9), num_warmup=warmup, num_samples=samples,
                num_chains=chains, chain_method='parallel', progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), jnp.array(A), jnp.array(shares(Y)), jnp.array(reg), R, jnp.array(prov), P, jnp.array(preg))
    th0 = np.asarray(mcmc.get_samples()['theta0'])
    sv = estimate_sigma_v(A, Y, prov, th0.mean(0), rng) * SV_MULT
    draws = stage2(A, Y, prov, th0, sv, rng, ndraw=ndraw)
    return mcmc, sv, draws


# ---------------------------------------------------------------------------------------------
# v3: survey-informed ecological inference.
#  * geography (region + province) shared across ages; no age-by-region interaction
#  * smooth age profile, with an outcome-specific step size: parties may differ a lot by age,
#    'other parties', 'spoiled' and 'did not vote' are shrunk toward no age difference unless the data insist
#  * optional survey terms: each says how a party's support RELATIVE TO A REFERENCE PARTY differs between
#    an age group and the whole sample. Overall poll levels are never used (polls missed the overall result).
#      model: c[g,j] = (alpha[g,cols_j]-alpha[g,ref]) - sum_g w[g] (alpha[g,cols_j]-alpha[g,ref])
#      data:  c_obs[g,j] ~ Normal(c[g,j], sd[g,j])     (sd = sampling error + poll-error allowance; 100 = no data)


def poll_term(groups_q, all_q, n, w, ref, bias_sd):
    """groups_q: G x J poll shares (np.nan where unknown), all_q: J overall shares, n: G respondents, w: G weights."""
    q = np.asarray(groups_q, float); a = np.asarray(all_q, float); n = np.asarray(n, float)
    lr = np.log(q) - np.log(q[:, [ref]]); la = np.log(a) - np.log(a[ref])
    c = lr - la[None, :]
    var = 1.0 / (n[:, None] * q / 100) + 1.0 / (n[:, None] * q[:, [ref]] / 100)
    sd = np.sqrt(var + bias_sd ** 2)
    bad = ~np.isfinite(c) | ~np.isfinite(sd)
    c[bad] = 0.0; sd[bad] = 100.0; sd[:, ref] = 100.0
    return dict(c=c, sd=sd, w=np.asarray(w, float) / np.sum(w))


def stage1_model_v3(A, S, reg, R, prov, P, preg, tau_scale, polls=()):
    C, G = A.shape
    K = S.shape[1]
    a0 = numpyro.sample('a0', dist.Normal(0, 2).expand([K - 1]).to_event(1))
    tau = numpyro.sample('tau', dist.HalfNormal(tau_scale).to_event(1))          # one step size per outcome
    za = numpyro.sample('za', dist.Normal(0, 1).expand([G - 1, K - 1]).to_event(2))
    alpha = jnp.concatenate([a0[None], a0[None] + jnp.cumsum(tau[None] * za, 0)], 0)
    su = numpyro.sample('sigma_u', dist.HalfNormal(1.0))
    zu = numpyro.sample('zu', dist.Normal(0, 1).expand([R, K - 1]).to_event(2))
    sw = numpyro.sample('sigma_w', dist.HalfNormal(0.7))
    zw = numpyro.sample('zw', dist.Normal(0, 1).expand([P, K - 1]).to_event(2))
    eta = alpha[None] + ((su * zu)[preg] + sw * zw)[:, None, :]
    eta = jnp.concatenate([eta, jnp.zeros((P, G, 1))], axis=-1)
    theta0 = jax.nn.softmax(eta, axis=-1)
    numpyro.deterministic('theta0', theta0)
    full = jnp.concatenate([alpha, jnp.zeros((G, 1))], -1)
    for t, (cols, ref, w, c_obs, sd) in enumerate(polls):
        lr = full[:, cols] - full[:, ref][:, None]
        c = lr - (w[:, None] * lr).sum(0)
        numpyro.sample('poll%d' % t, dist.Normal(c, sd).to_event(2), obs=c_obs)
    p = jnp.einsum('cg,cgk->ck', A, theta0[prov])
    kappa = numpyro.sample('kappa', dist.LogNormal(jnp.log(300.0), 1.5))
    numpyro.sample('obs', dist.Dirichlet(kappa * jnp.clip(p, 1e-6, 1.0)), obs=S)


def run_v3(A, Y, reg, prov, polls=(), tau_scale=None, seed=0, warmup=600, samples=500, ndraw=500, chains=2):
    """polls: list of dict(cols=[outcome idx], ref=outcome idx, c, sd, w) from poll_term (ref given as an outcome index).
    tau_scale: per-outcome prior scale of the age step (length K-1)."""
    rng = np.random.default_rng(seed)
    K = Y.shape[1]
    if tau_scale is None:
        tau_scale = np.array([0.6] * (K - 3) + [0.15, 0.15])
    R = int(reg.max()) + 1; P = int(prov.max()) + 1
    preg = np.zeros(P, int); preg[prov] = reg
    pp = tuple((jnp.array(p['cols']), int(p['ref']), jnp.array(p['w']), jnp.array(p['c']), jnp.array(p['sd'])) for p in polls)
    mcmc = MCMC(NUTS(stage1_model_v3, target_accept_prob=0.9), num_warmup=warmup, num_samples=samples,
                num_chains=chains, chain_method='parallel', progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), jnp.array(A), jnp.array(shares(Y)), jnp.array(reg), R, jnp.array(prov), P, jnp.array(preg),
             jnp.array(tau_scale), pp)
    th0 = np.asarray(mcmc.get_samples()['theta0'])
    sv = estimate_sigma_v(A, Y, prov, th0.mean(0), rng) * SV_MULT
    draws = stage2(A, Y, prov, th0, sv, rng, ndraw=ndraw)
    return mcmc, sv, draws
