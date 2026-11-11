"""Modelled party-list vote by age group in every 2026 constituency.

Method (poll-informed post-stratification, calibrated to official results):
 1. National age gradient. A small Bayesian model learns how much more or less each party is
    supported by each age group, RELATIVE to Bhumjaithai and to the whole electorate, from
    published poll figures (poll_prior). Poll overall levels are never used. Where polls are silent
    (Kla Tham, small parties) a tight prior assumes little age difference.
 2. Local fit. For every draw, each constituency starts from that gradient plus random local
    variation, then iterative proportional fitting adjusts it so that (a) every age group's shares
    sum to one and (b) the age groups add up exactly to the constituency's official party-list result.
    IPF keeps the age contrasts and takes the level from the real count.
Assumption: turnout is the same in every age group within a constituency (no turnout-by-age data).
Output: groups.js (GROUPS) with mean, 5th and 95th percentiles and P(top choice), per mille.
"""
import json, re, sys, os
import numpy as np
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax.scipy.stats import norm

ROOT = os.environ.get('TEA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'ei'))
from results import load, LABELS
from ei import ipf

PARTIES = LABELS[:6]                      # BJT, PP, PTP, Kla Tham, Democrat, Other parties
G = 4
BANDS = ['18–29', '30–44', '45–59', '60+']
W_POLL = np.array([.193, .280, .323, .204])          # KPI sample shares by generation
SIGMA_LOCAL = 0.30                                   # assumed local variation of the age gradient (logit scale)
nan = np.nan

# ---- published poll figures (party proxy: preferred PM for KPI) ------------------------------------
# columns: BJT, PP, PTP, Dem   (KPI Poll, 8-11 Jan 2026, n=2,000; Thairath 2909223)
KPI = np.array([[11.3, 33.8, 8.1, nan],
                [14.1, 19.6, 10.8, nan],
                [21.2, 11.9, 11.4, nan],
                [20.4, nan, 14.1, 15.1]])
KPI_ALL = np.array([16.9, 18.8, 10.9, 10.2])
# not in a group's top three, so below the third-placed figure (upper bounds), as (group, column, bound)
KPI_BOUNDS = [(0, 3, 8.1), (1, 3, 10.8), (2, 3, 11.4), (3, 1, 14.1)]
# columns: BJT, PP, PTP, Dem, Other (remainder)   (NIDA Poll, Jan 2026, ages 18-25; ISEAS Perspective 2026/34)
NIDA_Y = np.array([11.5, 69.1, 7.2, 5.6, 6.6])
NIDA_ALL = np.array([22.6, 34.2, 16.2, 13.2, 13.8])
KPI_COLS = [0, 1, 2, 4]; NIDA_COLS = [0, 1, 2, 4, 5]
BIAS_KPI, BIAS_NIDA, N_NIDA_Y = 0.40, 0.45, 250


def lr(q):
    return np.log(q) - np.log(q[..., :1])        # log ratio vs BJT (first column)


def gradient_model():
    # alpha[g, k]: log-odds vs BJT for the 6 parties (BJT fixed at 0)
    tau = numpyro.sample('tau', dist.HalfNormal(jnp.array([0.7, 0.7, 0.25, 0.7, 0.25])).to_event(1))  # PP PTP KT Dem Other
    a0 = numpyro.sample('a0', dist.Normal(0, 3).expand([5]).to_event(1))
    z = numpyro.sample('z', dist.Normal(0, 1).expand([G - 1, 5]).to_event(2))
    a = jnp.concatenate([a0[None], a0[None] + jnp.cumsum(tau[None] * z, 0)], 0)
    alpha = jnp.concatenate([jnp.zeros((G, 1)), a], 1)                       # G x 6
    numpyro.deterministic('alpha', alpha)
    w = jnp.array(W_POLL)
    cm = alpha - (w[:, None] * alpha).sum(0)                                # contrast vs whole electorate
    # KPI observed cells
    c_obs = lr(KPI) - lr(KPI_ALL)[None]
    n = 2000 * W_POLL
    sd = np.sqrt(1 / (n[:, None] * KPI / 100) + 1 / (n[:, None] * KPI[:, :1] / 100) + BIAS_KPI ** 2)
    m = np.isfinite(c_obs); m[:, 0] = False
    model_c = cm[:, KPI_COLS]
    numpyro.factor('kpi', jnp.sum(jnp.where(m, norm.logpdf(jnp.nan_to_num(c_obs), model_c, jnp.nan_to_num(sd, nan=1.0)), 0.0)))
    # KPI upper bounds (soft)
    for g, j, b in KPI_BOUNDS:
        ub = np.log(b) - np.log(KPI[g, 0]) - lr(KPI_ALL)[j]
        numpyro.factor('kb%d%d' % (g, j), norm.logcdf((ub - model_c[g, j]) / BIAS_KPI))
    # NIDA, youngest group only
    cn = lr(NIDA_Y) - lr(NIDA_ALL)
    sdn = np.sqrt(1 / (N_NIDA_Y * NIDA_Y / 100) + 1 / (N_NIDA_Y * NIDA_Y[0] / 100) + BIAS_NIDA ** 2)
    numpyro.factor('nida', jnp.sum(norm.logpdf(cn[1:], cm[0, NIDA_COLS][1:], sdn[1:])))


def main():
    ids, Y, N, _ = load()
    raw = {}
    for l in open(os.path.join(ROOT, 'data', 'age', 'age_by_constituency_2569.txt')):
        if l.startswith('#'): continue
        q = l.split(); raw[q[0]] = np.array(list(map(float, q[1:5])))
    A = np.array([raw[k] / raw[k].sum() for k in ids])
    V = Y[:, :6] / Y[:, :6].sum(1, keepdims=True)          # official party-list shares among valid party votes
    voters = Y[:, :6].sum(1)

    mc = MCMC(NUTS(gradient_model, target_accept_prob=0.95), num_warmup=1500, num_samples=1000, num_chains=2,
              chain_method='sequential', progress_bar=False)
    mc.run(jax.random.PRNGKey(11))
    from numpyro.diagnostics import summary
    sm = summary(mc.get_samples(group_by_chain=True))
    rhat = float(max(np.max(np.asarray(v['r_hat'])) for v in sm.values()))
    div = int(mc.get_extra_fields()['diverging'].sum())
    alpha = np.asarray(mc.get_samples()['alpha'])                  # S x G x 6
    rng = np.random.default_rng(2026)
    D = 400
    idx = rng.choice(len(alpha), D, replace=False)
    draws = np.empty((D, len(ids), G, 6), np.float32)
    for d, s in enumerate(idx):
        base = alpha[s][None] + rng.normal(0, SIGMA_LOCAL, (len(ids), G, 6))
        t = np.exp(base - base.max(-1, keepdims=True)); t /= t.sum(-1, keepdims=True)
        draws[d] = ipf(t, A, V, iters=300)
    fit = np.abs(np.einsum('cg,dcgk->dck', A, draws) - V[None]).max()

    q = lambda x: np.rint(np.asarray(x) * 1000).astype(int)
    mean = draws.mean(0); lo, hi = np.percentile(draws, [5, 95], axis=0)
    top = draws.argmax(-1); ptop = np.stack([(top == k).mean(0) for k in range(6)], -1)
    W = voters[:, None] * A                                         # estimated voters per constituency and group
    nat = np.einsum('cg,dcgk->dgk', W, draws) / W.sum(0)[None, :, None]
    # national ranges: draws differ in the gradient; widen to reflect poll bias beyond what the model encodes
    nm = nat.mean(0); nlo, nhi = np.percentile(nat, [5, 95], axis=0)
    out = dict(
        meta=dict(model='poll-informed post-stratification, calibrated to official results', election=2026, ballot='party-list',
                  sigma_local=SIGMA_LOCAL, draws=D, rhat_max=round(rhat, 3), divergences=div, max_fit_error=float(fit),
                  assumptions=['Turnout is the same for every age group within a constituency.',
                               'Polls show how party support differs between age groups. Their overall levels are not used.',
                               'KPI figures are preferred prime minister, used as a stand-in for party.',
                               'Polls do not report Kla Tham or small parties by age, so we assume little difference.',
                               'Poll age bands (18–25, 18–27, 28–43, 44–59, 60+) differ slightly from ours.'],
                  sources=['ECT 2026 result files (party-list ballot)', 'DOPA registered population by age, January 2026',
                           'KPI Poll, 8–11 January 2026, n=2,000 (Thairath, 9 Jan 2026)', 'NIDA Poll, January 2026, ages 18–25 (ISEAS Perspective 2026/34)']),
        ids=ids, groups=BANDS, parties=PARTIES,
        share=q(A).T.tolist(),
        everyone=q(V).T.tolist(),
        mean=q(mean).transpose(1, 2, 0).tolist(), lo=q(lo).transpose(1, 2, 0).tolist(), hi=q(hi).transpose(1, 2, 0).tolist(),
        ptop=q(ptop).transpose(1, 2, 0).tolist(),
        nat=dict(mean=q(nm).tolist(), lo=q(nlo).tolist(), hi=q(nhi).tolist()),
        natAll=q(Y[:, :6].sum(0) / Y[:, :6].sum()).tolist())
    with open(os.path.join(ROOT, 'groups.js'), 'w') as f:
        f.write('const GROUPS=' + json.dumps(out, ensure_ascii=False, separators=(',', ':')) + ';\n')
    print('rhat', round(rhat, 3), 'div', div, 'fit', fit)
    for g in range(G):
        print('%-6s' % BANDS[g], ' '.join('%s %4.1f (%4.1f-%4.1f)' % (PARTIES[k][:5], 100 * nm[g, k], 100 * nlo[g, k], 100 * nhi[g, k]) for k in range(6)))
    print('everyone', ' '.join('%s %4.1f' % (PARTIES[k][:5], v / 10) for k, v in enumerate(out['natAll'])))
    tops = [(np.asarray(out['ptop'])[g].argmax(0)) for g in range(G)]
    for g in range(G):
        mx = np.asarray(out['ptop'])[g].max(0)
        print(BANDS[g], 'top-choice counts', {PARTIES[k][:5]: int((tops[g] == k).sum()) for k in range(6)}, 'confident(>=900):', int((mx >= 900).sum()))


if __name__ == '__main__':
    main()
