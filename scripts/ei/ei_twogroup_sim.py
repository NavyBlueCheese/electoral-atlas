"""Simulation test for the two-group (where-people-live) model, using the real constituency
structure and the real group shares, with known truth.

Usage: python3 ei_twogroup_sim.py A.json GAP CONF SEED
GAP : size of the true group difference on the logit scale (e.g. 0.3 weak, 0.8 strong)
CONF: strength of a contextual effect that moves BOTH groups with the area's group share (0 = none)
Prints national error and 90% coverage for both groups, against the naive 'same as everyone' baseline.
"""
import json, os, re, sys
import numpy as np
import jax, jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.environ.get('TEA_OUT', 'out'); os.makedirs(OUT, exist_ok=True)
from ei import shares, stage2, estimate_sigma_v, SV_MULT
from variants import model
from results import load

path, GAP, CONF, SEED = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(SEED)
raw = json.load(open(path))
ids, Yr, N, _ = load()
keep = np.array([k in raw for k in ids]); ids = [k for k, f in zip(ids, keep) if f]; N = N[keep]
s1 = np.array([raw[k][0] / raw[k][1] for k in ids]).clip(0, 1); A = np.column_stack([s1, 1 - s1])
D = json.loads(re.search(r'const DATA=(.*?);\nconst GEO', open(os.path.join(ROOT, 'data.js')).read(), re.S).group(1))
regs = sorted({D['prov'][k.split('_')[0]]['reg'] for k in ids}); reg = np.array([regs.index(D['prov'][k.split('_')[0]]['reg']) for k in ids])
pcodes = sorted({k.split('_')[0] for k in ids}); prov = np.array([pcodes.index(k.split('_')[0]) for k in ids])
P = len(pcodes); preg = np.zeros(P, int); preg[prov] = reg
C, G, K = len(ids), 2, 8
a0 = rng.normal(0, 1, K - 1); a0[-1] += 0.3
alpha = np.stack([a0 + rng.normal(0, GAP, K - 1), a0])
geo = rng.normal(0, 0.6, (len(regs), K - 1))[reg] + rng.normal(0, 0.3, (P, K - 1))[prov]
z = (s1 - s1.mean()) / s1.std()
ctx = CONF * z[:, None] * rng.normal(0, 1, K - 1)[None]
eta = alpha[None] + (geo + ctx)[:, None, :] + rng.normal(0, 0.3, (C, G, K - 1))
eta = np.concatenate([eta, np.zeros((C, G, 1))], -1)
theta = np.exp(eta - eta.max(-1, keepdims=True)); theta /= theta.sum(-1, keepdims=True)
Y = np.zeros((C, K))
for c in range(C):
    for g in range(G):
        Y[c] += rng.multinomial(int(N[c] * A[c, g]), theta[c, g])
mc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=500, num_samples=400, num_chains=2, chain_method='parallel', progress_bar=False)
mc.run(jax.random.PRNGKey(SEED), jnp.array(A), jnp.array(shares(Y)), len(regs), jnp.array(prov), P, jnp.array(preg), 1.0, 0.0)
th = np.asarray(mc.get_samples()['theta0'])
sv = estimate_sigma_v(A, Y, prov, th.mean(0), rng) * SV_MULT
draws = stage2(A, Y, prov, th, sv, rng, ndraw=300)
W = N[:, None] * A
truth = np.einsum('cg,cgk->gk', W, theta) / W.sum(0)[:, None]
nat = np.einsum('cg,dcgk->dgk', W, draws) / W.sum(0)[None, :, None]
lo, hi = np.percentile(nat, [5, 95], 0); m = nat.mean(0)
naive = (Y / Y.sum(1, keepdims=True))
naive_nat = np.einsum('cg,ck->gk', W, naive) / W.sum(0)[:, None]
cmean = draws.mean(0); clo, chi = np.percentile(draws, [5, 95], 0)
print(json.dumps(dict(gap=GAP, conf=CONF, seed=SEED,
      national_err_pts=round(float(np.abs(m - truth).mean() * 100), 2), naive_national_err_pts=round(float(np.abs(naive_nat - truth).mean() * 100), 2),
      national_cov90=round(float(((truth >= lo) & (truth <= hi)).mean()), 3),
      national_cov90_widened=round(float(((truth >= m - 1.5 * (m - lo)) & (truth <= m + 1.5 * (hi - m))).mean()), 3),
      cons_err_pts=round(float(np.abs(cmean - theta).mean() * 100), 2), cons_naive_err_pts=round(float(np.abs(naive[:, None, :] - theta).mean() * 100), 2),
      cons_cov90=round(float(((theta >= clo) & (theta <= chi)).mean()), 3),
      true_gap_pp=round(float((truth[0, 1] / truth[0, :6].sum() - truth[1, 1] / truth[1, :6].sum()) * 100), 1))))
