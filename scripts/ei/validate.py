"""Test the engine on a simulated election where the true group behaviour is known.

The simulation uses the real 2026 constituency structure (400 constituencies, real regions,
real numbers of eligible voters) but invented age profiles and invented group preferences.
We hide the truth, give the engine only constituency totals, and check how close it gets.
"""
import json, re, sys, time, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get('TEA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ei import run, run_v2, run_v3, summarise

import os
rng = np.random.default_rng(int(os.environ.get('SEED', 42)))
SVT = float(os.environ.get('SV_TRUE', 0.3))
s = open(os.path.join(ROOT, 'data.js')).read()
D = json.loads(re.search(r'const DATA=(.*?);\nconst GEO', s, re.S).group(1))
cons = sorted(D['cons'].values(), key=lambda c: c['id'])
regs = sorted({D['prov'][c['p']]['reg'] for c in cons})
reg = np.array([regs.index(D['prov'][c['p']]['reg']) for c in cons])
pcodes = sorted({c['p'] for c in cons}); prov = np.array([pcodes.index(c['p']) for c in cons])
USEPROV = os.environ.get('PROV', '0') == '1'
N = np.array([c['el'] for c in cons])
C, G, K, R = len(cons), 4, 8, len(regs)   # groups: 18-29, 30-44, 45-59, 60+; outcomes: 6 parties/blocs, spoiled, did not vote

# invented age profiles: regional means plus constituency variation
base = {'Bangkok': [.22, .30, .27, .21], 'Central': [.19, .27, .29, .25], 'East': [.21, .29, .28, .22], 'West': [.18, .26, .29, .27],
        'North': [.16, .24, .29, .31], 'Northeast': [.18, .25, .29, .28], 'South': [.20, .27, .28, .25]}
A = np.array([rng.dirichlet(np.array(base[regs[r]]) * 120) for r in reg])
if os.environ.get('REAL_A') == '1':
    RA = {}
    for l in open(os.path.join(ROOT, 'data', 'age', 'age_by_constituency_2569.txt')):
        if l.startswith('#'): continue
        q = l.split(); RA[q[0]] = np.array(list(map(float, q[1:5])))
    A = np.array([RA[c['id']] / RA[c['id']].sum() for c in cons])

# invented truth: national age gradient, regional effects, constituency effects
alpha = rng.normal(0, 1, (G, K - 1)); alpha[:, -1] += 0.3            # 'did not vote' (vs reference 'spoiled') baseline
alpha[0, 1] += 1.0; alpha[3, 1] -= 0.8                                # party 2 strongly age-graded
alpha[3, 0] += 0.6; alpha[0, -1] += 0.5                               # older voters favour party 1; young abstain more
u = rng.normal(0, 0.6, (R, G, K - 1)); v = rng.normal(0, SVT, (C, G, K - 1))
if os.environ.get('SMOOTH') == '1':
    a0 = rng.normal(0, 1, K - 1); steps = rng.normal(0, float(os.environ.get('STEP', 0.45)), (G - 1, K - 1)); alpha = np.concatenate([a0[None], a0[None] + np.cumsum(steps, 0)]); alpha[:, -1] += 0.3
    ug = rng.normal(0, 0.6, (R, K - 1)); u = ug[:, None, :] + rng.normal(0, 0.15, (R, G, K - 1))
PW = float(os.environ.get('PW_TRUE', 0)); w = rng.normal(0, PW, (len(pcodes), 1 if os.environ.get('SMOOTH') == '1' else G, K - 1)); u = u[reg] + w[prov]
CONF = float(os.environ.get('CONF', 0))
if CONF > 0:
    yshare = A[:, :2].sum(1); z = yshare.copy()
    for p_ in set(prov): m_ = prov == p_; z[m_] -= z[m_].mean()
    z = z / z.std() + 0.5 * (yshare - yshare.mean()) / yshare.std()
    beta = rng.normal(0, CONF, K - 1)
    v = v + (z[:, None] * beta[None, :])[:, None, :]
eta = np.concatenate([alpha[None] + u + v, np.zeros((C, G, 1))], -1)
theta = np.exp(eta - eta.max(-1, keepdims=True)); theta /= theta.sum(-1, keepdims=True)
Y = np.zeros((C, K))
for c in range(C):
    for g in range(G):
        Y[c] += rng.multinomial(int(N[c] * A[c, g]), theta[c, g])

t0 = time.time()
POLL = os.environ.get('POLL')
poll = None
if POLL is not None:
    # a simulated pre-election poll: right direction, wrong overall level, noisy gradient, ~500 per age group
    W0 = N[:, None] * A
    natv = np.einsum('cg,cgk->gk', W0, theta) / W0.sum(0)[:, None]
    q = natv[:, :6] / natv[:, :6].sum(1, keepdims=True)
    q = q * np.exp(rng.normal(0, 0.3, 6))[None, :] * np.exp(rng.normal(0, float(POLL), q.shape)); q /= q.sum(1, keepdims=True)
    nn = np.array([500] * G); q = np.array([rng.multinomial(n_, q_) / n_ for n_, q_ in zip(nn, q)])
    poll = dict(q=q, n=nn, cols=list(range(6)), ref=1)
if os.environ.get('V3') == '1':
    m, sv, draws = run_v3(A, Y, reg, prov, poll=poll, tau_scale=float(os.environ.get('TAU', 0.5)), warmup=int(sys.argv[1]) if len(sys.argv) > 1 else 600, samples=int(sys.argv[2]) if len(sys.argv) > 2 else 500)
else:
  m, sv, draws = (run_v2(A, Y, reg, prov, warmup=int(sys.argv[1]) if len(sys.argv) > 1 else 600, samples=int(sys.argv[2]) if len(sys.argv) > 2 else 500) if os.environ.get('V2') == '1' else run(A, Y, reg, prov=prov if USEPROV else None, warmup=int(sys.argv[1]) if len(sys.argv) > 1 else 600, samples=int(sys.argv[2]) if len(sys.argv) > 2 else 500))
print('sigma_v chosen', round(sv,3), '(truth', SVT, ')')
el = time.time() - t0
mean, lo, hi, ptop = summarise(draws)
err = np.abs(mean - theta) * 100
cover = ((theta >= lo) & (theta <= hi)).mean()
top_true = theta.argmax(-1); top_est = mean.argmax(-1)
conf = ptop.max(-1)
naive = np.abs(Y / Y.sum(1, keepdims=True))[:, None, :].repeat(G, 1) - theta   # baseline: assume every group voted like the whole constituency
print(json.dumps(dict(seconds=round(el), mean_abs_error_pts=round(float(err.mean()), 2), median_abs_error_pts=round(float(np.median(err)), 2),
      baseline_same_as_everyone_error_pts=round(float(np.abs(naive).mean() * 100), 2), interval90_coverage=round(float(cover), 3),
      top_choice_correct=round(float((top_true == top_est).mean()), 3),
      top_choice_correct_when_conf_over_90=round(float((top_true == top_est)[conf > .9].mean()), 3), share_conf_over_90=round(float((conf > .9).mean()), 3),
      national_group_error_pts=[round(float(x), 2) for x in (np.abs(np.einsum('c,cg,cgk->gk', N, A, mean) - np.einsum('c,cg,cgk->gk', N, A, theta)) / np.einsum('c,cg->g', N, A)[:, None] * 100).max(1)],
      divergences=int(m.get_extra_fields()['diverging'].sum()))))
from numpyro.diagnostics import summary
sm=m.get_samples(group_by_chain=True)
for k in ['sigma_u','kappa']:
    if k not in sm: continue
    d=summary({k:sm[k]})[k];print(k,'mean',round(float(d['mean']),3),'r_hat',round(float(d['r_hat']),3),'n_eff',round(float(d['n_eff'])))
W = N[:, None] * A
nat_true = np.einsum('cg,cgk->gk', W, theta) / W.sum(0)[:, None]
nat_d = np.einsum('cg,dcgk->dgk', W, draws) / W.sum(0)[None, :, None]
nlo, nhi = np.percentile(nat_d, [5, 95], axis=0)
print('national 90% coverage', round(float(((nat_true >= nlo) & (nat_true <= nhi)).mean()), 3), 'mean width pts', round(float((nhi - nlo).mean() * 100), 2), 'mean error pts', round(float(np.abs(nat_d.mean(0) - nat_true).mean() * 100), 2))
mu = nat_d.mean(0)
for f in (1.25, 1.5, 1.75, 2.0, 2.5):
    l2 = mu - f * (mu - nlo); h2 = mu + f * (nhi - mu)
    print('nat_f', f, round(float(((nat_true >= l2) & (nat_true <= h2)).mean()), 3))
