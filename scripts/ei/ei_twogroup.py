"""Results-only ecological inference for a two-group split defined by where people live
(e.g. municipal vs non-municipal residents), with a sensitivity check across specifications.

Usage: python3 ei_twogroup.py A.json SPEC [warmup samples]
A.json: {constituency_id: [group1 count, total]}
SPEC: base | tight | inter | noBKK | noDS
Writes out/twogroup_<name>_<SPEC>.json with national shares (party-list, among valid party votes) and turnout.
"""
import json, os, re, sys, time
import numpy as np
import jax, jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.environ.get('TEA_OUT', 'out'); os.makedirs(OUT, exist_ok=True)
from ei import shares, stage2, estimate_sigma_v, SV_MULT
from results import load, LABELS
from variants import model

SPECS = {'base': (1.0, 0.0, None), 'tight': (0.5, 0.0, None), 'inter': (1.0, 0.2, None),
         'noBKK': (1.0, 0.0, ('10',)), 'noDS': (1.0, 0.0, ('94', '95', '96'))}

if __name__ == '__main__':
    path, spec = sys.argv[1], sys.argv[2]
    warm = int(sys.argv[3]) if len(sys.argv) > 3 else 600; samp = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    tau_s, si_s, drop = SPECS[spec]
    raw = json.load(open(path))
    ids, Y, N, _ = load()
    keep = np.array([k in raw and not (drop and k.split('_')[0] in drop) for k in ids])
    ids = [k for k, f in zip(ids, keep) if f]; Y = Y[keep]; N = N[keep]
    s1 = np.array([raw[k][0] / raw[k][1] for k in ids]).clip(0, 1)
    A = np.column_stack([s1, 1 - s1])
    D = json.loads(re.search(r'const DATA=(.*?);\nconst GEO', open(os.path.join(ROOT, 'data.js')).read(), re.S).group(1))
    regs = sorted({D['prov'][k.split('_')[0]]['reg'] for k in ids})
    reg = np.array([regs.index(D['prov'][k.split('_')[0]]['reg']) for k in ids])
    pcodes = sorted({k.split('_')[0] for k in ids}); prov = np.array([pcodes.index(k.split('_')[0]) for k in ids])
    P = len(pcodes); preg = np.zeros(P, int); preg[prov] = reg
    t0 = time.time()
    mc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warm, num_samples=samp, num_chains=2, chain_method='parallel', progress_bar=False)
    mc.run(jax.random.PRNGKey(5), jnp.array(A), jnp.array(shares(Y)), len(regs), jnp.array(prov), P, jnp.array(preg), tau_s, si_s)
    sm = mc.get_samples(group_by_chain=True)
    th = np.asarray(mc.get_samples()['theta0'])
    rng = np.random.default_rng(5)
    sv = estimate_sigma_v(A, Y, prov, th.mean(0), rng) * SV_MULT
    draws = stage2(A, Y, prov, th, sv, rng, ndraw=300)          # D x C x 2 x 8, exact fit
    W = N[:, None] * A
    nat = np.einsum('cg,dcgk->dgk', W, draws) / W.sum(0)[None, :, None]
    pv = nat[..., :6] / nat[..., :6].sum(-1, keepdims=True)
    name = os.path.splitext(os.path.basename(path))[0]
    out = dict(spec=spec, n=len(ids), sv=sv, seconds=round(time.time() - t0),
               theta_max_rhat=float(np.max(summary({'theta0': sm['theta0']})['theta0']['r_hat'])),
               divergences=int(mc.get_extra_fields()['diverging'].sum()),
               party=dict(mean=pv.mean(0).round(3).tolist(), lo=np.percentile(pv, 5, 0).round(3).tolist(), hi=np.percentile(pv, 95, 0).round(3).tolist()),
               turnout=(1 - nat[..., 7]).mean(0).round(3).tolist())
    json.dump(out, open(os.path.join(OUT, 'twogroup_%s_%s.json') % (name, spec), 'w'))
    if os.environ.get('SAVE') == '1':
        np.savez_compressed(os.path.join(OUT, 'twogroup_draws_%s.npz') % name, draws=draws.astype(np.float32), A=A, ids=np.array(ids), N=N)
    print(spec, 'n', len(ids), 'rhat %.3f div %d sv %.2f %ds' % (out['theta_max_rhat'], out['divergences'], sv, out['seconds']))
    for g, lab in enumerate(['group1', 'group2']):
        print('  %-7s' % lab, ' '.join('%s %4.1f (%4.1f-%4.1f)' % (LABELS[k][:6], 100 * out['party']['mean'][g][k], 100 * out['party']['lo'][g][k], 100 * out['party']['hi'][g][k]) for k in range(6)), '| turnout %.1f' % (100 * out['turnout'][g]))
