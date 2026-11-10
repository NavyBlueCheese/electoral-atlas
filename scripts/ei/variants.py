"""Sensitivity check on real data: does the age pattern survive reasonable changes to the model?

Usage: python3 variants.py NAME   (see VARIANTS below). Writes out/variant_NAME.json
"""
import json, re, sys, time, os
import numpy as np
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get('TEA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ei import shares
from results import load, LABELS

numpyro.set_host_device_count(2)


def model(A, S, R, prov, P, preg, tau_scale, si_scale):
    C, G = A.shape; K = S.shape[1]
    a0 = numpyro.sample('a0', dist.Normal(0, 2).expand([K - 1]).to_event(1))
    tau = numpyro.sample('tau', dist.HalfNormal(tau_scale))
    za = numpyro.sample('za', dist.Normal(0, 1).expand([G - 1, K - 1]).to_event(2))
    alpha = jnp.concatenate([a0[None], a0[None] + tau * jnp.cumsum(za, 0)], 0)
    su = numpyro.sample('sigma_u', dist.HalfNormal(1.0))
    zu = numpyro.sample('zu', dist.Normal(0, 1).expand([R, K - 1]).to_event(2))
    sw = numpyro.sample('sigma_w', dist.HalfNormal(0.7))
    zw = numpyro.sample('zw', dist.Normal(0, 1).expand([P, K - 1]).to_event(2))
    eta = alpha[None] + ((su * zu)[preg] + sw * zw)[:, None, :]
    if si_scale > 0:
        si = numpyro.sample('sigma_i', dist.HalfNormal(si_scale))
        zi = numpyro.sample('zi', dist.Normal(0, 1).expand([R, G, K - 1]).to_event(3))
        eta = eta + (si * zi)[preg]
    eta = jnp.concatenate([eta, jnp.zeros((P, G, 1))], -1)
    theta0 = jax.nn.softmax(eta, -1)
    numpyro.deterministic('alpha_nat', jax.nn.softmax(jnp.concatenate([alpha, jnp.zeros((G, 1))], -1), -1))
    p = jnp.einsum('cg,cgk->ck', A, theta0[prov])
    numpyro.deterministic('theta0', theta0)
    kappa = numpyro.sample('kappa', dist.LogNormal(jnp.log(300.0), 1.5))
    numpyro.sample('obs', dist.Dirichlet(kappa * jnp.clip(p, 1e-6, 1.0)), obs=S)


VARIANTS = {
    # name: (age grouping, tau prior scale, interaction prior scale, drop deep south)
    'g4_tau03':      ('4', 0.3, 0.0, False),
    'g4_tau03_noDS': ('4', 0.3, 0.0, True),
    'g3_tau03':      ('3', 0.3, 0.0, False),
    'g2_tau05':      ('2', 0.5, 0.0, False),
    'g2_tau05_noDS': ('2', 0.5, 0.0, True),
    'g3_tau03_int':  ('3', 0.3, 0.15, False),
}
MERGE = {'4': ([[0], [1], [2], [3]], ['18–29', '30–44', '45–59', '60+']),
         '3': ([[0, 1], [2], [3]], ['18–44', '45–59', '60+']),
         '2': ([[0, 1], [2, 3]], ['18–44', '45+'])}

if __name__ == '__main__':
    name = sys.argv[1]; warm = int(sys.argv[2]) if len(sys.argv) > 2 else 500; samp = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    grp, tau_s, si_s, noDS = VARIANTS[name]
    ids, Y, N, _ = load()
    raw = {}
    for l in open(os.path.join(ROOT, 'data', 'age', 'age_by_constituency_2569.txt')):
        if l.startswith('#'): continue
        q = l.split(); raw[q[0]] = np.array(list(map(float, q[1:5])))
    A4 = np.array([raw[k] / raw[k].sum() for k in ids])
    cols, glab = MERGE[grp]
    A = np.stack([A4[:, c].sum(1) for c in cols], 1)
    keep = np.array([not (noDS and k.split('_')[0] in ('94', '95', '96')) for k in ids])
    ids = [k for k, f in zip(ids, keep) if f]; A = A[keep]; Y = Y[keep]; N = N[keep]
    s = open(os.path.join(ROOT, 'data.js')).read()
    D = json.loads(re.search(r'const DATA=(.*?);\nconst GEO', s, re.S).group(1))
    regs = sorted({D['prov'][k.split('_')[0]]['reg'] for k in ids})
    reg = np.array([regs.index(D['prov'][k.split('_')[0]]['reg']) for k in ids])
    pcodes = sorted({k.split('_')[0] for k in ids}); prov = np.array([pcodes.index(k.split('_')[0]) for k in ids])
    P = len(pcodes); preg = np.zeros(P, int); preg[prov] = reg
    t0 = time.time()
    mc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warm, num_samples=samp, num_chains=2, chain_method='parallel', progress_bar=False)
    mc.run(jax.random.PRNGKey(7), jnp.array(A), jnp.array(shares(Y)), len(regs), jnp.array(prov), P, jnp.array(preg), tau_s, si_s)
    sm = mc.get_samples(group_by_chain=True)
    th = np.asarray(mc.get_samples()['theta0'])            # draws x P x G x K
    # national figure: weight province patterns by eligible people in each group
    W = np.zeros((P, A.shape[1]))
    for c in range(len(ids)): W[prov[c]] += N[c] * A[c]
    nat = np.einsum('pg,dpgk->dgk', W, th) / W.sum(0)[None, :, None]
    party = nat[..., :6] / nat[..., :6].sum(-1, keepdims=True)
    out = dict(name=name, groups=glab, seconds=round(time.time() - t0),
               tau=float(summary({'tau': sm['tau']})['tau']['mean']), tau_rhat=float(summary({'tau': sm['tau']})['tau']['r_hat']),
               theta_max_rhat=float(np.max(summary({'theta0': sm['theta0']})['theta0']['r_hat'])),
               divergences=int(mc.get_extra_fields()['diverging'].sum()),
               party=dict(mean=party.mean(0).round(3).tolist(), lo=np.percentile(party, 5, 0).round(3).tolist(), hi=np.percentile(party, 95, 0).round(3).tolist()),
               turnout=dict(mean=(1 - nat[..., 7]).mean(0).round(3).tolist(), lo=np.percentile(1 - nat[..., 7], 5, 0).round(3).tolist(), hi=np.percentile(1 - nat[..., 7], 95, 0).round(3).tolist()))
    json.dump(out, open('variant_%s.json' % name, 'w'), ensure_ascii=False)
    print(name, 'tau %.2f rhat %.3f thmax %.3f div %d  %ds' % (out['tau'], out['tau_rhat'], out['theta_max_rhat'], out['divergences'], out['seconds']))
    for g, lab in enumerate(glab):
        print('  %-6s' % lab, ' '.join('%s %4.1f' % (LABELS[k][:6], 100 * out['party']['mean'][g][k]) for k in range(6)), '| turnout %.1f' % (100 * out['turnout']['mean'][g]))
