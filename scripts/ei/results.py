"""Real 2026 constituency outcomes for the engine (party-list ballot).

Outcomes per constituency, counted among ALL eligible voters:
  Bhumjaithai, People's Party, Pheu Thai, Kla Tham, Democrat, other parties,
  spoiled or 'none of the above', did not vote.
"""
import csv, os
import numpy as np

# folder of the election69 project's 2569 CSV files
SRC = os.environ.get('TEA_RESULTS', 'election69/data/2569/')
PARTIES = ['ภูมิใจไทย', 'ประชาชน', 'เพื่อไทย', 'กล้าธรรม', 'ประชาธิปัตย์']
LABELS = ['Bhumjaithai', "People's Party", 'Pheu Thai', 'Kla Tham', 'Democrat', 'Other parties', 'Spoiled or none', 'Did not vote']
PARTY_IDX = [0, 1, 2, 3, 4, 5]


def rd(f):
    with open(os.path.join(SRC, f), encoding='utf-8-sig') as fh:
        r = list(csv.reader(fh))
    return r[0], r[1:]


def load():
    h, rows = rd('constituencies.csv')
    cons = {r[0]: r for r in rows}
    ph, pl = rd('party_list_votes.csv')
    names = ph[1:]
    ids = sorted(cons, key=lambda k: (int(k.split('_')[0]), int(k.split('_')[1])))
    Y = np.zeros((len(ids), len(LABELS)))
    N = np.zeros(len(ids))
    checks = []
    plmap = {r[0]: r for r in pl}
    for i, k in enumerate(ids):
        c = cons[k]
        el, lt, lvalid, linv, lnone = int(c[3]), int(c[8]), int(c[9]), int(c[10]), int(c[11])
        votes = dict(zip(names, (int(x or 0) for x in plmap[k][1:])))
        main = [votes.get(p, 0) for p in PARTIES]
        tot = sum(votes.values())
        Y[i] = main + [tot - sum(main), linv + lnone, el - lt]
        N[i] = el
        checks.append((k, tot - lvalid, lt - (lvalid + linv + lnone)))
    return ids, Y, N, checks


if __name__ == '__main__':
    ids, Y, N, checks = load()
    bad = [c for c in checks if c[1] != 0 or c[2] != 0]
    print(len(ids), 'constituencies; rows where party votes != valid list ballots, or turnout != valid+spoiled+none:', len(bad), bad[:5])
    print('national shares of eligible voters:', dict(zip(LABELS, np.round(100 * Y.sum(0) / N.sum(), 2))))
    print('Y sums to eligible in every row:', bool(np.allclose(Y.sum(1), N)))
