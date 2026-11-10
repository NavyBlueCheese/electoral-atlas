"""Survey information on how support differs by age, used as priors by the v3 engine.

Only RELATIVE differences between age groups are used (e.g. how much stronger People's Party is
than Bhumjaithai among the young compared with everyone). The polls' overall levels are not used.

1. KPI Poll (King Prajadhipok's Institute), 8-11 January 2026, n = 2,000, face-to-face, multi-stage sample.
   Preferred prime minister by generation, top three per group (as reported by Thairath, 2026, news/politic/2909223).
   Gen Z 18-27, Gen Y 28-43, Gen X 44-59, Baby Boomer 60+. Used as a proxy for party preference:
   Natthaphong = People's Party, Anutin = Bhumjaithai, Yotchanan = Pheu Thai, Abhisit = Democrat.
   Sample shares by age from KPI Poll #07 report: 19.3 / 28.0 / 32.3 / 20.4 percent of 2,000.
2. NIDA Poll, January 2026, party-list vote intention, ages 18-25 vs all respondents
   (Panarat Anamwathana, ISEAS Perspective 2026/34, Tables 2-3).
Age bands differ slightly from ours (18-29, 30-44, 45-59, 60+); the poll-error allowance covers this.
"""
import numpy as np
from ei import poll_term

nan = np.nan
# outcome indices in the engine: 0 BJT, 1 PP, 2 PTP, 3 Kla Tham, 4 Democrat, 5 other parties, 6 spoiled/none, 7 did not vote


def kpi_generations():
    cols = [0, 1, 2]                                   # BJT, PP, PTP; reference BJT (known for every group)
    q = np.array([[11.3, 33.8, 8.1],                   # Gen Z
                  [14.1, 19.6, 10.8],                  # Gen Y
                  [21.2, 11.9, 11.4],                  # Gen X
                  [20.4, nan, 14.1]])                  # Boomers (PP not in their top three: unknown, below 14.1)
    allq = np.array([16.9, 18.8, 10.9])
    n = 2000 * np.array([.193, .280, .323, .204])
    t = poll_term(q, allq, n, n, ref=0, bias_sd=0.35)
    t.update(cols=cols, ref=0)
    return t


def nida_young():
    cols = [0, 1, 2, 4, 5]                             # BJT, PP, PTP, Democrat, everything else (incl. Kla Tham)
    q = np.array([[11.5, 69.1, 7.2, 5.6, 6.6], [nan] * 5, [nan] * 5, [nan] * 5])
    allq = np.array([22.6, 34.2, 16.2, 13.2, 13.8])
    n = np.array([250, 1, 1, 1])                       # 18-25 respondents not published; 250 is conservative
    t = poll_term(q, allq, n, [.2, .28, .3, .22], ref=0, bias_sd=0.45)
    t.update(cols=cols, ref=0)
    return t


def all_polls():
    return [kpi_generations(), nida_young()]
