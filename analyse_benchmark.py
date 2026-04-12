"""
Analyse complète du fichier benchmark_results.csv
run avec  : python analyse_benchmark.py benchmark_results.csv
"""

import pandas as pd
import numpy as np
import sys

# ─── Chargement ───────────────────────────────────────────────────────────────
csv_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
df = pd.read_csv(csv_path)

SEP  = "=" * 65
SEP2 = "-" * 65

# ══════════════════════════════════════════════════════════════════════
# 0. STRUCTURE DU BENCHMARK
# ══════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  0. STRUCTURE DU BENCHMARK")
print(SEP)
dims     = sorted(df['n'].unique())
problems = df['problème'].nunique()
runs     = len(df)
print(f"  Dimensions testées  : {dims}")
print(f"  Problèmes uniques   : {problems}")
print(f"  Runs totaux         : {runs}")
print()
for n in dims:
    sub  = df[df['n'] == n]
    probs = sub['problème'].unique()
    sols  = sub['solveur'].unique()
    print(f"  n={n:>4} : {len(probs)} problèmes × {len(sols)} solveur(s) = {len(probs)*len(sols)} runs")
    for p in sorted(probs):
        print(f"           - {p}")

# ══════════════════════════════════════════════════════════════════════
# 1. RÉSUMÉ GLOBAL PAR SOLVEUR (toutes dimensions)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  1. RÉSUMÉ GLOBAL PAR SOLVEUR (toutes dimensions)")
print(SEP)
g = df.groupby('solveur').agg(
    Runs      =('succès', 'count'),
    Succès    =('succès', 'sum'),
    Robustesse=('succès', lambda x: f"{100*x.mean():.1f}%"),
    Iters_moy =('iters',  lambda x: f"{x.mean():.1f}"),
    Nfev_moy  =('nfev',   lambda x: f"{x.mean():.1f}"),
    CPU_moy   =('cpu',    lambda x: f"{x.mean():.4f}s"),
).reset_index()
print(g.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════
# 2. RÉSUMÉ PAR SOLVEUR × DIMENSION
# ══════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  2. ROBUSTESSE PAR SOLVEUR × DIMENSION")
print(SEP)
pivot_rob = df.pivot_table(
    index='solveur', columns='n',
    values='succès', aggfunc=lambda x: f"{100*x.mean():.0f}%"
)
print(pivot_rob.to_string())

# ══════════════════════════════════════════════════════════════════════
# 3. ITÉRATIONS PAR KAPPA (quadratiques uniquement)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  3. ITÉRATIONS PAR KAPPA — QUADRATIQUES (n=50)")
print(SEP)
quad50 = df[(df['catégorie'] == 'quadratic') & (df['n'] == 50)]
pivot_kappa = quad50.pivot_table(
    index='solveur', columns='kappa',
    values='iters', aggfunc='first'
)
pivot_kappa.columns = [f"κ={int(c)}" for c in pivot_kappa.columns]
print(pivot_kappa.astype(int).to_string())

# ══════════════════════════════════════════════════════════════════════
# 4. ITÉRATIONS MOYENNES PAR CATÉGORIE × DIMENSION
# ══════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  4. ITÉRATIONS MOYENNES PAR CATÉGORIE (n=50)")
print(SEP)
df50 = df[df['n'] == 50]
cat50 = df50.pivot_table(
    index='solveur', columns='catégorie',
    values='iters', aggfunc='mean'
).round(1)
print(cat50.to_string())

print(f"\n{SEP2}")
print("  4b. ITÉRATIONS MOYENNES PAR CATÉGORIE (n=100)")
print(SEP2)
df100 = df[df['n'] == 100]
cat100 = df100.pivot_table(
    index='solveur', columns='catégorie',
    values='iters', aggfunc='mean'
).round(1)
print(cat100.to_string())

# ══════════════════════════════════════════════════════════════════════
# 5. PROFILS DOLAN-MORÉ (métrique : iters)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  5. PROFILS DOLAN-MORÉ — iters (n=50, 8 problèmes)")
print(SEP)

def dolan_more(df_sub, metric='iters'):
    problems = df_sub['problème'].unique()
    solvers  = df_sub['solveur'].unique()
    ratios   = {s: [] for s in solvers}
    for prob in problems:
        sub  = df_sub[df_sub['problème'] == prob]
        vals = {}
        for _, row in sub.iterrows():
            vals[row['solveur']] = row[metric] if row['succès'] else float('inf')
        ok = [v for v in vals.values() if v != float('inf')]
        if not ok:
            continue
        best = min(ok)
        for s in solvers:
            r = vals.get(s, float('inf'))
            ratios[s].append(r / best if r != float('inf') else float('inf'))
    return ratios

taus = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0]
ratios50 = dolan_more(df[df['n'] == 50], 'iters')

header = f"{'Solveur':<22}" + "".join(f"  τ={t:<4}" for t in taus)
print(header)
print("-" * len(header))
for s, r in ratios50.items():
    r_arr = np.array(r)
    row = f"{s:<22}"
    for t in taus:
        rho = np.mean(r_arr <= t)
        row += f"  {rho:.3f}"
    # plafond max
    r_fin = np.mean(r_arr <= 1e9)
    row += f"  (max={r_fin:.3f})"
    print(row)

# ══════════════════════════════════════════════════════════════════════
# 6. DÉTAIL PAR PROBLÈME — succès / iters / cpu
# ══════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  6. DÉTAIL PAR PROBLÈME (n=50)")
print(SEP)
for prob in sorted(df[df['n']==50]['problème'].unique()):
    sub = df[(df['n']==50) & (df['problème']==prob)][
        ['solveur','succès','iters','nfev','cpu','grad_norm']
    ].copy()
    sub['cpu'] = sub['cpu'].map(lambda x: f"{x:.4f}s")
    sub['grad_norm'] = sub['grad_norm'].map(lambda x: f"{x:.2e}")
    sub['succès'] = sub['succès'].map(lambda x: "✓" if x else "✗")
    print(f"\n  ── {prob}")
    print(sub.to_string(index=False))

print(f"\n{SEP}")
print("  FIN DE L'ANALYSE")
print(SEP)