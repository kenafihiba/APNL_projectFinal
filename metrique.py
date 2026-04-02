"""
Affichage des métriques de benchmark en terminal — tableau formaté.
À ajouter à la fin de benchmark_complet.py (ou importer les SOLVERS/problems depuis ce fichier).
"""

import numpy as np
import time
from scipy.optimize import rosen, rosen_der


# ══════════════════════════════════════════════════════════════════════
# COPIER ICI LES SOLVEURS ET make_problems() DE benchmark_complet.py
# OU faire : from benchmark_complet import SOLVERS, make_problems
# ══════════════════════════════════════════════════════════════════════
try:
    from Profils_Performance import SOLVERS, make_problems
except ImportError:
    raise ImportError("Placez ce fichier dans le même dossier que benchmark_complet.py")


# ══════════════════════════════════════════════════════════════════════
# 1. COLLECTE DES MÉTRIQUES
# ══════════════════════════════════════════════════════════════════════

def run_full_metrics(problems, tol=1e-6):
    """
    Pour chaque solveur et chaque problème, collecte :
      - n_iter      : nombre d'itérations
      - n_grad      : nombre d'évaluations de gradient
      - cpu_time    : temps wall-clock (secondes)
      - final_grad  : norme du gradient final (précision)
      - success     : booléen (convergence vérifiée)

    Retourne
    --------
    dict : { solver_name -> { metric -> list of values (NaN si échec) } }
    """
    results = {
        name: {'n_iter': [], 'n_grad': [], 'cpu_time': [], 'final_grad': [], 'success': []}
        for name in SOLVERS
    }

    n_total = len(problems)

    for i, (f, grad, x0) in enumerate(problems):
        for name, solver in SOLVERS.items():
            # Compteur de gradient via wrapper
            grad_calls = [0]
            def grad_counted(x, _gc=grad_calls, _g=grad):
                _gc[0] += 1
                return _g(x)

            try:
                t0 = time.perf_counter()
                x_opt, f_opt, n_iter, hist_f, n_fevals = solver(f, grad_counted, x0, tol=tol)
                cpu = time.perf_counter() - t0

                norm_g = np.linalg.norm(grad(x_opt))
                ok = norm_g <= tol * 1e2

                results[name]['n_iter'].append(n_iter if ok else np.nan)
                results[name]['n_grad'].append(grad_calls[0] if ok else np.nan)
                results[name]['cpu_time'].append(cpu if ok else np.nan)
                results[name]['final_grad'].append(norm_g)
                results[name]['success'].append(ok)

            except Exception:
                results[name]['n_iter'].append(np.nan)
                results[name]['n_grad'].append(np.nan)
                results[name]['cpu_time'].append(np.nan)
                results[name]['final_grad'].append(np.nan)
                results[name]['success'].append(False)

    return results, n_total


# ══════════════════════════════════════════════════════════════════════
# 2. CALCUL DES STATISTIQUES AGRÉGÉES
# ══════════════════════════════════════════════════════════════════════

def aggregate(results, n_total):
    """
    Calcule pour chaque solveur :
      - iter_mean / iter_med : moyenne et médiane des itérations (problèmes convergés)
      - grad_mean / grad_med : idem pour évaluations de gradient
      - time_mean / time_med : idem pour temps CPU (ms)
      - prec_mean / prec_med : idem pour norme gradient finale
      - n_ok                 : nombre de problèmes résolus
      - robustness           : % de problèmes résolus
    """
    stats = {}
    for name, m in results.items():
        iters  = np.array(m['n_iter'],    dtype=float)
        grads  = np.array(m['n_grad'],    dtype=float)
        times  = np.array(m['cpu_time'],  dtype=float) * 1000   # → ms
        precs  = np.array(m['final_grad'],dtype=float)
        succ   = np.array(m['success'],   dtype=bool)

        n_ok = int(np.sum(succ))
        rob  = 100.0 * n_ok / n_total

        def safe_mean(a): v = a[~np.isnan(a)]; return np.mean(v)  if len(v) else np.nan
        def safe_med(a):  v = a[~np.isnan(a)]; return np.median(v) if len(v) else np.nan

        stats[name] = {
            'iter_mean' : safe_mean(iters),
            'iter_med'  : safe_med(iters),
            'grad_mean' : safe_mean(grads),
            'grad_med'  : safe_med(grads),
            'time_mean' : safe_mean(times),
            'time_med'  : safe_med(times),
            'prec_mean' : safe_mean(precs),
            'prec_med'  : safe_med(precs),
            'n_ok'      : n_ok,
            'robustness': rob,
        }
    return stats


# ══════════════════════════════════════════════════════════════════════
# 3. AFFICHAGE EN TERMINAL
# ══════════════════════════════════════════════════════════════════════

# Codes ANSI
BOLD    = "\033[1m"
RESET   = "\033[0m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
BLUE    = "\033[94m"
HEADER  = "\033[95m"
DIM     = "\033[2m"


def color_rob(v):
    """Colore la robustesse : vert ≥ 80%, jaune ≥ 50%, rouge sinon."""
    if v >= 80: return f"{GREEN}{v:5.1f}%{RESET}"
    if v >= 50: return f"{YELLOW}{v:5.1f}%{RESET}"
    return f"{RED}{v:5.1f}%{RESET}"


def color_best(val, best_val, fmt):
    """Met en gras+vert si c'est le meilleur (le plus petit)."""
    s = fmt.format(val)
    if abs(val - best_val) < 1e-9:
        return f"{BOLD}{GREEN}{s}{RESET}"
    return s


def fmt_sci(v):
    """Format scientifique compact."""
    if np.isnan(v): return "   N/A   "
    return f"{v:.2e}"


def fmt_f1(v):
    if np.isnan(v): return "  N/A  "
    return f"{v:7.1f}"


def fmt_f2(v):
    if np.isnan(v): return "   N/A  "
    return f"{v:8.2f}"


def print_metric_table(stats):
    solver_names = list(stats.keys())
    W = 14   # largeur colonne solveur

    # ─── Trouver les meilleurs (minimums) ───
    best = {}
    for key in ('iter_mean', 'iter_med', 'grad_mean', 'grad_med',
                'time_mean', 'time_med', 'prec_mean', 'robustness'):
        vals = [stats[n][key] for n in solver_names if not np.isnan(stats[n][key])]
        if key == 'robustness':
            best[key] = max(vals) if vals else np.nan   # max pour robustesse
        else:
            best[key] = min(vals) if vals else np.nan

    # ─── Largeur totale ───
    n_sol = len(solver_names)
    col = 12
    total_w = W + 1 + n_sol * (col + 1)
    sep   = "─" * total_w
    sep2  = "═" * total_w
    sep_t = "┼" + "─" * (W) + "┼" + ("─" * col + "┼") * n_sol
    top   = "┌" + "─" * W + "┬" + ("─" * col + "┬") * (n_sol - 1) + "─" * col + "┐"
    bot   = "└" + "─" * W + "┴" + ("─" * col + "┴") * (n_sol - 1) + "─" * col + "┘"

    def hdr_row(label):
        row = f"│{BOLD}{HEADER}{label:^{W}}{RESET}│"
        for n in solver_names:
            row += f"{BOLD}{CYAN}{n:^{col}}{RESET}│"
        return row

    def data_row(label, key, fmt_fn, high_is_best=False):
        is_rob = (key == 'robustness')
        row = f"│{DIM}{label:<{W}}{RESET}│"
        for n in solver_names:
            v = stats[n][key]
            if np.isnan(v):
                cell = f"{'N/A':^{col}}"
            elif is_rob:
                raw = f"{v:5.1f}%"
                if abs(v - best[key]) < 1e-6:
                    raw = f"{BOLD}{GREEN}{raw}{RESET}"
                elif v >= 50:
                    raw = f"{YELLOW}{raw}{RESET}"
                else:
                    raw = f"{RED}{raw}{RESET}"
                cell = f"{raw:^{col}}"
            else:
                formatted = fmt_fn(v)
                if not np.isnan(best.get(key, np.nan)) and abs(v - best[key]) < 1e-9 * max(1, abs(best[key])):
                    formatted = f"{BOLD}{GREEN}{formatted}{RESET}"
                cell = f"{formatted:^{col}}"
            row += cell + "│"
        return row

    def sub_row(label, key, fmt_fn):
        row = f"│{DIM}{label:<{W}}{RESET}│"
        for n in solver_names:
            v = stats[n][key]
            formatted = fmt_fn(v) if not np.isnan(v) else "N/A"
            row += f"{formatted:^{col}}│"
        return row

    def section(title):
        inner = f" {BOLD}{BLUE}{title}{RESET} "
        pad   = max(0, total_w - 2 - len(title) - 2)
        return "├" + "─" * 1 + inner + "─" * pad + "┤"

    # ════════════════════════════════════════
    print()
    print(f"{BOLD}{HEADER}  BENCHMARK — Métriques de performance des solveurs quasi-Newton{RESET}")
    print(f"{DIM}  Problèmes : Rosenbrock | Dimensions : 2, 5, 10, 20 | Points de départ : 5/dim{RESET}")
    print()
    print(top)
    print(hdr_row("Solveur"))
    print("├" + "─" * W + "┼" + ("─" * col + "┼") * (n_sol - 1) + "─" * col + "┤")

    # ── 1. Itérations ──
    print(section("1. Itérations (convergés seulement)"))
    print(data_row("  Moyenne         ", 'iter_mean', lambda v: f"{v:7.1f}"))
    print(sub_row( "  Médiane         ", 'iter_med',  lambda v: f"{v:7.1f}"))

    print("├" + "─" * W + "┼" + ("─" * col + "┼") * (n_sol - 1) + "─" * col + "┤")

    # ── 2. Évaluations de gradient ──
    print(section("2. Évaluations de gradient"))
    print(data_row("  Moyenne         ", 'grad_mean', lambda v: f"{v:7.1f}"))
    print(sub_row( "  Médiane         ", 'grad_med',  lambda v: f"{v:7.1f}"))

    print("├" + "─" * W + "┼" + ("─" * col + "┼") * (n_sol - 1) + "─" * col + "┤")

    # ── 3. Temps CPU ──
    print(section("3. Temps CPU  (ms)"))
    print(data_row("  Moyenne         ", 'time_mean', lambda v: f"{v:8.3f}"))
    print(sub_row( "  Médiane         ", 'time_med',  lambda v: f"{v:8.3f}"))

    print("├" + "─" * W + "┼" + ("─" * col + "┼") * (n_sol - 1) + "─" * col + "┤")

    # ── 4. Précision finale ──
    print(section("4. Précision  ||∇f||"))
    print(data_row("  Moyenne         ", 'prec_mean', fmt_sci))
    print(sub_row( "  Médiane         ", 'prec_med',  fmt_sci))

    print("├" + "─" * W + "┼" + ("─" * col + "┼") * (n_sol - 1) + "─" * col + "┤")

    # ── 5. Robustesse ──
    print(section("5. Robustesse"))
    print(data_row("  % succès        ", 'robustness', None, high_is_best=True))
    # Ligne count succès/total
    row = f"│{'  Succès/Total':<{W}}│"
    for n in solver_names:
        s = f"{stats[n]['n_ok']}/{len(problems)}"
        row += f"{s:^{col}}│"
    print(row)

    print(bot)

    # ── Légende ──
    print(f"\n  {BOLD}{GREEN}■{RESET} Meilleure valeur dans la catégorie   "
          f"{YELLOW}■{RESET} Robustesse ≥ 50 %   "
          f"{RED}■{RESET} Robustesse < 50 %")
    print(f"  {DIM}Moyenne/Médiane calculées sur les problèmes convergés uniquement.{RESET}\n")


# ══════════════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Génération des problèmes de test...")
    problems = make_problems(dims=[2, 5, 10, 20], n_starts=5, seed=42)
    print(f"  → {len(problems)} problèmes\n")

    print("Exécution du benchmark (patience)...")
    results, n_total = run_full_metrics(problems, tol=1e-6)

    stats = aggregate(results, n_total)
    print_metric_table(stats)