"""
Benchmark complet — Profils de performance Dolan-Moré (2002)
Tous les solveurs sont intégrés directement, aucun import local requis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import rosen, rosen_der
from scipy.linalg import solve


# ══════════════════════════════════════════════════════════════════════
# 1. RECHERCHE LINÉAIRE (Wolfe conditions)
# ══════════════════════════════════════════════════════════════════════

def wolfe_line_search(f, grad, x, d, f0, g0, c1=1e-4, c2=0.9, alpha_max=1.0, max_iter=50):
    """
    Recherche linéaire satisfaisant les conditions de Wolfe (forte Wolfe).
    Retourne le pas alpha.
    """
    alpha = alpha_max
    alpha_lo, alpha_hi = 0.0, alpha_max
    gd = np.dot(g0, d)

    for _ in range(max_iter):
        x_new = x + alpha * d
        f_new = f(x_new)
        g_new = grad(x_new)

        # Condition d'Armijo
        if f_new > f0 + c1 * alpha * gd or (alpha < alpha_max and f_new >= f(x + alpha_lo * d)):
            alpha_hi = alpha
        else:
            # Condition de courbure forte
            if abs(np.dot(g_new, d)) <= c2 * abs(gd):
                return alpha, x_new, f_new, g_new
            if np.dot(g_new, d) * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha

        if abs(alpha_hi - alpha_lo) < 1e-14:
            break
        alpha = 0.5 * (alpha_lo + alpha_hi)

    x_new = x + alpha * d
    return alpha, x_new, f(x_new), grad(x_new)


def backtracking_line_search(f, grad, x, d, f0, g0, alpha0=1.0, rho=0.5, c=1e-4, max_iter=50):
    """Backtracking simple avec condition d'Armijo."""
    alpha = alpha0
    gd = np.dot(g0, d)
    for _ in range(max_iter):
        if f(x + alpha * d) <= f0 + c * alpha * gd:
            break
        alpha *= rho
    x_new = x + alpha * d
    return alpha, x_new, f(x_new), grad(x_new)


# ══════════════════════════════════════════════════════════════════════
# 2. SOLVEURS QUASI-NEWTON
# ══════════════════════════════════════════════════════════════════════

MAX_ITER = 2000
TOL_GRAD = 1e-6


def bfgs_solver(f, grad, x0, tol=1e-6):
    """BFGS avec mise à jour complète de l'inverse hessien."""
    n = len(x0)
    x = x0.copy().astype(float)
    H = np.eye(n)          # approximation inverse Hessien
    g = grad(x)
    f_val = f(x)
    hist_f = [f_val]
    nfev = 1

    for k in range(MAX_ITER):
        if np.linalg.norm(g) <= tol:
            break

        d = -H @ g
        # S'assurer que c'est une direction de descente
        if np.dot(d, g) >= 0:
            d = -g
            H = np.eye(n)

        alpha, x_new, f_new, g_new = wolfe_line_search(f, grad, x, d, f_val, g)
        nfev += 2

        s = x_new - x
        y = g_new - g
        sy = np.dot(s, y)

        if sy > 1e-12:
            rho_k = 1.0 / sy
            I = np.eye(n)
            sv = s[:, None]
            yv = y[:, None]
            A = I - rho_k * (sv @ yv.T)
            H = A @ H @ A.T + rho_k * (sv @ sv.T)

        x, g, f_val = x_new, g_new, f_new
        hist_f.append(f_val)

    return x, f_val, k + 1, hist_f, nfev


def dfp_solver(f, grad, x0, tol=1e-6):
    """DFP (Davidon-Fletcher-Powell)."""
    n = len(x0)
    x = x0.copy().astype(float)
    H = np.eye(n)
    g = grad(x)
    f_val = f(x)
    hist_f = [f_val]
    nfev = 1

    for k in range(MAX_ITER):
        if np.linalg.norm(g) <= tol:
            break

        d = -H @ g
        if np.dot(d, g) >= 0:
            d = -g
            H = np.eye(n)

        alpha, x_new, f_new, g_new = wolfe_line_search(f, grad, x, d, f_val, g)
        nfev += 2

        s = x_new - x
        y = g_new - g
        sy = np.dot(s, y)
        yHy = np.dot(y, H @ y)

        if sy > 1e-12 and yHy > 1e-12:
            sv = s[:, None]
            yv = y[:, None]
            Hy = H @ yv
            H = H + sv @ sv.T / sy - Hy @ Hy.T / yHy

        x, g, f_val = x_new, g_new, f_new
        hist_f.append(f_val)

    return x, f_val, k + 1, hist_f, nfev


def sr1_solver(f, grad, x0, tol=1e-6):
    """SR1 (Symmetric Rank-1)."""
    n = len(x0)
    x = x0.copy().astype(float)
    H = np.eye(n)
    g = grad(x)
    f_val = f(x)
    hist_f = [f_val]
    nfev = 1
    skip_tol = 1e-8

    for k in range(MAX_ITER):
        if np.linalg.norm(g) <= tol:
            break

        d = -H @ g
        if np.dot(d, g) >= 0:
            d = -g
            H = np.eye(n)

        alpha, x_new, f_new, g_new = backtracking_line_search(f, grad, x, d, f_val, g)
        nfev += 1

        s = x_new - x
        y = g_new - g
        v = s - H @ y
        denom = np.dot(v, y)

        if abs(denom) >= skip_tol * np.linalg.norm(v) * np.linalg.norm(y):
            vv = v[:, None]
            H = H + vv @ vv.T / denom

        x, g, f_val = x_new, g_new, f_new
        hist_f.append(f_val)

    return x, f_val, k + 1, hist_f, nfev


def broyden_solver(f, grad, x0, tol=1e-6):
    """Broyden 'good' (rang-1, non symétrique)."""
    n = len(x0)
    x = x0.copy().astype(float)
    H = np.eye(n)
    g = grad(x)
    f_val = f(x)
    hist_f = [f_val]
    nfev = 1

    for k in range(MAX_ITER):
        if np.linalg.norm(g) <= tol:
            break

        d = -H @ g
        if np.dot(d, g) >= 0:
            d = -g
            H = np.eye(n)

        alpha, x_new, f_new, g_new = backtracking_line_search(f, grad, x, d, f_val, g)
        nfev += 1

        s = x_new - x
        y = g_new - g
        Hy = H @ y
        sHy = np.dot(s, Hy)

        if abs(sHy) > 1e-12:
            sv = s[:, None]
            H = H + (sv - H @ y[:, None]) @ (sv.T @ H) / sHy

        x, g, f_val = x_new, g_new, f_new
        hist_f.append(f_val)

    return x, f_val, k + 1, hist_f, nfev


def hybrid_bfgs_sr1_solver(f, grad, x0, tol=1e-6):
    """
    Hybride BFGS / SR1 :
    - utilise SR1 dans la phase initiale (exploration)
    - bascule sur BFGS quand la convergence est proche
    """
    n = len(x0)
    x = x0.copy().astype(float)
    H = np.eye(n)
    g = grad(x)
    f_val = f(x)
    hist_f = [f_val]
    nfev = 1
    skip_tol = 1e-8

    for k in range(MAX_ITER):
        norm_g = np.linalg.norm(g)
        if norm_g <= tol:
            break

        d = -H @ g
        if np.dot(d, g) >= 0:
            d = -g
            H = np.eye(n)

        alpha, x_new, f_new, g_new = wolfe_line_search(f, grad, x, d, f_val, g)
        nfev += 2

        s = x_new - x
        y = g_new - g
        sy = np.dot(s, y)

        use_bfgs = (norm_g < 1e-2)      # bascule vers BFGS près de la solution

        if use_bfgs:
            if sy > 1e-12:
                rho_k = 1.0 / sy
                I = np.eye(n)
                sv, yv = s[:, None], y[:, None]
                A = I - rho_k * (sv @ yv.T)
                H = A @ H @ A.T + rho_k * (sv @ sv.T)
        else:   # SR1
            v = s - H @ y
            denom = np.dot(v, y)
            if abs(denom) >= skip_tol * np.linalg.norm(v) * np.linalg.norm(y):
                vv = v[:, None]
                H = H + vv @ vv.T / denom

        x, g, f_val = x_new, g_new, f_new
        hist_f.append(f_val)

    return x, f_val, k + 1, hist_f, nfev


def lbfgs_solver(f, grad, x0, m=10, tol=1e-6):
    """L-BFGS avec mémoire m (two-loop recursion)."""
    n = len(x0)
    x = x0.copy().astype(float)
    g = grad(x)
    f_val = f(x)
    hist_f = [f_val]
    nfev = 1
    s_list, y_list, rho_list = [], [], []

    def two_loop(q, s_lst, y_lst, rho_lst):
        a = []
        for si, yi, ri in zip(reversed(s_lst), reversed(y_lst), reversed(rho_lst)):
            ai = ri * np.dot(si, q)
            q = q - ai * yi
            a.append(ai)
        if s_lst:
            sy = np.dot(s_lst[-1], y_lst[-1])
            yy = np.dot(y_lst[-1], y_lst[-1])
            gamma = sy / yy if yy > 1e-12 else 1.0
        else:
            gamma = 1.0
        r = gamma * q
        for si, yi, ri, ai in zip(s_lst, y_lst, rho_lst, reversed(a)):
            beta = ri * np.dot(yi, r)
            r = r + si * (ai - beta)
        return r

    for k in range(MAX_ITER):
        if np.linalg.norm(g) <= tol:
            break

        d = -two_loop(g.copy(), s_list, y_list, rho_list)
        if np.dot(d, g) >= 0:
            d = -g

        alpha, x_new, f_new, g_new = wolfe_line_search(f, grad, x, d, f_val, g)
        nfev += 2

        s = x_new - x
        y = g_new - g
        sy = np.dot(s, y)

        if sy > 1e-12:
            if len(s_list) == m:
                s_list.pop(0); y_list.pop(0); rho_list.pop(0)
            s_list.append(s)
            y_list.append(y)
            rho_list.append(1.0 / sy)

        x, g, f_val = x_new, g_new, f_new
        hist_f.append(f_val)

    return x, f_val, k + 1, hist_f, nfev


# Wrappers L-BFGS pour différentes tailles mémoire
def lbfgs_m5(f, grad, x0, tol=1e-6):
    return lbfgs_solver(f, grad, x0, m=5,  tol=tol)

def lbfgs_m10(f, grad, x0, tol=1e-6):
    return lbfgs_solver(f, grad, x0, m=10, tol=tol)

def lbfgs_m20(f, grad, x0, tol=1e-6):
    return lbfgs_solver(f, grad, x0, m=20, tol=tol)


# ══════════════════════════════════════════════════════════════════════
# 3. DICTIONNAIRE DES SOLVEURS
# ══════════════════════════════════════════════════════════════════════

SOLVERS = {
    'BFGS'      : bfgs_solver,
    'DFP'       : dfp_solver,
    'SR1'       : sr1_solver,
    'Broyden'   : broyden_solver,
    'Hybrid'    : hybrid_bfgs_sr1_solver,
    'L-BFGS-5'  : lbfgs_m5,
    'L-BFGS-10' : lbfgs_m10,
    'L-BFGS-20' : lbfgs_m20,
}


# ══════════════════════════════════════════════════════════════════════
# 4. PROBLÈMES DE TEST
# ══════════════════════════════════════════════════════════════════════

def make_problems(dims=(2, 5, 10, 20), n_starts=5, seed=42):
    """Génère une suite de problèmes Rosenbrock en dimensions variées."""
    rng = np.random.default_rng(seed)
    problems = []
    for d in dims:
        for _ in range(n_starts):
            x0 = rng.uniform(-2.0, 2.0, d)
            problems.append((rosen, rosen_der, x0))
    return problems


# ══════════════════════════════════════════════════════════════════════
# 5. BENCHMARK
# ══════════════════════════════════════════════════════════════════════

def run_benchmark(problems, metric='n_iter', tol=1e-6):
    """
    Exécute tous les solveurs sur tous les problèmes.
    
    Paramètres
    ----------
    problems : list of (f, grad, x0)
    metric   : 'n_iter' | 'n_fevals'
    
    Retourne
    --------
    T[i, j]  : performance du solveur j sur problème i (NaN = échec)
    names    : liste des noms de solveurs
    """
    n_prob = len(problems)
    solver_names = list(SOLVERS.keys())
    T = np.full((n_prob, len(solver_names)), np.nan)

    for i, (f, grad, x0) in enumerate(problems):
        for j, (name, solver) in enumerate(SOLVERS.items()):
            try:
                x_opt, f_opt, n_iter, hist_f, n_fevals = solver(f, grad, x0, tol=tol)
                # Vérifier convergence : ||grad|| <= 100 * tol
                if np.linalg.norm(grad(x_opt)) <= tol * 1e2:
                    T[i, j] = n_iter if metric == 'n_iter' else n_fevals
            except Exception:
                pass    # NaN = échec

    return T, solver_names


# ══════════════════════════════════════════════════════════════════════
# 6. PROFILS DE PERFORMANCE (Dolan-Moré, 2002)
# ══════════════════════════════════════════════════════════════════════

def performance_profile(T, tau_max=100, n_tau=500):
    """
    Calcule les profils de performance.

    r[i,j] = T[i,j] / min_s T[i,s]          (ratio performance)
    rho_s(tau) = |{i : r[i,s] <= tau}| / n_p  (fraction de problèmes résolus)

    Retourne
    --------
    tau_grid : grille log de tau dans [1, tau_max]
    rho      : tableau (n_solveurs, n_tau)
    """
    n_p, n_s = T.shape
    tau_grid = np.logspace(0, np.log10(tau_max), n_tau)

    best = np.nanmin(T, axis=1, keepdims=True)   # meilleur solveur par problème
    R = T / best                                   # ratio r[i,j]

    rho = np.zeros((n_s, n_tau))
    for k, tau in enumerate(tau_grid):
        rho[:, k] = np.nansum(R <= tau, axis=0) / n_p

    return tau_grid, rho


# ══════════════════════════════════════════════════════════════════════
# 7. AFFICHAGE
# ══════════════════════════════════════════════════════════════════════

def plot_performance_profiles(tau_grid, rho, solver_names,
                               title='Profils de performance (Dolan-Moré 2002)',
                               metric_label='Nombre d\'itérations',
                               filename=None):
    colors = plt.cm.tab10(np.linspace(0, 1, len(solver_names)))
    styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

    fig, ax = plt.subplots(figsize=(10, 6))
    for j, name in enumerate(solver_names):
        ax.semilogx(tau_grid, rho[j], label=name,
                    color=colors[j], ls=styles[j % 8], lw=2)

    ax.set_xlabel(r'$\tau$ — facteur par rapport au meilleur solveur')
    ax.set_ylabel(r'$\rho_s(\tau)$ — fraction de problèmes résolus')
    ax.set_title(f'{title}\nMétrique : {metric_label}')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([1, tau_grid[-1]])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée : {filename}")
    plt.show()


def print_results_table(T, solver_names, title="Résultats Benchmark"):
    col_w = 13
    print(f"\n{'='*80}")
    print(title)
    print('='*80)
    header = "Problème".ljust(10)
    for name in solver_names:
        header += name.ljust(col_w)
    print(header)
    print('-'*80)
    for i, row in enumerate(T):
        line = f"P{i+1}".ljust(10)
        for val in row:
            line += ("FAIL" if np.isnan(val) else str(int(val))).ljust(col_w)
        print(line)
    print('='*80)


# ══════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Génération des problèmes...")
    problems = make_problems(dims=[2, 5, 10, 20], n_starts=5, seed=42)
    print(f"  → {len(problems)} problèmes générés\n")

    print("Benchmark (itérations)...")
    T_iter, names = run_benchmark(problems, metric='n_iter')

    print("Benchmark (évaluations de f)...")
    T_fevals, _   = run_benchmark(problems, metric='n_fevals')

    # Tableaux de résultats
    print_results_table(T_iter,   names, "Tableau — Nombre d'itérations")
    print_results_table(T_fevals, names, "Tableau — Nombre d'évaluations de f")

    # Profils de performance
    tau, rho_iter   = performance_profile(T_iter,   tau_max=100)
    tau, rho_fevals = performance_profile(T_fevals, tau_max=100)

    plot_performance_profiles(tau, rho_iter,   names,
                               metric_label="Itérations",
                               filename="perf_profile_iter.pdf")

    plot_performance_profiles(tau, rho_fevals, names,
                               metric_label="Évaluations de f",
                               filename="perf_profile_fevals.pdf")
    
    
    
    
    
    