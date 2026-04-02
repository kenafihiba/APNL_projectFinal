"""
L-BFGS — Limited-memory BFGS
Stocke seulement les m dernières paires (s_k, y_k) au lieu de H complète.
Supporte m = 5, 10, 20 (ou toute autre valeur).

"""

import numpy as np
from collections import deque


def line_search_wolfe(f, grad_f, x, d, f0, g0, c1=1e-4, c2=0.9, max_iter=50):
    alpha, alpha_min, alpha_max = 1.0, 0.0, np.inf
    for i in range(max_iter):
        x_new = x + alpha * d
        f_new = f(x_new)
        if f_new > f0 + c1 * alpha * (g0 @ d) or (i > 0 and f_new >= f(x + alpha_min * d)):
            alpha = _zoom(f, grad_f, x, d, alpha_min, alpha, f0, g0, c1, c2)
            break
        g_new = grad_f(x_new)
        if abs(g_new @ d) <= -c2 * (g0 @ d):
            break
        if g_new @ d >= 0:
            alpha = _zoom(f, grad_f, x, d, alpha, alpha_min, f0, g0, c1, c2)
            break
        alpha_min = alpha
        alpha = min(2 * alpha, alpha_max) if alpha_max == np.inf else (alpha + alpha_max) / 2
    return max(alpha, 1e-10)


def _zoom(f, grad_f, x, d, alpha_lo, alpha_hi, f0, g0, c1, c2, max_iter=20):
    for _ in range(max_iter):
        alpha = (alpha_lo + alpha_hi) / 2
        f_alpha = f(x + alpha * d)
        f_lo    = f(x + alpha_lo * d)
        if f_alpha > f0 + c1 * alpha * (g0 @ d) or f_alpha >= f_lo:
            alpha_hi = alpha
        else:
            g_alpha = grad_f(x + alpha * d)
            if abs(g_alpha @ d) <= -c2 * (g0 @ d):
                return alpha
            if g_alpha @ d * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha
    return alpha


def lbfgs_two_loop(q, s_list, y_list, rho_list):
    """
    L-BFGS two-loop recursion.
    Calcule H_k * q sans former H_k explicitement.

    Entrées
    -------
    q        : vecteur (typiquement le gradient courant)
    s_list   : deque des m derniers s_k = x_{k+1} - x_k
    y_list   : deque des m derniers y_k = grad_{k+1} - grad_k
    rho_list : deque des m derniers rho_k = 1 / (y_k^T s_k)

    Retourne
    --------
    r : approximation de H_k * q (direction de descente si q = -grad)
    """
    m = len(s_list)
    alpha_list = []

    # Boucle arrière
    for i in range(m - 1, -1, -1):
        alpha_i = rho_list[i] * (s_list[i] @ q)
        alpha_list.append(alpha_i)
        q = q - alpha_i * y_list[i]

    alpha_list.reverse()

    # Initialisation de H0 (scaling de Barzilai-Borwein)
    if m > 0:
        s_last = s_list[-1]
        y_last = y_list[-1]
        gamma = (s_last @ y_last) / (y_last @ y_last)
        r = gamma * q
    else:
        r = q.copy()

    # Boucle avant
    for i in range(m):
        beta_i = rho_list[i] * (y_list[i] @ r)
        r = r + (alpha_list[i] - beta_i) * s_list[i]

    return r


def lbfgs(f, grad_f, x0, m=10, tol=1e-6, max_iter=1000):
    """
    L-BFGS avec mémoire limitée m.

    Paramètre clé : m (taille de la fenêtre mémoire)
      - m = 5  : faible mémoire, rapide mais moins précis
      - m = 10 : bon compromis (valeur par défaut)
      - m = 20 : meilleure approximation, plus de mémoire

    Parameters
    ----------
    f       : fonction objectif
    grad_f  : gradient de f
    x0      : point initial
    m       : nombre de paires (s, y) stockées
    tol     : tolérance sur la norme du gradient
    max_iter: nombre max d'itérations

    Returns
    -------
    x       : solution approchée
    history : dict avec les traces de convergence
    """
    x = x0.copy().astype(float)

    # Stockage circulaire des m dernières paires
    s_list   = deque(maxlen=m)
    y_list   = deque(maxlen=m)
    rho_list = deque(maxlen=m)

    history = {"x": [x.copy()], "f": [f(x)], "grad_norm": [], "iter": 0, "m": m}

    for k in range(max_iter):
        g = grad_f(x)
        grad_norm = np.linalg.norm(g)
        history["grad_norm"].append(grad_norm)

        if grad_norm < tol:
            print(f"[L-BFGS m={m}] Convergé en {k} itérations, ||grad|| = {grad_norm:.2e}")
            break

        # Direction de descente via two-loop recursion
        d = -lbfgs_two_loop(g.copy(), s_list, y_list, rho_list)

        # Vérification que d est bien une direction de descente
        if d @ g > 0:
            d = -g  # Fallback gradient

        # Recherche linéaire
        alpha = line_search_wolfe(f, grad_f, x, d, f(x), g)

        x_new = x + alpha * d
        g_new = grad_f(x_new)

        s = x_new - x
        y = g_new - g
        sy = y @ s

        if sy > 1e-10:
            s_list.append(s)
            y_list.append(y)
            rho_list.append(1.0 / sy)

        x = x_new
        history["x"].append(x.copy())
        history["f"].append(f(x))

    history["iter"] = k + 1
    return x, history


def lbfgs_m5(f, grad_f, x0, **kwargs):
    """L-BFGS avec m=5."""
    return lbfgs(f, grad_f, x0, m=5, **kwargs)


def lbfgs_m10(f, grad_f, x0, **kwargs):
    """L-BFGS avec m=10."""
    return lbfgs(f, grad_f, x0, m=10, **kwargs)


def lbfgs_m20(f, grad_f, x0, **kwargs):
    """L-BFGS avec m=20."""
    return lbfgs(f, grad_f, x0, m=20, **kwargs)


# ─── Exemple d'utilisation ────────────────────────────────────────────────────
if __name__ == "__main__":
    def rosenbrock(x):
        return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))
 
    def rosenbrock_grad(x):
        g = np.zeros_like(x)
        for i in range(len(x)-1):
            g[i]   += -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
            g[i+1] +=  200 * (x[i+1] - x[i]**2)
        return g
 
    x0 = np.array([-1.2, 1.0, -0.5, 0.8])
 
    for m_val in [5, 10, 20]:
        x_opt, hist = lbfgs(rosenbrock, rosenbrock_grad, x0.copy(), m=m_val)
        print(f"L-BFGS m={m_val}: f(x*)={rosenbrock(x_opt):.2e}, iters={hist['iter']}")
 