
"""
SR1 — Symmetric Rank-One Update
Mise à jour de rang 1 de l'approximation du Hessien (ou de son inverse).

Formule de mise à jour de B (approximation du Hessien) :
    B_{k+1} = B_k + (y_k - B_k s_k)(y_k - B_k s_k)^T / ((y_k - B_k s_k)^T s_k)

Formule de mise à jour de H = B^{-1} (via Sherman-Morrison) :
    H_{k+1} = H_k + (s_k - H_k y_k)(s_k - H_k y_k)^T / ((s_k - H_k y_k)^T y_k)

Avantages : capture les courbures négatives, pas besoin de condition de courbure positive.
Inconvénient : H peut perdre son caractère défini positif.
"""


import numpy as np


def line_search_backtrack(f, grad_f, x, d, f0, g0, alpha0=1.0, rho=0.5, c=1e-4, max_iter=50):
    """Backtracking line search (Armijo)."""
    alpha = alpha0
    for _ in range(max_iter):
        if f(x + alpha * d) <= f0 + c * alpha * (g0 @ d):
            return alpha
        alpha *= rho
    return max(alpha, 1e-12)


def sr1(f, grad_f, x0, tol=1e-6, max_iter=1000, r=1e-8, update_H=True):
    """
    SR1 — Symmetric Rank-One.

    Paramètres SR1
    --------------
    r         : seuil de sécurité pour éviter les mises à jour instables.
                On saute la mise à jour si :
                |v^T s| < r * ||v|| * ||s||
                où v = s_k - H_k y_k (ou y_k - B_k s_k)
    update_H  : si True, met à jour H (inverse du Hessien).
                si False, met à jour B (Hessien) et résout B*d = -g.

    Parameters
    ----------
    f, grad_f : fonction et gradient
    x0        : point initial
    tol       : tolérance convergence
    max_iter  : max itérations
    r         : paramètre de stabilité SR1 (0 < r < 1, typiquement 1e-8)
    update_H  : True → met à jour H, False → met à jour B

    Returns
    -------
    x       : solution
    history : historique de convergence
    """
    n = len(x0)
    x = x0.copy().astype(float)

    if update_H:
        H = np.eye(n)   # Approximation de B^{-1}
    else:
        B = np.eye(n)   # Approximation du Hessien B

    history = {
        "x": [x.copy()], "f": [f(x)],
        "grad_norm": [], "iter": 0,
        "skipped_updates": 0
    }

    for k in range(max_iter):
        g = grad_f(x)
        grad_norm = np.linalg.norm(g)
        history["grad_norm"].append(grad_norm)

        if grad_norm < tol:
            print(f"[SR1] Convergé en {k} itérations, ||grad|| = {grad_norm:.2e}")
            break

        # ── Direction de descente ─────────────────────────────────────────
        if update_H:
            d = -H @ g
        else:
            try:
                d = np.linalg.solve(B, -g)
            except np.linalg.LinAlgError:
                d = -g  # Fallback

        # Vérification direction de descente
        if d @ g >= 0:
            d = -g

        # ── Recherche linéaire ────────────────────────────────────────────
        alpha = line_search_backtrack(f, grad_f, x, d, f(x), g)

        x_new = x + alpha * d
        g_new = grad_f(x_new)

        s = x_new - x
        y = g_new - g

        # ── Mise à jour SR1 ───────────────────────────────────────────────
        if update_H:
            # Mise à jour de H (inverse du Hessien)
            v = s - H @ y
            denom = v @ y
            # Test de stabilité : skip si |v^T y| < r * ||v|| * ||y||
            if abs(denom) >= r * np.linalg.norm(v) * np.linalg.norm(y):
                H = H + np.outer(v, v) / denom
            else:
                history["skipped_updates"] += 1
        else:
            # Mise à jour de B (Hessien direct)
            v = y - B @ s
            denom = v @ s
            if abs(denom) >= r * np.linalg.norm(v) * np.linalg.norm(s):
                B = B + np.outer(v, v) / denom
            else:
                history["skipped_updates"] += 1

        x = x_new
        history["x"].append(x.copy())
        history["f"].append(f(x))

    history["iter"] = k + 1
    if history["skipped_updates"] > 0:
        print(f"[SR1] {history['skipped_updates']} mises à jour ignorées (instabilité).")
    return x, history
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

    print("=== SR1 (mise à jour de H) ===")
    x_opt, hist = sr1(rosenbrock, rosenbrock_grad, x0.copy(), update_H=True)
    print(f"f(x*) = {rosenbrock(x_opt):.6e}, iters = {hist['iter']}")

    print("\n=== SR1 (mise à jour de B) ===")
    x_opt, hist = sr1(rosenbrock, rosenbrock_grad, x0.copy(), update_H=False)
    print(f"f(x*) = {rosenbrock(x_opt):.6e}, iters = {hist['iter']}")