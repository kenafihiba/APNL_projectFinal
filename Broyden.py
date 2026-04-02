"""
Famille de Broyden — φ-paramétrisée (Broyden class)
Interpolation convexe entre DFP (φ=0) et BFGS (φ=1).

Formule unifiée de mise à jour de H (inverse du Hessien) :
    H_{k+1} = H_k^{DFP}  +  φ_k * (y_k^T H_k y_k) * w_k w_k^T

où :
    H_k^{DFP} = H_k - (H_k y_k y_k^T H_k)/(y_k^T H_k y_k) + (s_k s_k^T)/(y_k^T s_k)
    w_k       = s_k/(y_k^T s_k) - H_k y_k/(y_k^T H_k y_k)
    φ ∈ [0, 1] : φ=0 → DFP, φ=1 → BFGS

Formulation équivalente (plus stable numériquement) :
    H_{k+1} = (1-φ) * H_k^{DFP} + φ * H_k^{BFGS}

Cas particuliers notables :
    φ = 0   → DFP
    φ = 0.5 → Méthode de Pearson (symétrique)
    φ = 1   → BFGS
    φ > 1   → hors de la famille convexe (possible mais moins stable)
"""

import numpy as np


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


def _dfp_update(H, s, y):
    """Mise à jour DFP de H."""
    Hy  = H @ y
    yHy = y @ Hy
    sy  = y @ s
    if yHy < 1e-12 or sy < 1e-12:
        return H
    return H - np.outer(Hy, Hy) / yHy + np.outer(s, s) / sy


def _bfgs_update(H, s, y):
    """Mise à jour BFGS de H."""
    sy = y @ s
    if sy < 1e-12:
        return H
    rho = 1.0 / sy
    n = len(s)
    I = np.eye(n)
    A = I - rho * np.outer(s, y)
    B = I - rho * np.outer(y, s)
    return A @ H @ B + rho * np.outer(s, s)


def broyden_update(H, s, y, phi):
    """
    Mise à jour de la famille de Broyden avec paramètre φ.

    Formule : H_{k+1} = (1-φ) * H_DFP + φ * H_BFGS

    Parameters
    ----------
    H   : approximation courante de l'inverse du Hessien
    s   : s_k = x_{k+1} - x_k
    y   : y_k = grad_{k+1} - grad_k
    phi : paramètre de Broyden ∈ [0, 1]
          0 → DFP, 1 → BFGS, 0.5 → Pearson

    Returns
    -------
    H_new : approximation mise à jour
    """
    sy = y @ s
    if sy < 1e-10:
        return H

    if abs(phi) < 1e-12:
        return _dfp_update(H, s, y)
    elif abs(phi - 1.0) < 1e-12:
        return _bfgs_update(H, s, y)
    else:
        # Interpolation convexe (numérique plus stable que la formule w_k)
        H_dfp  = _dfp_update(H, s, y)
        H_bfgs = _bfgs_update(H, s, y)
        return (1 - phi) * H_dfp + phi * H_bfgs


def broyden(f, grad_f, x0, phi=1.0, tol=1e-6, max_iter=1000,
            adaptive_phi=False):
    """
    Famille de Broyden φ-paramétrisée.

    Parameters
    ----------
    f, grad_f    : fonction et gradient
    x0           : point initial
    phi          : paramètre ∈ [0, 1] (0=DFP, 1=BFGS, 0.5=Pearson)
    tol          : tolérance convergence
    max_iter     : max itérations
    adaptive_phi : si True, adapte φ à chaque itération selon la courbure.
                   φ → 1 (BFGS) quand y^T s est grand (bonne courbure)
                   φ → 0 (DFP)  quand y^T s est petit  (mauvaise courbure)

    Returns
    -------
    x       : solution approchée
    history : dict historique
    """
    n = len(x0)
    x = x0.copy().astype(float)
    H = np.eye(n)

    history = {
        "x": [x.copy()], "f": [f(x)],
        "grad_norm": [], "iter": 0,
        "phi_values": [phi]
    }

    current_phi = phi

    for k in range(max_iter):
        g = grad_f(x)
        grad_norm = np.linalg.norm(g)
        history["grad_norm"].append(grad_norm)

        if grad_norm < tol:
            print(f"[Broyden φ={phi:.2f}] Convergé en {k} itérations, ||grad|| = {grad_norm:.2e}")
            break

        d = -H @ g
        if d @ g >= 0:
            d = -g

        alpha = line_search_wolfe(f, grad_f, x, d, f(x), g)

        x_new = x + alpha * d
        g_new = grad_f(x_new)

        s = x_new - x
        y = g_new - g
        sy = y @ s

        # ── φ adaptatif ───────────────────────────────────────────────────
        if adaptive_phi and sy > 1e-10:
            # Mesure de qualité : cos(angle entre s et y)
            cos_angle = sy / (np.linalg.norm(s) * np.linalg.norm(y) + 1e-12)
            # Bonne courbure → favoriser BFGS (φ→1)
            # Mauvaise courbure → favoriser DFP (φ→0)
            current_phi = float(np.clip(cos_angle, 0.0, 1.0))
            history["phi_values"].append(current_phi)

        # ── Mise à jour de H ──────────────────────────────────────────────
        H = broyden_update(H, s, y, current_phi)

        x = x_new
        history["x"].append(x.copy())
        history["f"].append(f(x))

    history["iter"] = k + 1
    return x, history


# Raccourcis pour les cas particuliers
def broyden_dfp(f, grad_f, x0, **kw):
    """Broyden avec φ=0 → DFP."""
    return broyden(f, grad_f, x0, phi=0.0, **kw)

def broyden_bfgs(f, grad_f, x0, **kw):
    """Broyden avec φ=1 → BFGS."""
    return broyden(f, grad_f, x0, phi=1.0, **kw)

def broyden_pearson(f, grad_f, x0, **kw):
    """Broyden avec φ=0.5 → Méthode de Pearson."""
    return broyden(f, grad_f, x0, phi=0.5, **kw)


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

    for phi_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x_opt, hist = broyden(rosenbrock, rosenbrock_grad, x0.copy(), phi=phi_val)
        label = {0.0: "(DFP)", 0.5: "(Pearson)", 1.0: "(BFGS)"}.get(phi_val, "")
        print(f"Broyden φ={phi_val:.2f} {label:10s}: f(x*)={rosenbrock(x_opt):.2e}, iters={hist['iter']}")

    print("\n=== Broyden φ adaptatif ===")
    x_opt, hist = broyden(rosenbrock, rosenbrock_grad, x0.copy(), phi=1.0, adaptive_phi=True)
    print(f"f(x*) = {rosenbrock(x_opt):.2e}, iters = {hist['iter']}")