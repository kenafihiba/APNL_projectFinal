"""
Hybrid BFGS / SR1 — Alternance selon la courbure locale
Utilise BFGS quand la courbure est favorable (y^T s > 0),
bascule sur SR1 quand la condition de courbure positive n'est pas satisfaite
ou quand BFGS risque de dégrader H.

Stratégie d'alternance :
    1. Condition primaire   : si y^T s > epsilon → BFGS
    2. Condition secondaire : sinon, si la mise à jour SR1 est stable → SR1
    3. Fallback             : si ni l'une ni l'autre, H reste inchangée

Logique de décision à chaque itération :
    - Calculer sy = y^T s et v = s - H y (résidu SR1)
    - Si sy > thr_curvature                    → BFGS  (bonne courbure)
    - Elif |v^T y| >= r * ||v|| * ||y||        → SR1   (stable)
    - Else                                     → skip  (H inchangée)
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


def _bfgs_update(H, s, y, sy):
    rho = 1.0 / sy
    n = len(s)
    I = np.eye(n)
    A = I - rho * np.outer(s, y)
    B = I - rho * np.outer(y, s)
    return A @ H @ B + rho * np.outer(s, s)

def _sr1_update(H, s, y):
    v = s - H @ y
    denom = v @ y
    return H + np.outer(v, v) / denom, denom, v


def hybrid_bfgs_sr1(f, grad_f, x0, tol=1e-6, max_iter=1000,
                    thr_curvature=1e-10, sr1_safety=1e-8):
    """
    Méthode Hybrid BFGS/SR1.

    Règle d'alternance à chaque itération :
        1. Calculer sy = y^T s
        2. Si sy > thr_curvature
               → BFGS  (condition de courbure positive vérifiée)
        3. Sinon calculer v = s - H y et denom = v^T y
               → SR1 si |denom| >= sr1_safety * ||v|| * ||y||
               → skip sinon

    Parameters
    ----------
    f, grad_f      : fonction et gradient
    x0             : point initial
    tol            : tolérance convergence
    max_iter       : max itérations
    thr_curvature  : seuil y^T s pour choisir BFGS (défaut 1e-10)
    sr1_safety     : paramètre r de stabilité SR1 (défaut 1e-8)

    Returns
    -------
    x       : solution approchée
    history : dict historique (avec compteurs bfgs_steps, sr1_steps, skipped)
    """
    n = len(x0)
    x = x0.copy().astype(float)
    H = np.eye(n)

    history = {
        "x": [x.copy()], "f": [f(x)],
        "grad_norm": [], "iter": 0,
        "method_used": [],   # trace de quelle méthode utilisée à chaque iter
        "bfgs_steps": 0,
        "sr1_steps": 0,
        "skipped": 0
    }

    for k in range(max_iter):
        g = grad_f(x)
        grad_norm = np.linalg.norm(g)
        history["grad_norm"].append(grad_norm)

        if grad_norm < tol:
            print(f"[Hybrid] Convergé en {k} itérations, ||grad|| = {grad_norm:.2e}")
            break

        # ── Direction de descente ─────────────────────────────────────────
        d = -H @ g
        # Vérification que d descend vraiment
        if d @ g > 0:
            # H a perdu son caractère défini positif → reset
            H = np.eye(n)
            d = -g
            history["method_used"].append("reset")
        
        # ── Recherche linéaire ────────────────────────────────────────────
        alpha = line_search_wolfe(f, grad_f, x, d, f(x), g)

        x_new = x + alpha * d
        g_new = grad_f(x_new)

        s  = x_new - x
        y  = g_new - g
        sy = y @ s

        # ── Décision BFGS vs SR1 ──────────────────────────────────────────
        if sy > thr_curvature:
            # Condition de courbure positive → BFGS
            H = _bfgs_update(H, s, y, sy)
            history["method_used"].append("BFGS")
            history["bfgs_steps"] += 1
        else:
            # Pas de condition de courbure → tenter SR1
            v = s - H @ y
            denom = v @ y
            norm_v = np.linalg.norm(v)
            norm_y = np.linalg.norm(y)

            if abs(denom) >= sr1_safety * norm_v * norm_y and abs(denom) > 1e-14:
                H = H + np.outer(v, v) / denom
                history["method_used"].append("SR1")
                history["sr1_steps"] += 1
            else:
                # Ni BFGS ni SR1 stables : skip
                history["method_used"].append("skip")
                history["skipped"] += 1

        x = x_new
        history["x"].append(x.copy())
        history["f"].append(f(x))

    history["iter"] = k + 1
    total = history["bfgs_steps"] + history["sr1_steps"]
    if total > 0:
        bfgs_pct = 100 * history["bfgs_steps"] / total
        sr1_pct  = 100 * history["sr1_steps"] / total
        print(f"[Hybrid] BFGS: {history['bfgs_steps']} ({bfgs_pct:.1f}%), "
              f"SR1: {history['sr1_steps']} ({sr1_pct:.1f}%), "
              f"Skip: {history['skipped']}")
    return x, history


# ─── Variante avec fenêtre glissante (plus robuste) ──────────────────────────

def hybrid_bfgs_sr1_adaptive(f, grad_f, x0, tol=1e-6, max_iter=1000,
                              window=5, bfgs_threshold=0.6):
    """
    Variante adaptative : utilise une fenêtre glissante pour décider
    quel schéma dominant appliquer dans les prochaines itérations.

    Si dans les `window` dernières itérations la fraction BFGS > bfgs_threshold,
    on reste en BFGS. Sinon on préfère SR1 pour explorer des directions non-PD.

    Parameters
    ----------
    window          : taille de la fenêtre d'historique (défaut 5)
    bfgs_threshold  : fraction minimale BFGS pour rester en mode BFGS (défaut 0.6)
    """
    n = len(x0)
    x = x0.copy().astype(float)
    H = np.eye(n)

    recent_methods = []  # Fenêtre glissante des méthodes utilisées
    history = {
        "x": [x.copy()], "f": [f(x)], "grad_norm": [],
        "iter": 0, "method_used": [], "bfgs_steps": 0, "sr1_steps": 0
    }

    for k in range(max_iter):
        g = grad_f(x)
        grad_norm = np.linalg.norm(g)
        history["grad_norm"].append(grad_norm)

        if grad_norm < tol:
            print(f"[Hybrid-Adaptive] Convergé en {k} itérations")
            break

        d = -H @ g
        if d @ g > 0:
            H = np.eye(n)
            d = -g

        alpha = line_search_wolfe(f, grad_f, x, d, f(x), g)
        x_new = x + alpha * d
        g_new = grad_f(x_new)

        s  = x_new - x
        y  = g_new - g
        sy = y @ s

        # Décision basée sur la fenêtre
        if len(recent_methods) >= window:
            bfgs_frac = recent_methods.count("BFGS") / window
            prefer_bfgs = bfgs_frac >= bfgs_threshold
        else:
            prefer_bfgs = True  # Par défaut BFGS au départ

        method_used = "skip"
        if prefer_bfgs and sy > 1e-10:
            H = _bfgs_update(H, s, y, sy)
            method_used = "BFGS"
            history["bfgs_steps"] += 1
        else:
            v = s - H @ y
            denom = v @ y
            if abs(denom) >= 1e-8 * np.linalg.norm(v) * np.linalg.norm(y) and abs(denom) > 1e-14:
                H = H + np.outer(v, v) / denom
                method_used = "SR1"
                history["sr1_steps"] += 1
            elif sy > 1e-10:
                # Fallback BFGS
                H = _bfgs_update(H, s, y, sy)
                method_used = "BFGS"
                history["bfgs_steps"] += 1

        # Mise à jour fenêtre glissante
        recent_methods.append(method_used)
        if len(recent_methods) > window:
            recent_methods.pop(0)

        history["method_used"].append(method_used)
        x = x_new
        history["x"].append(x.copy())
        history["f"].append(f(x))

    history["iter"] = k + 1
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

    print("=== Hybrid BFGS/SR1 (standard) ===")
    x_opt, hist = hybrid_bfgs_sr1(rosenbrock, rosenbrock_grad, x0.copy())
    print(f"f(x*) = {rosenbrock(x_opt):.6e}, iters = {hist['iter']}")

    print("\n=== Hybrid BFGS/SR1 (adaptatif fenêtre=5) ===")
    x_opt, hist = hybrid_bfgs_sr1_adaptive(rosenbrock, rosenbrock_grad, x0.copy())
    print(f"f(x*) = {rosenbrock(x_opt):.6e}, iters = {hist['iter']}")
    print(f"BFGS: {hist['bfgs_steps']}, SR1: {hist['sr1_steps']}")