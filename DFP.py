"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          DFP — Davidon-Fletcher-Powell (1959-1963)                         ║
║          Implémentation commentée — Projet 2                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Contexte historique :
─────────────────────
  DFP est la PREMIÈRE méthode quasi-Newton de l'histoire.
  Proposée par Davidon (1959), puis publiée par Fletcher et Powell (1963).
  Elle a été supplantée par BFGS (1970) qui lui est supérieur en pratique,
  mais reste une référence théorique fondamentale.

Différence clé avec BFGS :
───────────────────────────
  BFGS met à jour l'approximation de l'INVERSE Hessienne Hₖ ≈ [∇²f(xₖ)]⁻¹
  DFP  met également à jour Hₖ ≈ [∇²f(xₖ)]⁻¹, MAIS avec une formule
  différente (duale de BFGS).

  En fait, DFP et BFGS sont "duaux" l'un de l'autre :
    - La formule BFGS appliquée à Bₖ ≈ ∇²f(xₖ)   donne la mise à jour BFGS
    - La formule BFGS appliquée à Hₖ = Bₖ⁻¹       donne la mise à jour DFP !

  Autrement dit : DFP(Hₖ) ↔ BFGS(Bₖ = Hₖ⁻¹)

Formule de mise à jour DFP :
─────────────────────────────
    Hₖ₊₁ = Hₖ − (Hₖ yₖ yₖᵀ Hₖ) / (yₖᵀ Hₖ yₖ)  +  (sₖ sₖᵀ) / (yₖᵀ sₖ)
              ────────────────────────────────────     ───────────────────
                    terme de "correction"                terme de "mise
                    (retire l'ancienne info)              à jour sécante"

  où :
    sₖ = xₖ₊₁ − xₖ           (déplacement)
    yₖ = ∇f(xₖ₊₁) − ∇f(xₖ)  (variation du gradient)

Pourquoi DFP est moins robuste que BFGS :
──────────────────────────────────────────
  La formule DFP est sensible à la condition  yₖᵀ Hₖ yₖ > 0.
  Si Hₖ devient mal conditionnée (valeurs propres proches de 0),
  le dénominateur peut être très petit → instabilité numérique.
  BFGS est plus robuste car son terme correctif implique  yₖᵀ sₖ
  qui est mieux contrôlé par la recherche de Wolfe.
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 1 : Recherche linéaire de Wolfe (identique à BFGS)
# ══════════════════════════════════════════════════════════════════════════════

def wolfe_line_search(f, grad, xk, dk, fk, gk, c1=1e-4, c2=0.9, max_iter=50):
    """
    Recherche linéaire satisfaisant les conditions de Wolfe fortes.

    Conditions :
      (W1) Armijo  : f(xₖ + α dₖ) ≤ f(xₖ) + c₁ α ∇f(xₖ)ᵀdₖ
      (W2) Courbure: |∇f(xₖ + α dₖ)ᵀdₖ| ≤ c₂ |∇f(xₖ)ᵀdₖ|

    La condition W2 garantit yₖᵀsₖ > 0, nécessaire pour DFP et BFGS.

    Paramètres
    ──────────
    f, grad  : fonction et gradient
    xk       : point courant
    dk       : direction de descente (dₖᵀ∇f(xₖ) < 0)
    fk, gk   : valeurs pré-calculées en xk
    c1, c2   : paramètres de Wolfe (c1 < c2 < 1)
    max_iter : max réductions de pas

    Retourne : (alpha, f_new, g_new)
    """
    gd = gk @ dk        # pente directionnelle ∇f(xₖ)ᵀdₖ  (< 0)

    alpha    = 1.0      # essai initial : pas plein
    alpha_lo = 0.0
    alpha_hi = np.inf

    for _ in range(max_iter):
        x_new = xk + alpha * dk
        f_new = f(x_new)
        g_new = grad(x_new)

        # W1 : décroissance suffisante ?
        if f_new > fk + c1 * alpha * gd:
            alpha_hi = alpha
            alpha = (alpha_lo + alpha_hi) / 2
            continue

        # W2 : courbure suffisante ?
        if abs(g_new @ dk) <= c2 * abs(gd):
            return alpha, f_new, g_new

        # Affiner l'intervalle
        if g_new @ dk < 0:
            alpha_lo = alpha
            alpha = min(2 * alpha, alpha_hi) if np.isinf(alpha_hi) else (alpha_lo + alpha_hi) / 2
        else:
            alpha_hi = alpha
            alpha = (alpha_lo + alpha_hi) / 2

    return alpha, f(xk + alpha * dk), grad(xk + alpha * dk)


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 2 : Classe DFP
# ══════════════════════════════════════════════════════════════════════════════

class DFP:
    """
    DFP — Davidon-Fletcher-Powell.

    Comme BFGS, DFP maintient une approximation Hₖ de l'inverse Hessienne.
    La différence réside uniquement dans la formule de mise à jour.

    Attributs
    ─────────
    H : np.ndarray (n×n)
        Approximation courante de [∇²f(xₖ)]⁻¹.
        Initialisée à Iₙ.

    Utilisation
    ───────────
        solver = DFP(n=10)
        result = solver.optimize(f, grad, x0)
    """

    name = "DFP"

    def __init__(self, n):
        """
        Paramètres
        ──────────
        n : int  — dimension du problème
        """
        self.n = n
        self.H = np.eye(n)   # H₀ = Iₙ  (approximation initiale neutre)

    # ──────────────────────────────────────────────────────────────────────────
    def direction(self, gk):
        """
        Direction de descente quasi-Newton.

            dₖ = −Hₖ · ∇f(xₖ)

        Identique à BFGS : si Hₖ est DDP, dₖ est une direction de descente.

        Paramètres
        ──────────
        gk : np.ndarray (n,)  — gradient ∇f(xₖ)

        Retourne
        ────────
        dk : np.ndarray (n,)  — direction de descente
        """
        return -self.H @ gk

    # ──────────────────────────────────────────────────────────────────────────
    def update(self, sk, yk):
        """
        Mise à jour DFP : Hₖ → Hₖ₊₁.

        Formule complète :
        ──────────────────
            Hₖ₊₁ = Hₖ
                  − ──────────────────────    ←  terme de RETRAIT (rang 1)
                     yₖᵀ Hₖ yₖ
                  + ────────────────           ←  terme SÉCANTE  (rang 1)
                     yₖᵀ sₖ

        Plus précisément :
            Hₖ₊₁ = Hₖ  −  (Hₖ yₖ)(Hₖ yₖ)ᵀ / (yₖᵀ Hₖ yₖ)
                        +  sₖ sₖᵀ / (yₖᵀ sₖ)

        Interprétation :
        ─────────────────
          • Le terme sécante  sₖsₖᵀ/(yₖᵀsₖ)  impose Hₖ₊₁ yₖ = sₖ
            (équation sécante : l'approx doit satisfaire la relation
             ∇²f · s ≈ y  ↔  s ≈ H · y)
          • Le terme de retrait enlève la composante "ancienne" de Hₖ
            dans la direction yₖ pour éviter la double information.

        Conditions de validité :
        ─────────────────────────
          1. yₖᵀsₖ > 0     (garantie par Wolfe W2 → dénominateur sécante)
          2. yₖᵀHₖyₖ > 0   (garantie si Hₖ DDP → dénominateur retrait)
          Si l'une échoue → on saute la mise à jour.

        Paramètres
        ──────────
        sk : np.ndarray (n,)  — déplacement sₖ = xₖ₊₁ − xₖ
        yk : np.ndarray (n,)  — variation gradient yₖ = ∇f(xₖ₊₁) − ∇f(xₖ)
        """

        # ── Dénominateur du terme sécante : yₖᵀsₖ ────────────────────────
        # Doit être > 0. Garanti par les conditions de Wolfe.
        sy = yk @ sk              # yₖᵀsₖ  (scalaire)

        if sy < 1e-10:
            # Condition de courbure violée → skip
            return

        # ── Vecteur Hₖ yₖ  (utilisé deux fois) ──────────────────────────
        Hy = self.H @ yk          # Hₖ yₖ  (vecteur, taille n)

        # ── Dénominateur du terme de retrait : yₖᵀ Hₖ yₖ ─────────────────
        # = yₖᵀ (Hₖ yₖ) = yₖ · Hy
        # Doit être > 0 si Hₖ est DDP. Peut devenir ≤ 0 si Hₖ se dégrade.
        yHy = yk @ Hy             # yₖᵀ Hₖ yₖ  (scalaire)

        if yHy < 1e-10:
            # Hₖ a perdu sa définie-positivité → skip pour éviter l'explosion
            return

        # ── Terme de RETRAIT : −(Hₖ yₖ)(Hₖ yₖ)ᵀ / (yₖᵀ Hₖ yₖ) ──────────
        # Retire la composante de Hₖ dans la direction yₖ
        # C'est un terme de rang 1 négatif semi-défini
        term_retrait = np.outer(Hy, Hy) / yHy

        # ── Terme SÉCANTE : sₖ sₖᵀ / (yₖᵀ sₖ) ───────────────────────────
        # Ajoute l'information de courbure observée entre xₖ et xₖ₊₁
        # C'est un terme de rang 1 positif semi-défini
        term_secante = np.outer(sk, sk) / sy

        # ── Mise à jour finale ────────────────────────────────────────────
        self.H = self.H - term_retrait + term_secante

        # Remarque : la mise à jour DFP est de rang 2 au total
        # (soustraction d'un rang 1 + addition d'un rang 1).
        # Contrairement à BFGS, elle n'a PAS la forme symétrique
        # (I − ρ s yᵀ) H (I − ρ y sᵀ), ce qui la rend moins stable
        # face aux erreurs d'arrondi accumulées.

    # ──────────────────────────────────────────────────────────────────────────
    def optimize(self, f, grad, x0, max_iter=1000, tol=1e-6):
        """
        Lance l'optimisation DFP depuis x0.

        Algorithme (identique à BFGS, seule la mise à jour change) :
        ──────────────────────────────────────────────────────────────
          Pour k = 0, 1, 2, ... :
            1. dₖ = −Hₖ ∇f(xₖ)                  (direction DFP)
            2. αₖ = wolfe_line_search(...)         (pas)
            3. sₖ = αₖ dₖ,  yₖ = ∇f(xₖ+sₖ)−∇f(xₖ)
            4. Hₖ₊₁ = DFP_update(Hₖ, sₖ, yₖ)    (mise à jour DFP !)
            5. xₖ₊₁ = xₖ + sₖ

        Critère d'arrêt : ||∇f(xₖ)||/||∇f(x₀)|| < tol

        Paramètres
        ──────────
        f        : callable  — f : ℝⁿ → ℝ
        grad     : callable  — ∇f : ℝⁿ → ℝⁿ
        x0       : np.ndarray (n,)
        max_iter : int
        tol      : float

        Retourne
        ────────
        dict : x, f_final, iters, nfev, ngev, f (historique), grad_norm
        """

        # ── Initialisation ────────────────────────────────────────────────
        xk = x0.copy().astype(float)
        self.H = np.eye(self.n)    # réinitialiser H₀ = Iₙ

        fk = f(xk)
        gk = grad(xk)

        g0_norm = np.linalg.norm(gk)
        if g0_norm < 1e-12:
            return {'x': xk, 'f_final': fk, 'iters': 0,
                    'nfev': 1, 'ngev': 1, 'f': [fk], 'grad_norm': [0.0]}

        nfev = 1
        ngev = 1
        history_f     = [fk]
        history_gnorm = [1.0]

        # ── Boucle principale ─────────────────────────────────────────────
        for k in range(max_iter):

            # ── Critère d'arrêt ───────────────────────────────────────────
            gnorm = np.linalg.norm(gk) / g0_norm
            history_gnorm.append(gnorm)
            if gnorm < tol:
                break

            # ── Étape 1 : direction de descente DFP ──────────────────────
            dk = self.direction(gk)    # dₖ = −Hₖ ∇f(xₖ)

            # Sécurité : si dₖ n'est pas une direction de descente
            # (Hₖ mal conditionnée), on repart du gradient pur et
            # on réinitialise H pour "repartir proprement".
            if dk @ gk >= 0:
                dk = -gk
                self.H = np.eye(self.n)   # reset H → prochain pas = gradient

            # ── Étape 2 : recherche linéaire ──────────────────────────────
            alpha, fk_new, gk_new = wolfe_line_search(
                f, grad, xk, dk, fk, gk
            )
            nfev += 1
            ngev += 1

            # ── Étape 3 : déplacement et variation du gradient ────────────
            sk = alpha * dk        # sₖ = αₖ dₖ
            yk = gk_new - gk       # yₖ = ∇f(xₖ₊₁) − ∇f(xₖ)

            # ── Étape 4 : mise à jour DFP de Hₖ ──────────────────────────
            # C'est ici que DFP diffère de BFGS !
            self.update(sk, yk)

            # ── Étape 5 : mise à jour de xₖ ──────────────────────────────
            xk = xk + sk
            fk = fk_new
            gk = gk_new

            history_f.append(fk)

        return {
            'x'         : xk,
            'f_final'   : fk,
            'iters'     : k + 1,
            'nfev'      : nfev,
            'ngev'      : ngev,
            'f'         : history_f,
            'grad_norm' : history_gnorm,
        }


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 3 : Comparaison DFP vs BFGS côte à côte
# ══════════════════════════════════════════════════════════════════════════════

def rosenbrock(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
               for i in range(len(x) - 1))

def rosenbrock_grad(x):
    n = len(x)
    g = np.zeros(n)
    for i in range(n - 1):
        g[i]   += -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
        g[i+1] +=  200 * (x[i+1] - x[i]**2)
    return g


# Import BFGS depuis le fichier précédent pour comparaison directe
class BFGS_ref:
    """BFGS de référence (copie minimale pour la comparaison)."""
    name = "BFGS"
    def __init__(self, n):
        self.n = n; self.H = np.eye(n)
    def update(self, sk, yk):
        rho_d = yk @ sk
        if rho_d < 1e-10: return
        rho = 1.0 / rho_d
        I = np.eye(self.n)
        A = I - rho * np.outer(sk, yk)
        B = I - rho * np.outer(yk, sk)
        self.H = A @ self.H @ B + rho * np.outer(sk, sk)
    def optimize(self, f, grad, x0, max_iter=1000, tol=1e-6):
        xk = x0.copy().astype(float); self.H = np.eye(self.n)
        fk = f(xk); gk = grad(xk); g0 = np.linalg.norm(gk)
        nfev = ngev = 1; hf = [fk]; hg = [1.0]
        for k in range(max_iter):
            gn = np.linalg.norm(gk)/g0; hg.append(gn)
            if gn < tol: break
            dk = -self.H @ gk
            if dk @ gk >= 0: dk = -gk; self.H = np.eye(self.n)
            alpha, fk, gk_new = wolfe_line_search(f, grad, xk, dk, fk, gk)
            nfev += 1; ngev += 1
            sk = alpha * dk; yk = gk_new - gk
            self.update(sk, yk)
            xk = xk + sk; gk = gk_new; hf.append(fk)
        return {'x': xk, 'f_final': fk, 'iters': k+1, 'nfev': nfev,
                'ngev': ngev, 'f': hf, 'grad_norm': hg}


if __name__ == "__main__":

    print("=" * 65)
    print("Comparaison DFP vs BFGS sur Rosenbrock étendu")
    print("=" * 65)
    print(f"\n{'Méthode':<10} {'n':>5} {'Iters':>7} {'nfev':>6} "
          f"{'||g||/||g0||':>14} {'f final':>14} {'Succès':>8}")
    print("─" * 65)

    for n in [2, 10, 50, 100]:
        x0 = np.array([-1.2, 1.0] * (n // 2))
        for SolverClass in [BFGS_ref, DFP]:
            s = SolverClass(n)
            r = s.optimize(rosenbrock, rosenbrock_grad, x0, max_iter=2000)
            ok = "✓" if r['grad_norm'][-1] < 1e-5 else "✗"
            print(f"{s.name:<10} {n:>5} {r['iters']:>7} {r['nfev']:>6} "
                  f"{r['grad_norm'][-1]:>14.2e} {r['f_final']:>14.6e} {ok:>8}")
        print()

    print("\n── Différence principale entre DFP et BFGS ─────────────────────")
    print("""
  BFGS update :
    Hₖ₊₁ = (I − ρ s yᵀ) Hₖ (I − ρ y sᵀ) + ρ s sᵀ
    → forme multiplicative, plus robuste aux erreurs d'arrondi

  DFP update :
    Hₖ₊₁ = Hₖ − (Hₖ y)(Hₖ y)ᵀ / (yᵀ Hₖ y) + s sᵀ / (yᵀ s)
    → forme additive, sensible si yᵀHy devient petit

  En grande dimension, DFP accumule les erreurs plus vite → moins robuste.
    """)