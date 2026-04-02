"""
═══════════════════════════════════════════════════════════════════════════════
MÉTHODE BFGS — Broyden-Fletcher-Goldfarb-Shanno
═══════════════════════════════════════════════════════════════════════════════

Contexte théorique :
────────────────────
Les méthodes quasi-Newton cherchent le minimum d'une fonction f : ℝⁿ → ℝ
en construisant une approximation Hₖ de l'inverse de la Hessienne ∇²f(xₖ)⁻¹,
sans jamais calculer cette Hessienne exacte (trop coûteuse : O(n²) en mémoire,
O(n³) pour l'inversion).

L'idée clé est de mettre à jour Hₖ à chaque itération en utilisant uniquement
les gradients calculés, via la formule BFGS.

La direction de descente est : dₖ = -Hₖ · ∇f(xₖ)
Si Hₖ est une bonne approximation de ∇²f(xₖ)⁻¹, alors dₖ ≈ direction de Newton.

Formule de mise à jour BFGS :
──────────────────────────────
Soient :
  sₖ = xₖ₊₁ - xₖ           (pas dans l'espace des itérés)
  yₖ = ∇f(xₖ₊₁) - ∇f(xₖ)  (pas dans l'espace des gradients)
  ρₖ = 1 / (yₖᵀ sₖ)        (scalaire)

Mise à jour :
  Hₖ₊₁ = (I - ρₖ sₖ yₖᵀ) Hₖ (I - ρₖ yₖ sₖᵀ) + ρₖ sₖ sₖᵀ

Propriétés :
  - Hₖ₊₁ reste symétrique définie positive si yₖᵀsₖ > 0
  - Satisfait l'équation sécante : Hₖ₊₁ yₖ = sₖ
  - Convergence super-linéaire sur fonctions convexes lisses
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Recherche linéaire de Wolfe (forte)
# ─────────────────────────────────────────────────────────────────────────────

def wolfe_line_search(f, grad, xk, dk, fk, gk, c1=1e-4, c2=0.9, max_iter=50):
    """
    Recherche linéaire satisfaisant les conditions de Wolfe fortes.

    Les conditions de Wolfe garantissent :
      1. Condition d'Armijo (décroissance suffisante) :
            f(xₖ + α dₖ) ≤ f(xₖ) + c1 · α · (∇f(xₖ)ᵀ dₖ)
         → on accepte α seulement si f a assez diminué

      2. Condition de courbure forte :
            |∇f(xₖ + α dₖ)ᵀ dₖ| ≤ c2 · |∇f(xₖ)ᵀ dₖ|
         → on rejette les α trop petits (pente encore trop forte)

    Ces deux conditions ensemble garantissent yₖᵀsₖ > 0,
    ce qui est indispensable pour que la mise à jour BFGS reste SPD.

    Paramètres :
    ────────────
    f, grad  : fonction objectif et son gradient
    xk       : point courant
    dk       : direction de descente
    fk, gk   : f(xk) et ∇f(xk) déjà calculés (évite un appel)
    c1       : paramètre Armijo ∈ (0, 1),  typiquement 1e-4
    c2       : paramètre courbure ∈ (c1, 1), typiquement 0.9 pour quasi-Newton

    Retourne :
    ──────────
    alpha    : pas accepté
    f_new    : f(xk + alpha * dk)
    g_new    : ∇f(xk + alpha * dk)
    """
    alpha = 1.0          # on commence toujours avec le pas unité
    alpha_lo = 0.0       # borne inférieure de l'intervalle de recherche
    alpha_hi = np.inf    # borne supérieure (inconnue au début)

    # Pente directionnelle initiale : ∇f(xₖ)ᵀ dₖ (doit être < 0 si dk descend)
    gd = gk @ dk

    for _ in range(max_iter):
        x_new = xk + alpha * dk
        f_new = f(x_new)
        g_new = grad(x_new)

        # ── Test 1 : condition d'Armijo ───────────────────────────────────────
        if f_new > fk + c1 * alpha * gd:
            # f n'a pas assez diminué → alpha trop grand → on rétrécit
            alpha_hi = alpha
            alpha = (alpha_lo + alpha_hi) / 2
            continue

        # ── Test 2 : condition de courbure forte ──────────────────────────────
        if abs(g_new @ dk) <= c2 * abs(gd):
            # Les deux conditions sont satisfaites → on accepte alpha
            return alpha, f_new, g_new

        # Armijo OK mais courbure trop forte → alpha trop petit → on agrandit
        if g_new @ dk < 0:
            alpha_lo = alpha
            # Soit on double, soit on prend le milieu si alpha_hi est borné
            alpha = min(2 * alpha, alpha_hi) if np.isinf(alpha_hi) else (alpha_lo + alpha_hi) / 2
        else:
            alpha_hi = alpha
            alpha = (alpha_lo + alpha_hi) / 2

    # Si on dépasse max_iter, on retourne le dernier alpha calculé
    return alpha, f(xk + alpha * dk), grad(xk + alpha * dk)


# ─────────────────────────────────────────────────────────────────────────────
# Classe BFGS
# ─────────────────────────────────────────────────────────────────────────────

class BFGS:
    """
    Implémentation de la méthode BFGS.

    Algorithme général (une itération) :
    ─────────────────────────────────────
      1. Calculer la direction : dₖ = -Hₖ ∇f(xₖ)
      2. Recherche linéaire : trouver αₖ satisfaisant Wolfe
      3. Mettre à jour le point : xₖ₊₁ = xₖ + αₖ dₖ
      4. Calculer sₖ = xₖ₊₁ - xₖ  et  yₖ = ∇f(xₖ₊₁) - ∇f(xₖ)
      5. Mettre à jour Hₖ → Hₖ₊₁ via la formule BFGS

    Complexité par itération :
      - Direction : O(n²)  (produit matrice-vecteur Hₖ · gₖ)
      - Mise à jour BFGS : O(n²)  (produits externes sₖsₖᵀ, etc.)
      → Inadapté pour n > 10 000 (utiliser L-BFGS à la place)
    """

    name = "BFGS"

    def __init__(self, n):
        """
        Initialise l'approximation inverse Hessienne à l'identité.

        H₀ = I est le choix standard : au départ on ne sait rien de la
        courbure, donc on suppose qu'elle est isotrope (= gradient descent
        avec pas unité). H s'adaptera automatiquement au fil des itérations.

        Paramètres :
        ────────────
        n : dimension du problème
        """
        self.n = n
        self.H = np.eye(n)   # H₀ = Iₙ  (approximation initiale)

    def direction(self, gk):
        """
        Calcule la direction de descente quasi-Newton.

            dₖ = -Hₖ · ∇f(xₖ)

        Si Hₖ = ∇²f(xₖ)⁻¹ exactement → direction de Newton.
        Si Hₖ = I → gradient négatif (steepest descent).

        Paramètres :
        ────────────
        gk : ∇f(xₖ), vecteur de taille n

        Retourne :
        ──────────
        dk : direction de descente, vecteur de taille n
        """
        return -self.H @ gk

    def update(self, sk, yk):
        """
        Mise à jour de l'approximation inverse Hessienne Hₖ → Hₖ₊₁.

        Formule BFGS (rang-2, symétrique) :

            ρₖ   = 1 / (yₖᵀ sₖ)
            A    = I - ρₖ sₖ yₖᵀ
            B    = I - ρₖ yₖ sₖᵀ  (= Aᵀ)
            Hₖ₊₁ = A Hₖ B + ρₖ sₖ sₖᵀ

        Interprétation :
          - Le terme A Hₖ B "projette" l'ancienne approximation dans
            le sous-espace compatible avec la nouvelle information de courbure.
          - Le terme ρₖ sₖsₖᵀ est une correction de rang 1 qui force
            la condition sécante : Hₖ₊₁ yₖ = sₖ.

        Condition de mise à jour :
          - yₖᵀ sₖ > 0 est nécessaire pour que ρₖ > 0 et Hₖ₊₁ reste SPD.
          - Si cette condition échoue (courbure négative ou nulle), on
            SAUTE la mise à jour (H reste inchangée) pour ne pas corrompre
            l'approximation.

        Paramètres :
        ────────────
        sk : sₖ = xₖ₊₁ - xₖ           (vecteur de taille n)
        yk : yₖ = ∇f(xₖ₊₁) - ∇f(xₖ) (vecteur de taille n)
        """
        # ── Vérification de la condition de courbure ─────────────────────────
        rho_denom = yk @ sk   # = yₖᵀ sₖ

        if rho_denom < 1e-10:
            # Condition de courbure positive non satisfaite.
            # On ne met PAS à jour H pour éviter de la rendre indéfinie.
            # Cela peut arriver sur des fonctions non-convexes ou quand
            # la recherche linéaire est imprécise.
            return

        # ── Calcul du scalaire ρₖ = 1 / (yₖᵀ sₖ) ────────────────────────────
        rho = 1.0 / rho_denom

        # ── Construction des matrices A et B = Aᵀ ────────────────────────────
        n = len(sk)
        I = np.eye(n)

        # A = I - ρₖ sₖ yₖᵀ  (matrice n×n, mise à jour de rang 1 sur I)
        A = I - rho * np.outer(sk, yk)

        # B = I - ρₖ yₖ sₖᵀ  (= Aᵀ, par symétrie de la formule BFGS)
        B = I - rho * np.outer(yk, sk)

        # ── Formule BFGS complète ─────────────────────────────────────────────
        # Hₖ₊₁ = A Hₖ Bᵀ + ρₖ sₖ sₖᵀ
        self.H = A @ self.H @ B + rho * np.outer(sk, sk)

    def optimize(self, f, grad, x0, max_iter=1000, tol=1e-6):
        """
        Boucle principale de l'optimisation BFGS.

        Critère d'arrêt :
          ||∇f(xₖ)|| / ||∇f(x₀)|| < tol
          → on normalise par la norme initiale pour être invariant à l'échelle

        Paramètres :
        ────────────
        f        : fonction objectif  f : ℝⁿ → ℝ
        grad     : gradient           ∇f : ℝⁿ → ℝⁿ
        x0       : point de départ    vecteur de taille n
        max_iter : nombre max d'itérations
        tol      : tolérance sur le gradient normalisé

        Retourne :
        ──────────
        history : dict contenant
          - 'x'        : solution approchée finale
          - 'f_final'  : valeur de f en x final
          - 'iters'    : nombre d'itérations effectuées
          - 'f'        : liste des valeurs de f à chaque itération
          - 'grad_norm': liste des normes de gradient normalisées
          - 'nfev'     : nombre d'évaluations de f
          - 'ngev'     : nombre d'évaluations de ∇f
        """
        # ── Initialisation ────────────────────────────────────────────────────
        xk = x0.copy().astype(float)   # copie pour ne pas modifier x0
        self.H = np.eye(self.n)        # réinitialise H à chaque appel

        fk = f(xk)
        gk = grad(xk)
        g0_norm = np.linalg.norm(gk)   # norme initiale pour normalisation

        # Historique des métriques (utile pour les profils de performance)
        history = {
            "f":         [fk],
            "grad_norm": [1.0],    # au départ = 1 par définition
            "nfev":      1,        # compteur d'évaluations de f
            "ngev":      1,        # compteur d'évaluations de ∇f
        }

        # ── Boucle principale ─────────────────────────────────────────────────
        for k in range(max_iter):

            # ── Critère d'arrêt : gradient suffisamment petit ─────────────────
            gnorm = np.linalg.norm(gk) / max(g0_norm, 1e-12)
            history["grad_norm"].append(gnorm)

            if gnorm < tol:
                break   # convergence atteinte

            # ── Étape 1 : direction de descente quasi-Newton ──────────────────
            dk = self.direction(gk)    # dk = -Hk · gk

            # Vérification de sécurité : dk doit pointer vers le bas
            # Si dk @ gk >= 0, la direction monte → on replie sur le gradient
            if dk @ gk >= 0:
                dk = -gk   # fallback sur la descente de gradient
                # Cela arrive si H est devenue presque singulière ou indéfinie

            # ── Étape 2 : recherche linéaire (conditions de Wolfe) ────────────
            alpha, fk_new, gk_new = wolfe_line_search(f, grad, xk, dk, fk, gk)
            history["nfev"] += 1
            history["ngev"] += 1

            # ── Étape 3 : mise à jour du point ────────────────────────────────
            sk = alpha * dk              # sₖ = αₖ dₖ = xₖ₊₁ - xₖ
            yk = gk_new - gk             # yₖ = ∇f(xₖ₊₁) - ∇f(xₖ)

            xk = xk + sk                 # xₖ₊₁ = xₖ + sₖ
            fk = fk_new
            gk = gk_new

            # ── Étape 4 : mise à jour de l'approximation Hessienne ───────────
            self.update(sk, yk)          # Hₖ → Hₖ₊₁ via formule BFGS

            # Enregistrement des métriques
            history["f"].append(fk)

        # ── Résultat final ─────────────────────────────────────────────────────
        history["x"]       = xk
        history["f_final"] = fk
        history["iters"]   = k + 1

        return history


# ─────────────────────────────────────────────────────────────────────────────
# Test sur la fonction de Rosenbrock (n=2)
# ─────────────────────────────────────────────────────────────────────────────

def rosenbrock(x):
    """f(x,y) = (1-x)² + 100(y-x²)²   — minimum global en (1,1), f*=0"""
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))

def rosenbrock_grad(x):
    """Gradient analytique de Rosenbrock."""
    n = len(x)
    g = np.zeros(n)
    for i in range(n - 1):
        g[i]   += -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
        g[i+1] += 200 * (x[i+1] - x[i]**2)
    return g


if __name__ == "__main__":
    n  = 2
    x0 = np.array([-1.2, 1.0])   # point de départ classique

    solver = BFGS(n)
    res = solver.optimize(rosenbrock, rosenbrock_grad, x0, tol=1e-8)

    print("═" * 50)
    print("BFGS — Rosenbrock n=2")
    print("═" * 50)
    print(f"  Point initial x0     : {x0}")
    print(f"  Solution trouvée x*  : {res['x'].round(8)}")
    print(f"  f(x*)                : {res['f_final']:.2e}  (attendu : 0)")
    print(f"  ||∇f||/||∇f₀||       : {res['grad_norm'][-1]:.2e}")
    print(f"  Itérations           : {res['iters']}")
    print(f"  Évaluations f        : {res['nfev']}")
    print(f"  Évaluations ∇f       : {res['ngev']}")
    print("═" * 50)