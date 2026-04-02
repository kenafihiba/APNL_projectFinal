"""
Projet 2 - Étape 2 : Benchmark structuré COMPLET (50+ problèmes)
=================================================================
Couvre :
  - Dimensions     : n = 50, 100, 500, 1000, 5000
  - Conditionnement: κ = 10, 100, 1000, 10000
  - Familles       : Quadratiques, Rosenbrock, CUTEst-like, Non-convexes
  test----
  
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Problem:
    name: str
    f: Callable
    grad: Callable
    x0: np.ndarray
    n: int
    category: str
    kappa: Optional[float] = None
    f_opt: float = 0.0

    def __repr__(self):
        k = f"{self.kappa:.0f}" if self.kappa else "—"
        return f"Problem({self.name}, n={self.n}, cat={self.category}, κ={k})"


# ══════════════════════════════════════════════════════════════════════════════
# A. QUADRATIQUES  —  f(x) = ½ xᵀAx + bᵀx
#    κ(A) ∈ {10, 100, 1000, 10000}  ×  n ∈ {50, 100, 500, 1000, 5000}
#    → 20 problèmes
# ══════════════════════════════════════════════════════════════════════════════

def make_quadratic(n, kappa, seed=42):
    rng = np.random.default_rng(seed)
    eigvals = np.exp(np.linspace(0, np.log(kappa), n))
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Q @ np.diag(eigvals) @ Q.T
    A = (A + A.T) / 2
    b = rng.standard_normal(n)
    x_opt = np.linalg.solve(A, -b)
    f_opt = 0.5 * x_opt @ A @ x_opt + b @ x_opt

    def f(x):   return 0.5 * x @ A @ x + b @ x
    def g(x):   return A @ x + b

    x0 = rng.standard_normal(n)
    return Problem(f"Quadratic_n{n}_k{int(kappa)}", f, g,
                   x0, n, "quadratic", float(kappa), f_opt)


# ══════════════════════════════════════════════════════════════════════════════
# B. ROSENBROCK ÉTENDU  —  Σ[100(x_{i+1}−xᵢ²)² + (1−xᵢ)²]
#    n ∈ {50, 100, 500, 1000, 5000}  → 5 problèmes
# ══════════════════════════════════════════════════════════════════════════════

def make_rosenbrock(n):
    def f(x):
        return sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2
                   for i in range(n-1))
    def g(x):
        gr = np.zeros(n)
        for i in range(n-1):
            gr[i]   += -400*x[i]*(x[i+1]-x[i]**2) - 2*(1-x[i])
            gr[i+1] +=  200*(x[i+1]-x[i]**2)
        return gr
    x0 = np.array([-1.2, 1.0] * (n//2))
    return Problem(f"Rosenbrock_n{n}", f, g, x0, n, "rosenbrock")


# ══════════════════════════════════════════════════════════════════════════════
# C. PROBLÈMES CUTEst SÉLECTIONNÉS (implémentés analytiquement)
# ══════════════════════════════════════════════════════════════════════════════

# ── C1. ARWHEAD (Arrowhead) ──────────────────────────────────────────────────
# f(x) = Σᵢ₌₁ⁿ⁻¹ [( xᵢ²+xₙ²)² − 4xᵢ + 3]
# Structure en flèche (sparse), bien conditionné
# n ∈ {50, 100, 500, 1000, 5000}  → 5 problèmes

def make_arwhead(n):
    def f(x):
        val = 0.0
        for i in range(n-1):
            val += (x[i]**2 + x[n-1]**2)**2 - 4*x[i] + 3
        return val
    def g(x):
        gr = np.zeros(n)
        for i in range(n-1):
            t = x[i]**2 + x[n-1]**2
            gr[i]   += 4*x[i]*t - 4
            gr[n-1] += 4*x[n-1]*t
        return gr
    x0 = np.ones(n)
    return Problem(f"ARWHEAD_n{n}", f, g, x0, n, "cutest")


# ── C2. ENGVAL1 ──────────────────────────────────────────────────────────────
# f(x) = Σᵢ₌₁ⁿ⁻¹ [(xᵢ²+xᵢ₊₁²)² − 4xᵢ + 3]
# Similaire à ARWHEAD mais structure chaîne (tridiagonale)
# n ∈ {50, 100, 500, 1000, 5000}  → 5 problèmes

def make_engval1(n):
    def f(x):
        val = 0.0
        for i in range(n-1):
            val += (x[i]**2 + x[i+1]**2)**2 - 4*x[i] + 3
        return val
    def g(x):
        gr = np.zeros(n)
        for i in range(n-1):
            t = x[i]**2 + x[i+1]**2
            gr[i]   += 4*x[i]*t - 4
            gr[i+1] += 4*x[i+1]*t
        return gr
    x0 = np.ones(n)
    return Problem(f"ENGVAL1_n{n}", f, g, x0, n, "cutest")


# ── C3. TRIDIA (Tridiagonale) ────────────────────────────────────────────────
# f(x) = (x₁−1)² + Σᵢ₌₂ⁿ i(2xᵢ−xᵢ₋₁)²
# Très mal conditionné (κ ~ n²), teste la robustesse
# n ∈ {50, 100, 500, 1000, 5000}  → 5 problèmes

def make_tridia(n):
    def f(x):
        val = (x[0]-1)**2
        for i in range(1, n):
            val += (i+1) * (2*x[i] - x[i-1])**2
        return val
    def g(x):
        gr = np.zeros(n)
        gr[0] = 2*(x[0]-1)
        for i in range(1, n):
            diff = 2*x[i] - x[i-1]
            gr[i]   +=  (i+1)*2*diff*2
            gr[i-1] += -(i+1)*2*diff
        return gr
    x0 = np.ones(n) / n
    return Problem(f"TRIDIA_n{n}", f, g, x0, n, "cutest",
                   kappa=float(n**2))


# ── C4. NONDQUAR (Non-Diagonal Quartic) ──────────────────────────────────────
# f(x) = (x₁+x₂)⁴ + Σᵢ₌₁ⁿ⁻² (xᵢ−xᵢ₊₂)⁴ + (xₙ₋₁+xₙ)⁴ - (x₁+x₂)⁴/...
# Non-quadratique, structure creuse
# n ∈ {50, 100, 500, 1000}  → 4 problèmes

def make_nondquar(n):
    def f(x):
        val = (x[0] + x[1])**4
        for i in range(n-2):
            val += (x[i] - x[i+2])**4
        val += (x[n-2] + x[n-1])**4
        return val
    def g(x):
        gr = np.zeros(n)
        gr[0] += 4*(x[0]+x[1])**3
        gr[1] += 4*(x[0]+x[1])**3
        for i in range(n-2):
            t = 4*(x[i]-x[i+2])**3
            gr[i]   += t
            gr[i+2] -= t
        gr[n-2] += 4*(x[n-2]+x[n-1])**3
        gr[n-1] += 4*(x[n-2]+x[n-1])**3
        return gr
    rng = np.random.default_rng(7)
    x0 = rng.standard_normal(n)
    return Problem(f"NONDQUAR_n{n}", f, g, x0, n, "cutest")


# ── C5. DIXMAANE (Dixon-Maany famille E) ─────────────────────────────────────
# f(x) = 1 + Σᵢ (αxᵢ² + βxᵢ²xᵢ₊ₘ² + γxᵢ²xᵢ₊₂ₘ² + δxᵢxᵢ₊ₘ)
# avec α=1, β=γ=δ=0.125, m=n//3
# Non-séparable, bien conditionné
# n ∈ {300, 900, 3000} (multiples de 3)  → 3 problèmes

def make_dixmaane(n):
    assert n % 3 == 0
    m = n // 3
    alpha, beta, gamma, delta = 1.0, 0.125, 0.125, 0.125

    def f(x):
        val = 1.0
        for i in range(n):
            val += alpha * x[i]**2
            if i + m < n:
                val += beta  * x[i]**2 * x[i+m]**2
            if i + 2*m < n:
                val += gamma * x[i]**2 * x[i+2*m]**2
            if i + m < n:
                val += delta * x[i] * x[i+m]
        return val

    def g(x):
        gr = np.zeros(n)
        for i in range(n):
            gr[i] += 2*alpha * x[i]
            if i + m < n:
                gr[i]   += 2*beta  * x[i] * x[i+m]**2 + delta
                gr[i+m] += 2*beta  * x[i]**2 * x[i+m]
            if i + 2*m < n:
                gr[i]     += 2*gamma * x[i] * x[i+2*m]**2
                gr[i+2*m] += 2*gamma * x[i]**2 * x[i+2*m]
            if i - m >= 0:
                gr[i] += delta
        return gr

    x0 = np.ones(n) * 2.0
    return Problem(f"DIXMAANE_n{n}", f, g, x0, n, "cutest")


# ══════════════════════════════════════════════════════════════════════════════
# D. NON-CONVEXES  —  plusieurs minima locaux
# ══════════════════════════════════════════════════════════════════════════════

# ── D1. Rastrigin ─────────────────────────────────────────────────────────────
# f(x) = An + Σ[xᵢ² − A cos(2πxᵢ)]   A=10
# O(10ⁿ) minima locaux, très difficile
# n ∈ {50, 100, 500, 1000}  → 4 problèmes

def make_rastrigin(n, seed=42):
    A = 10.0
    def f(x):  return A*n + np.sum(x**2 - A*np.cos(2*np.pi*x))
    def g(x):  return 2*x + 2*np.pi*A*np.sin(2*np.pi*x)
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-3, 3, n)
    return Problem(f"Rastrigin_n{n}", f, g, x0, n, "nonconvex")


# ── D2. Styblinski-Tang ───────────────────────────────────────────────────────
# f(x) = Σ(xᵢ⁴ − 16xᵢ² + 5xᵢ)/2
# Minimum global ≈ −39.166×n  en  x*≈(−2.9035,…)
# n ∈ {50, 100, 500, 1000}  → 4 problèmes

def make_styblinski(n, seed=42):
    def f(x):  return np.sum(x**4 - 16*x**2 + 5*x) / 2
    def g(x):  return (4*x**3 - 32*x + 5) / 2
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-4, 4, n)
    return Problem(f"StyblinskiTang_n{n}", f, g, x0, n, "nonconvex",
                   f_opt=-39.16599*n)


# ── D3. Griewank ──────────────────────────────────────────────────────────────
# f(x) = 1 + Σxᵢ²/4000 − Π cos(xᵢ/√i)
# Multi-modale mais devient presque convexe en grande dim
# n ∈ {50, 100, 500, 1000}  → 4 problèmes

def make_griewank(n, seed=42):
    def f(x):
        s = np.sum(x**2) / 4000
        p = np.prod(np.cos(x / np.sqrt(np.arange(1, n+1))))
        return 1 + s - p
    def g(x):
        idx = np.arange(1, n+1, dtype=float)
        gr = x / 2000
        p = np.prod(np.cos(x / np.sqrt(idx)))
        for i in range(n):
            c = np.cos(x[i] / np.sqrt(idx[i]))
            if abs(c) > 1e-12:
                gr[i] += p / c * np.sin(x[i]/np.sqrt(idx[i])) / np.sqrt(idx[i])
        return gr
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-5, 5, n)
    return Problem(f"Griewank_n{n}", f, g, x0, n, "nonconvex")


# ── D4. Schwefel ──────────────────────────────────────────────────────────────
# f(x) = 418.9829·n − Σ xᵢ sin(√|xᵢ|)
# Minimum global trompeur (loin du centre), xᵢ* ≈ 420.97
# n ∈ {50, 100}  (dimensions modestes car très chaotique)  → 2 problèmes

def make_schwefel(n, seed=42):
    def f(x):
        return 418.9829*n - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    def g(x):
        sqx = np.sqrt(np.abs(x) + 1e-12)
        return -(np.sin(sqx) + x * np.cos(sqx) / (2*sqx))
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-400, 400, n)
    return Problem(f"Schwefel_n{n}", f, g, x0, n, "nonconvex",
                   f_opt=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLAGE DU BENCHMARK COMPLET
# ══════════════════════════════════════════════════════════════════════════════

def build_benchmark():
    problems = []

    # ── A. Quadratiques : 4 κ × 5 dimensions = 20 problèmes ─────────────────
    for n in [50, 100, 500, 1000, 5000]:
        for kappa in [10, 100, 1000, 10000]:
            problems.append(make_quadratic(n, kappa, seed=n+int(kappa)))

    # ── B. Rosenbrock : 5 dimensions = 5 problèmes ───────────────────────────
    for n in [50, 100, 500, 1000, 5000]:
        problems.append(make_rosenbrock(n))

    # ── C. CUTEst-like : 5 familles ──────────────────────────────────────────
    # ARWHEAD : 5 dimensions = 5 problèmes
    for n in [50, 100, 500, 1000, 5000]:
        problems.append(make_arwhead(n))

    # ENGVAL1 : 5 dimensions = 5 problèmes
    for n in [50, 100, 500, 1000, 5000]:
        problems.append(make_engval1(n))

    # TRIDIA : 5 dimensions = 5 problèmes
    for n in [50, 100, 500, 1000, 5000]:
        problems.append(make_tridia(n))

    # NONDQUAR : 4 dimensions (pas 5000 — trop lent) = 4 problèmes
    for n in [50, 100, 500, 1000]:
        problems.append(make_nondquar(n))

    # DIXMAANE : multiples de 3 proches des cibles = 3 problèmes
    for n in [300, 900, 3000]:
        problems.append(make_dixmaane(n))

    # ── D. Non-convexes : 4 familles ─────────────────────────────────────────
    # Rastrigin : 4 dimensions = 4 problèmes
    for n in [50, 100, 500, 1000]:
        problems.append(make_rastrigin(n))

    # Styblinski-Tang : 4 dimensions = 4 problèmes
    for n in [50, 100, 500, 1000]:
        problems.append(make_styblinski(n))

    # Griewank : 4 dimensions = 4 problèmes
    for n in [50, 100, 500, 1000]:
        problems.append(make_griewank(n))

    # Schwefel : 2 dimensions = 2 problèmes
    for n in [50, 100]:
        problems.append(make_schwefel(n))

    return problems


# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    problems = build_benchmark()

    # Comptage par catégorie
    cats = {}
    for p in problems:
        cats.setdefault(p.category, []).append(p)

    print("=" * 60)
    print(f"  BENCHMARK COMPLET  —  {len(problems)} problèmes")
    print("=" * 60)
    for cat, probs in cats.items():
        dims = sorted(set(p.n for p in probs))
        print(f"  {cat:<14}: {len(probs):>3} problèmes  | n ∈ {dims}")

    print()
    # Tableau complet
    print(f"  {'#':<4} {'Nom':<25} {'n':>6} {'catégorie':<14} {'κ':>10}")
    print("  " + "─" * 63)
    for i, p in enumerate(problems, 1):
        k = f"{p.kappa:.0f}" if p.kappa else "—"
        print(f"  {i:<4} {p.name:<25} {p.n:>6} {p.category:<14} {k:>10}")

    # Vérification dimensionnelle
    print()
    print("  Dimensions couvertes :", sorted(set(p.n for p in problems)))
    print("  κ couverts :", sorted(set(int(p.kappa) for p in problems if p.kappa)))
