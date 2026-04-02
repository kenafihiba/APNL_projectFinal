"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Quasi-Newton Benchmark — Application Streamlit                           ║
║   Projet 2 : Comparaison empirique de variantes quasi-Newton               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Installation :
    pip install streamlit matplotlib seaborn pandas numpy

Lancement :
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import time
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from collections import deque

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Quasi-Newton Benchmark",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    code, .stCode { font-family: 'JetBrains Mono', monospace !important; }

    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #334155;
    }
    .main-header h1 {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
    }

    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1;
    }
    .metric-card .label {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #0f172a;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #3266ad;
        display: inline-block;
    }

    .solver-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }

    .stProgress > div > div { background-color: #3266ad; }

    div[data-testid="stSidebar"] {
        background: #f1f5f9;
        border-right: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SOLVEURS (copiés inline pour autonomie du fichier)
# ══════════════════════════════════════════════════════════════════════════════

def wolfe_line_search(f, grad, xk, dk, fk, gk, c1=1e-4, c2=0.9, max_iter=50):
    gd = gk @ dk
    alpha, alpha_lo, alpha_hi = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        x_new = xk + alpha * dk
        f_new = f(x_new)
        g_new = grad(x_new)
        if f_new > fk + c1 * alpha * gd:
            alpha_hi = alpha
            alpha = (alpha_lo + alpha_hi) / 2
            continue
        if abs(g_new @ dk) <= c2 * abs(gd):
            return alpha, f_new, g_new
        if g_new @ dk < 0:
            alpha_lo = alpha
            alpha = min(2*alpha, alpha_hi) if np.isinf(alpha_hi) else (alpha_lo+alpha_hi)/2
        else:
            alpha_hi = alpha
            alpha = (alpha_lo + alpha_hi) / 2
    return alpha, f(xk+alpha*dk), grad(xk+alpha*dk)


def _base_optimize(solver_obj, f, grad, x0, max_iter, tol):
    xk = x0.copy().astype(float)
    fk = f(xk); gk = grad(xk)
    g0 = np.linalg.norm(gk)
    if g0 < 1e-12:
        return {'x':xk,'f_final':fk,'iters':0,'nfev':1,'ngev':1,'f':[fk],'grad_norm':[0.0]}
    nfev=ngev=1; hf=[fk]; hg=[1.0]
    for k in range(max_iter):
        gn = np.linalg.norm(gk)/g0; hg.append(gn)
        if gn < tol: break
        dk = solver_obj._direction(gk)
        if dk @ gk >= 0: dk=-gk; solver_obj._reset()
        alpha, fk, gk_new = wolfe_line_search(f, grad, xk, dk, fk, gk)
        nfev+=1; ngev+=1
        sk=alpha*dk; yk=gk_new-gk
        solver_obj._update(sk, yk)
        xk=xk+sk; gk=gk_new; hf.append(fk)
    return {'x':xk,'f_final':fk,'iters':k+1,'nfev':nfev,'ngev':ngev,'f':hf,'grad_norm':hg}


class BFGS:
    name="BFGS"
    color="#3266ad"
    def __init__(self,n=2): self.n=n; self.H=np.eye(n)
    def _reset(self): self.H=np.eye(self.n)
    def _direction(self,g): return -self.H@g
    def _update(self,s,y):
        r=y@s
        if r<1e-10: return
        rho=1/r; I=np.eye(self.n)
        A=I-rho*np.outer(s,y); B=I-rho*np.outer(y,s)
        self.H=A@self.H@B+rho*np.outer(s,s)
    def optimize(self,f,g,x0,max_iter=1000,tol=1e-6):
        self.n=len(x0); self.H=np.eye(self.n)
        return _base_optimize(self,f,g,x0,max_iter,tol)

class LBFGS:
    color="#1d9e75"
    def __init__(self,m=10): self.m=m; self.name=f"L-BFGS(m={m})"; self.s=[]; self.y=[]; self.r=[]
    def _reset(self): self.s.clear(); self.y.clear(); self.r.clear()
    def _direction(self,g):
        q=g.copy(); alphas=[]
        for s,y,rho in zip(reversed(self.s),reversed(self.y),reversed(self.r)):
            a=rho*(s@q); q=q-a*y; alphas.append(a)
        gamma=((self.s[-1]@self.y[-1])/(self.y[-1]@self.y[-1])) if self.s else 1.0
        r=gamma*q
        for s,y,rho,a in zip(self.s,self.y,self.r,reversed(alphas)):
            b=rho*(y@r); r=r+s*(a-b)
        return -r
    def _update(self,s,y):
        d=y@s
        if d>1e-10:
            if len(self.s)>=self.m: self.s.pop(0); self.y.pop(0); self.r.pop(0)
            self.s.append(s.copy()); self.y.append(y.copy()); self.r.append(1/d)
    def optimize(self,f,g,x0,max_iter=1000,tol=1e-6):
        self._reset(); return _base_optimize(self,f,g,x0,max_iter,tol)

class SR1:
    name="SR1"; color="#d4537e"
    def __init__(self,n=2): self.n=n; self.H=np.eye(n)
    def _reset(self): self.H=np.eye(self.n)
    def _direction(self,g): return -self.H@g
    def _update(self,s,y):
        v=s-self.H@y; d=v@y
        if abs(d)>=1e-8*np.linalg.norm(v)*np.linalg.norm(y):
            self.H+=np.outer(v,v)/d
    def optimize(self,f,g,x0,max_iter=1000,tol=1e-6):
        self.n=len(x0); self.H=np.eye(self.n); return _base_optimize(self,f,g,x0,max_iter,tol)

class DFP:
    name="DFP"; color="#e24b4a"
    def __init__(self,n=2): self.n=n; self.H=np.eye(n)
    def _reset(self): self.H=np.eye(self.n)
    def _direction(self,g): return -self.H@g
    def _update(self,s,y):
        sy=y@s; Hy=self.H@y; yHy=y@Hy
        if sy<1e-10 or yHy<1e-10: return
        self.H=self.H-np.outer(Hy,Hy)/yHy+np.outer(s,s)/sy
    def optimize(self,f,g,x0,max_iter=1000,tol=1e-6):
        self.n=len(x0); self.H=np.eye(self.n); return _base_optimize(self,f,g,x0,max_iter,tol)

class Broyden:
    color="#ba7517"
    def __init__(self,n=2,phi=0.5): self.n=n; self.phi=phi; self.H=np.eye(n); self.name=f"Broyden(φ={phi})"
    def _reset(self): self.H=np.eye(self.n)
    def _direction(self,g): return -self.H@g
    def _update(self,s,y):
        sy=y@s
        if sy<1e-10: return
        rho=1/sy; I=np.eye(self.n)
        A=I-rho*np.outer(s,y); B=I-rho*np.outer(y,s)
        H_bfgs=A@self.H@B+rho*np.outer(s,s)
        yHy=y@self.H@y
        if yHy<1e-10: self.H=H_bfgs; return
        H_dfp=self.H-np.outer(self.H@y,self.H@y)/yHy+np.outer(s,s)/sy
        self.H=(1-self.phi)*H_dfp+self.phi*H_bfgs
    def optimize(self,f,g,x0,max_iter=1000,tol=1e-6):
        self.n=len(x0); self.H=np.eye(self.n); return _base_optimize(self,f,g,x0,max_iter,tol)

class HybridBFGSSR1:
    name="Hybrid"; color="#7f77dd"
    def __init__(self,n=2,thr=0.1): self.n=n; self.H=np.eye(n); self.thr=thr
    def _reset(self): self.H=np.eye(self.n)
    def _direction(self,g): return -self.H@g
    def _update(self,s,y):
        sy=y@s; ns=np.linalg.norm(s); ny=np.linalg.norm(y)
        if sy>self.thr*ns*ny:
            rho=1/sy; I=np.eye(self.n)
            A=I-rho*np.outer(s,y); B=I-rho*np.outer(y,s)
            self.H=A@self.H@B+rho*np.outer(s,s)
        else:
            v=s-self.H@y; d=v@y
            if abs(d)>=1e-8*np.linalg.norm(v)*max(ny,1e-12):
                self.H+=np.outer(v,v)/d
    def optimize(self,f,g,x0,max_iter=1000,tol=1e-6):
        self.n=len(x0); self.H=np.eye(self.n); return _base_optimize(self,f,g,x0,max_iter,tol)


# ══════════════════════════════════════════════════════════════════════════════
# PROBLÈMES
# ══════════════════════════════════════════════════════════════════════════════

def make_quadratic(n, kappa, seed=42):
    rng=np.random.default_rng(seed)
    eigvals=np.exp(np.linspace(0,np.log(kappa),n))
    Q,_=np.linalg.qr(rng.standard_normal((n,n)))
    A=Q@np.diag(eigvals)@Q.T; A=(A+A.T)/2
    b=rng.standard_normal(n)
    x0=rng.standard_normal(n)
    return {'name':f'Quadratic κ={int(kappa)}','f':lambda x:0.5*x@A@x+b@x,
            'g':lambda x:A@x+b,'x0':x0,'cat':'quadratic','kappa':kappa}

def make_rosenbrock(n):
    def f(x): return sum(100*(x[i+1]-x[i]**2)**2+(1-x[i])**2 for i in range(n-1))
    def g(x):
        gr=np.zeros(n)
        for i in range(n-1):
            gr[i]+=-400*x[i]*(x[i+1]-x[i]**2)-2*(1-x[i]); gr[i+1]+=200*(x[i+1]-x[i]**2)
        return gr
    return {'name':f'Rosenbrock n={n}','f':f,'g':g,
            'x0':np.array([-1.2,1.0]*(n//2)),'cat':'rosenbrock','kappa':None}

def make_arwhead(n):
    def f(x):
        v=0
        for i in range(n-1): v+=(x[i]**2+x[n-1]**2)**2-4*x[i]+3
        return v
    def g(x):
        gr=np.zeros(n)
        for i in range(n-1):
            t=x[i]**2+x[n-1]**2; gr[i]+=4*x[i]*t-4; gr[n-1]+=4*x[n-1]*t
        return gr
    return {'name':f'ARWHEAD n={n}','f':f,'g':g,'x0':np.ones(n),'cat':'cutest','kappa':None}

def make_rastrigin(n):
    A=10.0
    def f(x): return A*n+np.sum(x**2-A*np.cos(2*np.pi*x))
    def g(x): return 2*x+2*np.pi*A*np.sin(2*np.pi*x)
    rng=np.random.default_rng(42)
    return {'name':f'Rastrigin n={n}','f':f,'g':g,'x0':rng.uniform(-3,3,n),'cat':'nonconvex','kappa':None}

def make_styblinski(n):
    def f(x): return np.sum(x**4-16*x**2+5*x)/2
    def g(x): return (4*x**3-32*x+5)/2
    rng=np.random.default_rng(42)
    return {'name':f'Styblinski-Tang n={n}','f':f,'g':g,'x0':rng.uniform(-4,4,n),'cat':'nonconvex','kappa':None}


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS DE VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {"BFGS":"#3266ad","L-BFGS(m=10)":"#1d9e75","SR1":"#d4537e",
           "DFP":"#e24b4a","Broyden(φ=0.5)":"#ba7517","Hybrid":"#7f77dd"}

def plot_convergence(results_dict, title="Convergence — ||∇f||/||∇f₀||"):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')
    for name, res in results_dict.items():
        if res and 'grad_norm' in res:
            gn = res['grad_norm']
            color = PALETTE.get(name, '#888')
            ax.semilogy(gn, label=name, color=color, linewidth=2,
                        linestyle='--' if res.get('success')==False else '-')
    ax.axhline(1e-6, color='#94a3b8', linestyle=':', linewidth=1, label='tolérance 1e-6')
    ax.set_xlabel("Itérations", fontsize=11)
    ax.set_ylabel("||∇f||/||∇f₀||", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    return fig


def plot_dolan_more(all_results, metric='nfev', title="Profil de performance (Dolan-Moré)"):
    """
    Profil de Dolan-Moré :
      ρ(τ) = fraction de problèmes où solveur ≤ τ × meilleur solveur
    """
    # Regrouper par problème
    by_prob = defaultdict(dict)
    for (sol, prob), val in all_results.items():
        by_prob[prob][sol] = val

    solvers = sorted(set(s for s,_ in all_results.keys()))
    ratios = defaultdict(list)

    for prob, sols in by_prob.items():
        vals = {s:v for s,v in sols.items() if v != float('inf') and v>0}
        if not vals: continue
        best = min(vals.values())
        for sol in solvers:
            r = vals[sol]/best if sol in vals else float('inf')
            ratios[sol].append(r)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')

    tau_max = 10
    tau_vals = np.linspace(1, tau_max, 500)

    for sol in solvers:
        rs = np.array(ratios[sol])
        n_probs = len(rs)
        if n_probs == 0: continue
        rho = [np.mean(rs <= t) for t in tau_vals]
        color = PALETTE.get(sol, '#888')
        ax.plot(tau_vals, rho, label=sol, color=color, linewidth=2.5)

    ax.set_xlabel("τ  (ratio vs meilleur solveur)", fontsize=11)
    ax.set_ylabel("ρ(τ)  fraction de problèmes résolus", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_xlim(1, tau_max); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, framealpha=0.9, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    return fig


def plot_heatmap(df_results):
    pivot = df_results.pivot_table(
        index='solveur', columns='catégorie', values='succès', aggfunc='mean'
    )
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#f8fafc')
    sns.heatmap(pivot, ax=ax, annot=True, fmt='.0%', cmap='YlGnBu',
                linewidths=0.5, linecolor='#e2e8f0',
                cbar_kws={'shrink':0.8}, vmin=0, vmax=1)
    ax.set_title("Taux de succès — Solveur × Catégorie", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel(""); ax.set_ylabel("")
    plt.tight_layout()
    return fig


def plot_iters_vs_kappa(df_results):
    df_q = df_results[(df_results['catégorie']=='quadratic') & df_results['succès']]
    if df_q.empty: return None
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')
    for sol in df_q['solveur'].unique():
        sub = df_q[df_q['solveur']==sol].groupby('kappa')['iters'].mean()
        color = PALETTE.get(sol, '#888')
        ax.semilogx(sub.index, sub.values, marker='o', label=sol,
                    color=color, linewidth=2, markersize=5)
    ax.set_xlabel("Conditionnement κ", fontsize=11)
    ax.set_ylabel("Itérations moyennes", fontsize=11)
    ax.set_title("Itérations vs Conditionnement (problèmes quadratiques)", fontsize=12, fontweight='bold', pad=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, linestyle=':')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**Solveurs**")
    use_bfgs    = st.checkbox("BFGS",           value=True)
    use_lbfgs   = st.checkbox("L-BFGS (m=10)",  value=True)
    use_sr1     = st.checkbox("SR1",             value=True)
    use_dfp     = st.checkbox("DFP",             value=True)
    use_broyden = st.checkbox("Broyden (φ=0.5)", value=True)
    use_hybrid  = st.checkbox("Hybrid BFGS/SR1", value=True)

    st.markdown("---")
    st.markdown("**Dimension n**")
    n_val = st.select_slider("", options=[10, 50, 100, 200, 500], value=50)

    st.markdown("**Problème test**")
    prob_choice = st.selectbox("", [
        "Rosenbrock", "Quadratic κ=10", "Quadratic κ=100",
        "Quadratic κ=1000", "Quadratic κ=10000",
        "ARWHEAD", "Rastrigin", "Styblinski-Tang"
    ])

    st.markdown("**Paramètres**")
    max_iter = st.slider("Max itérations", 100, 2000, 500, 100)
    tol      = st.select_slider("Tolérance",
                                options=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                                value=1e-6,
                                format_func=lambda x: f"{x:.0e}")

    st.markdown("---")
    st.markdown("**Benchmark complet**")
    st.markdown("**Dimensions pour benchmark**")

    bench_dims = st.multiselect(
        "Choisir dimensions",
        [10, 50, 100, 200, 500],
        default=[50, 100]
    )

    run_bench = st.button(
        "▶ Lancer le benchmark",
        type="primary",
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
  <h1>⚡ Quasi-Newton Benchmark</h1>
  <p>Projet 2 — Comparaison empirique et théorique de variantes quasi-Newton pour problèmes de grande dimension</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SÉLECTION DES SOLVEURS ACTIFS
# ══════════════════════════════════════════════════════════════════════════════

active_solvers = []
if use_bfgs    : active_solvers.append(BFGS())
if use_lbfgs   : active_solvers.append(LBFGS(m=10))
if use_sr1     : active_solvers.append(SR1())
if use_dfp     : active_solvers.append(DFP())
if use_broyden : active_solvers.append(Broyden())
if use_hybrid  : active_solvers.append(HybridBFGSSR1())

if not active_solvers:
    st.warning("Sélectionne au moins un solveur dans la sidebar.")
    st.stop()

# Construire le problème choisi
prob_map = {
    "Rosenbrock"      : make_rosenbrock(n_val if n_val%2==0 else n_val-1),
    "Quadratic κ=10"  : make_quadratic(n_val, 10),
    "Quadratic κ=100" : make_quadratic(n_val, 100),
    "Quadratic κ=1000": make_quadratic(n_val, 1000),
    "Quadratic κ=10000":make_quadratic(n_val, 10000),
    "ARWHEAD"         : make_arwhead(n_val),
    "Rastrigin"       : make_rastrigin(n_val),
    "Styblinski-Tang" : make_styblinski(n_val),
}
prob = prob_map[prob_choice]


# ══════════════════════════════════════════════════════════════════════════════
# ONGLETS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Run unique",
    "📊 Benchmark complet",
    "📈 Profil Dolan-Moré",
    "ℹ️ À propos"
])


# ──────────────────────────────────────────────────────────────────────────────
# ONGLET 1 : Run unique
# ──────────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown(f'<p class="section-title">Problème : {prob["name"]}  —  n = {n_val}</p>',
                unsafe_allow_html=True)

    if st.button("▶ Lancer ce problème", type="primary"):
        results = {}
        prog = st.progress(0)
        status = st.empty()

        for i, solver in enumerate(active_solvers):
            status.text(f"Exécution : {solver.name} ...")
            t0 = time.perf_counter()
            res = solver.optimize(prob['f'], prob['g'], prob['x0'],
                                  max_iter=max_iter, tol=tol)
            res['cpu'] = time.perf_counter() - t0
            res['success'] = res['grad_norm'][-1] < tol * 10
            results[solver.name] = res
            prog.progress((i+1)/len(active_solvers))

        status.empty(); prog.empty()
        st.session_state['single_results'] = results
        st.session_state['single_prob'] = prob['name']

    if 'single_results' in st.session_state:
        results = st.session_state['single_results']

        # Métriques
        cols = st.columns(len(results))
        for col, (name, res) in zip(cols, results.items()):
            color = PALETTE.get(name,'#888')
            ok = "✓" if res['success'] else "✗"
            col.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid {color};">
              <div class="value">{ok} {res['iters']}</div>
              <div class="label">{name}<br>iters · {res['cpu']*1000:.0f}ms</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Tableau détaillé
        rows = []
        for name, res in results.items():
            rows.append({
                'Solveur': name,
                'Statut': '✓ Convergé' if res['success'] else '✗ Échec',
                'Itérations': res['iters'],
                'nfev': res['nfev'],
                '||∇f||/||∇f₀||': f"{res['grad_norm'][-1]:.2e}",
                'f final': f"{res['f_final']:.6e}",
                'Temps (ms)': f"{res['cpu']*1000:.1f}"
            })
        st.dataframe(pd.DataFrame(rows).set_index('Solveur'), use_container_width=True)

        # Courbe de convergence
        fig = plot_convergence(results,
            title=f"Convergence — {st.session_state['single_prob']} (n={n_val})")
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# ONGLET 2 : Benchmark complet
# ──────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown('<p class="section-title">Benchmark automatisé</p>', unsafe_allow_html=True)

    if run_bench:
        if not bench_dims:
            st.warning("Sélectionne au moins une dimension.")
        else:
            problems_bench = []
            for nd in bench_dims:
                nd2 = nd if nd % 2 == 0 else nd - 1
                for k in [10, 100, 1000, 10000]:
                    problems_bench.append(make_quadratic(nd, k))
                problems_bench.append(make_rosenbrock(nd2))
                problems_bench.append(make_arwhead(nd))
                problems_bench.append(make_rastrigin(nd))
                problems_bench.append(make_styblinski(nd))

            total_runs = len(active_solvers) * len(problems_bench)
            prog2  = st.progress(0)
            status2 = st.empty()
            done   = 0

            bench_rows = []
            perf_ratios = {}

            for prob_b in problems_bench:
                prob_vals = {}
                for solver in active_solvers:
                    if solver.name in {'BFGS','DFP','SR1','Broyden(φ=0.5)','Hybrid'} and prob_b['x0'].shape[0] > 500:
                        done += 1; continue
                    status2.text(f"[{done+1}/{total_runs}] {solver.name} × {prob_b['name']}")
                    t0 = time.perf_counter()
                    res = solver.optimize(prob_b['f'], prob_b['g'], prob_b['x0'],
                                         max_iter=500, tol=1e-6)
                    cpu = time.perf_counter() - t0
                    success = res['grad_norm'][-1] < 1e-5
                    bench_rows.append({
                        'solveur': solver.name,
                        'problème': prob_b['name'],
                        'catégorie': prob_b['cat'],
                        'n': len(prob_b['x0']),
                        'kappa': prob_b.get('kappa'),
                        'succès': success,
                        'iters': res['iters'],
                        'nfev': res['nfev'],
                        'cpu': cpu,
                        'grad_norm': res['grad_norm'][-1],
                    })
                    if success:
                        prob_vals[solver.name] = res['nfev']
                    else:
                        prob_vals[solver.name] = float('inf')
                    done += 1
                    prog2.progress(done / total_runs)

                best = min((v for v in prob_vals.values() if v != float('inf')), default=None)
                if best:
                    for sol, val in prob_vals.items():
                        perf_ratios[(sol, prob_b['name'])] = val/best if val!=float('inf') else float('inf')

            status2.empty(); prog2.empty()

            df = pd.DataFrame(bench_rows)
            st.session_state['bench_df'] = df
            st.session_state['bench_ratios'] = perf_ratios
            st.success(f"Benchmark terminé — {len(bench_rows)} runs")

    if 'bench_df' in st.session_state:
        df = st.session_state['bench_df']

        # KPIs
        n_ok = df['succès'].sum(); n_tot = len(df)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Runs total", n_tot)
        c2.metric("Réussis", int(n_ok))
        c3.metric("Taux succès", f"{100*n_ok/n_tot:.1f}%")
        c4.metric("Temps total", f"{df['cpu'].sum():.1f}s")

        # Tableau résumé par solveur
        st.markdown('<p class="section-title">Résumé par solveur</p>', unsafe_allow_html=True)
        summ = df.groupby('solveur').agg(
            Runs=('succès','count'),
            Succès=('succès','sum'),
            Taux=('succès','mean'),
            Iters_moy=('iters','mean'),
            Nfev_moy=('nfev','mean'),
            Temps_moy=('cpu','mean')
        ).reset_index()
        summ['Taux'] = summ['Taux'].map(lambda x: f"{x:.1%}")
        summ['Iters_moy'] = summ['Iters_moy'].map(lambda x: f"{x:.1f}")
        summ['Nfev_moy']  = summ['Nfev_moy'].map(lambda x: f"{x:.1f}")
        summ['Temps_moy'] = summ['Temps_moy'].map(lambda x: f"{x*1000:.1f}ms")
        st.dataframe(summ.set_index('solveur'), use_container_width=True)

        # Heatmap succès
        st.markdown('<p class="section-title">Heatmap — Taux de succès</p>', unsafe_allow_html=True)
        fig_h = plot_heatmap(df)
        st.pyplot(fig_h, use_container_width=True); plt.close()

        # ── Profil Dolan-Moré nfev ────────────────────────────────────────────
        st.markdown('<p class="section-title">Profil de performance Dolan-Moré — nfev</p>',
                    unsafe_allow_html=True)

        if 'bench_ratios' in st.session_state and st.session_state['bench_ratios']:
            fig_dm = plot_dolan_more(
                st.session_state['bench_ratios'],
                metric='nfev',
                title="Profil Dolan-Moré — évaluations gradient (nfev)"
            )
            st.pyplot(fig_dm, use_container_width=True)
            plt.close()

            # Tableau ρ(τ) à τ fixes
            from collections import defaultdict as _dd
            _by_prob = _dd(dict)
            for (sol, prob), val in st.session_state['bench_ratios'].items():
                _by_prob[prob][sol] = val
            _solvers_dm = sorted(set(s for s,_ in st.session_state['bench_ratios'].keys()))
            _ratios_dm  = _dd(list)
            for prob, sols in _by_prob.items():
                _ok = {s:v for s,v in sols.items() if v != float('inf') and v > 0}
                if not _ok: continue
                _best = min(_ok.values())
                for s in _solvers_dm:
                    _ratios_dm[s].append(_ok[s]/_best if s in _ok else float('inf'))

            _tau_points = [1.0, 1.5, 2.0, 5.0, 10.0]
            _table_rows = []
            for s in _solvers_dm:
                r = np.array(_ratios_dm[s])
                row = {'Solveur': s}
                for t in _tau_points:
                    row[f'rho(tau={t})'] = f"{np.mean(r <= t):.1%}"
                _ok_r = r[r != float('inf')]
                row['tau moyen'] = f"{np.mean(_ok_r):.2f}" if len(_ok_r) else "x"
                _table_rows.append(row)

            _table_rows.sort(key=lambda x: float(x['tau moyen'].replace('x','99')))
            st.dataframe(pd.DataFrame(_table_rows).set_index('Solveur'),
                         use_container_width=True)

            st.caption("rho(tau) = fraction de problemes ou le solveur est dans un facteur tau du meilleur. "
                       "Plus tau moyen est petit et rho(1) grand, meilleur est le solveur.")
        else:
            st.info("Les ratios de performance seront calcules apres le benchmark.")

        # ── Iters vs kappa ────────────────────────────────────────────────────
        fig_k = plot_iters_vs_kappa(df)
        if fig_k:
            st.markdown('<p class="section-title">Itérations vs Conditionnement κ</p>',
                        unsafe_allow_html=True)
            st.pyplot(fig_k, use_container_width=True); plt.close()

        # Export JSON
        st.markdown('<p class="section-title">Export</p>', unsafe_allow_html=True)
        json_str = df.to_json(orient='records', indent=2)
        st.download_button("⬇ Télécharger résultats JSON", json_str,
                           "benchmark_results.json", "application/json")
        csv_str = df.to_csv(index=False)
        st.download_button("⬇ Télécharger résultats CSV", csv_str,
                           "benchmark_results.csv", "text/csv")


# ──────────────────────────────────────────────────────────────────────────────
# ONGLET 3 : Profil Dolan-Moré
# ──────────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown('<p class="section-title">Profil de performance de Dolan-Moré (2002)</p>',
                unsafe_allow_html=True)
    st.markdown("""
    Le profil de performance compare les solveurs sur un ensemble de problèmes.
    - **τ** = ratio entre la métrique du solveur et le meilleur solveur sur ce problème
    - **ρ(τ)** = fraction de problèmes où le solveur est dans un facteur τ du meilleur
    - Plus la courbe est **haute et à gauche**, meilleur est le solveur
    """)

    metric_dm = st.radio("Métrique", ["nfev", "iters", "cpu"],
                         horizontal=True,
                         format_func=lambda x: {"nfev":"Évaluations gradient",
                                                "iters":"Itérations",
                                                "cpu":"Temps CPU"}[x])

    if 'bench_ratios' in st.session_state:
        ratios = st.session_state['bench_ratios']
        if metric_dm != 'nfev' and 'bench_df' in st.session_state:
            df2 = st.session_state['bench_df']
            ratios2 = {}
            for prob_name in df2['problème'].unique():
                sub = df2[df2['problème']==prob_name]
                vals = {row['solveur']:row[metric_dm]
                        for _,row in sub.iterrows() if row['succès']}
                if not vals: continue
                best = min(vals.values())
                for sol, val in vals.items():
                    ratios2[(sol, prob_name)] = val/best
                for _,row in sub.iterrows():
                    if not row['succès']:
                        ratios2[(row['solveur'], prob_name)] = float('inf')
            ratios = ratios2

        fig_dm = plot_dolan_more(ratios,
            title=f"Profil Dolan-Moré — métrique : {metric_dm}")
        st.pyplot(fig_dm, use_container_width=True)
        plt.close()
    else:
        st.info("Lance d'abord le benchmark complet (onglet 📊) pour générer les profils.")

        # Profil de démo avec données fictives
        st.markdown("**Aperçu avec données de démonstration :**")
        demo_ratios = {}
        np.random.seed(42)
        demo_solvers = list(PALETTE.keys())
        demo_probs   = [f"prob_{i}" for i in range(20)]
        for prob_d in demo_probs:
            best = 1.0
            for sol in demo_solvers:
                scale = {"BFGS":1.5,"L-BFGS(m=10)":1.2,"SR1":1.3,
                         "DFP":2.5,"Broyden(φ=0.5)":1.6,"Hybrid":1.4}.get(sol,2.0)
                demo_ratios[(sol, prob_d)] = max(1.0, np.random.exponential(scale))
        fig_demo = plot_dolan_more(demo_ratios,
            title="Profil Dolan-Moré — DÉMONSTRATION (données simulées)")
        st.pyplot(fig_demo, use_container_width=True)
        plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# ONGLET 4 : À propos
# ──────────────────────────────────────────────────────────────────────────────

with tab4:
    st.markdown("""
    ## Projet 2 — Quasi-Newton Benchmark

    ### Méthodes implémentées
    | Méthode | Mise à jour | Coût mémoire | Propriété |
    |---|---|---|---|
    | **BFGS** | (I−ρsyᵀ)H(I−ρysᵀ)+ρssᵀ | O(n²) | Gold standard |
    | **L-BFGS** | 2 boucles de Nocedal | O(mn) | Grande dimension |
    | **SR1** | H + vvᵀ/(vᵀy) | O(n²) | Capture courbures indéfinies |
    | **DFP** | H − HyyᵀH/(yᵀHy) + ssᵀ/(yᵀs) | O(n²) | Historique (1959) |
    | **Broyden** | (1−φ)·DFP + φ·BFGS | O(n²) | Famille paramétrée |
    | **Hybrid** | BFGS si yᵀs>0, SR1 sinon | O(n²) | Adaptatif |

    ### Familles de problèmes
    - **Quadratiques** — matrices SPD, κ ∈ {10, 100, 1000, 10000}
    - **Rosenbrock étendu** — classique quasi-Newton, non-convexe
    - **CUTEst** — ARWHEAD, ENGVAL1, TRIDIA, NONDQUAR, DIXMAANE
    - **Non-convexes** — Rastrigin, Styblinski-Tang, Griewank, Schwefel

    ### Référence profils de performance
    > Dolan, E. D., & Moré, J. J. (2002). Benchmarking optimization software
    > with performance profiles. *Mathematical Programming*, 91(2), 201-213.
    """)