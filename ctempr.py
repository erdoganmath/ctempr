"""
Reproducible implementation of Affine (ATEMPR) and Conic (CTEMPR)
Transformational Enhanced Multivariance Products Representation,
constant-level truncation.

Formulation (see Section 4 of the paper):
  g = a0 + a1 f + a2 f^2,   a_k polynomials of degree m (in a normalized variable)
  constant EMPR component  g0 = <s^2, g> / <s^2, s^2>
  constancy measurer       sigma0 = ||g0 s^2||^2 / ||g||^2
  maximise sigma0  <=>  minimise ||g - g0 s^2||^2 subject to ||a1 f + a2 f^2|| = 1,
  which is a generalized symmetric eigenvalue problem (smallest eigenpair).
  Approximant: root of a2 F^2 + a1 F + (a0 - g0 s^2) = 0, branch chosen by
  eps = sign(2 a2 f + a1) at the quadrature nodes (Section 4.4).

All integrals are evaluated with Gauss-Legendre quadrature.

Usage:  python ctempr.py
Output: figures/*.pdf and results.json next to this script.
Requirements: numpy, scipy, matplotlib.
"""
import json, math, os
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures") + os.sep
os.makedirs(FIG, exist_ok=True)
NQ = 400


def quad(a, b, n=NQ):
    t, w = leggauss(n)
    x = 0.5 * (b - a) * t + 0.5 * (b + a)
    return x, 0.5 * (b - a) * w


def poly_basis(t, m):
    """Shifted Legendre polynomials P_i(2t-1), t in [0,1]; spans the same space as 1,x,...,x^m."""
    return np.stack([np.polynomial.legendre.Legendre.basis(i)(2 * t - 1) for i in range(m + 1)], axis=-1)


def solve_constancy(V, Phi1, mu, rtol=1e-12):
    """Minimise ||(I-Pi_V) Phi1 c|| / ||Phi1 c|| in the weighted L2 norm, computed with
    orthogonal factorizations (no normal equations).  Returns c1, the coefficients of V
    that represent Pi_V Phi1 c1, and the minimal ratio squared."""
    r = np.sqrt(mu)[:, None]
    Qv, _ = np.linalg.qr(r * V)
    B = r * Phi1
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    keep = S > rtol * S[0]
    U, S, Vt = U[:, keep], S[keep], Vt[keep]       # orthonormal basis of range(Phi1)
    M = U - Qv @ (Qv.T @ U)                          # project out V
    _, sig, Wt = np.linalg.svd(M, full_matrices=False)
    y = Wt[-1]                                        # unit vector in range(Phi1)
    c1 = Vt.T @ (y / S)
    coef = np.linalg.lstsq(r * V, B @ c1, rcond=None)[0]
    return c1, coef, sig[-1] ** 2


class TEMPR:
    """Constant-level transformational EMPR. order=1 -> affine, order=2 -> conic."""

    def __init__(self, f, a, b, order=2, m=2, W=None, s=None):
        self.f, self.a, self.b, self.order, self.m = f, a, b, order, m
        x, w = quad(a, b)
        Wx = np.ones_like(x) if W is None else W(x)
        Wx = Wx / np.sum(w * Wx)                    # normalized weight: int W = 1
        sx = np.ones_like(x) if s is None else s(x)
        self.s_scale = 1.0 / np.sqrt(np.sum(w * Wx * sx ** 2))   # int W s^2 = 1
        self.W, self.s = W, s
        sx = sx * self.s_scale
        self.x, self.mu = x, w * Wx                 # quadrature measure incl. weight
        fx = f(x)
        P = poly_basis(self._t(x), m)
        blocks = [P * fx[:, None] ** k for k in range(order + 1)]
        self.Phi = np.hstack(blocks)                # columns: x^i f^k
        S2 = sx ** 2
        ip = lambda A, B: A.T @ (self.mu[:, None] * B) if A.ndim > 1 else A @ (self.mu * B)
        # project out span{s^2} U {poly part} from the f-dependent columns
        V = np.hstack([S2[:, None], P])
        n0 = m + 1
        Phi1 = self.Phi[:, n0:]
        c1, full, res = solve_constancy(V, Phi1, self.mu)
        # optimal polynomial part a0: least-squares fit of -a1 f - a2 f^2 orthogonal to s^2
        c0 = -full[1:]
        self.c = np.concatenate([c0, c1])
        g = self.Phi @ self.c
        self.g0 = ip(S2, g) / ip(S2, S2)
        self.sigma0 = (self.g0 ** 2 * ip(S2, S2)) / ip(g, g)
        self.residual = res
        # branch indicator eps(x) = sign(2 a2 f + a1) at the quadrature nodes; the two roots
        # can exchange roles where the discriminant has a double zero.  For evaluation
        # eps is extended as a piecewise-constant function (nearest node).
        if order == 2:
            a0_, a1_, a2_ = self.coeff_funcs(x)
            self.eps_nodes = np.sign(2 * a2_ * fx + a1_)
            self.eps_nodes[self.eps_nodes == 0] = 1
            self.nswitch = int(np.sum(np.diff(self.eps_nodes) != 0))

    def _t(self, x):
        return (x - self.a) / (self.b - self.a)

    def coeff_funcs(self, x):
        P = poly_basis(self._t(np.asarray(x, float)), self.m)
        n = self.m + 1
        return [P @ self.c[k * n:(k + 1) * n] for k in range(self.order + 1)]

    def s2(self, x):
        sx = np.ones_like(x) if self.s is None else self.s(x)
        return (sx * self.s_scale) ** 2

    def __call__(self, x):
        x = np.asarray(x, float)
        a = self.coeff_funcs(x)
        rhs = self.g0 * self.s2(x)
        if self.order == 1:
            return (rhs - a[0]) / a[1]
        a0, a1, a2 = a
        C = a0 - rhs
        disc = np.clip(a1 ** 2 - 4 * a2 * C, 0.0, None)   # clip tiny negative values
        idx = np.clip(np.searchsorted(self.x, x), 0, len(self.x) - 1)
        idx2 = np.clip(idx - 1, 0, len(self.x) - 1)
        near = np.where(np.abs(self.x[idx2] - x) < np.abs(self.x[idx] - x), idx2, idx)
        eps = self.eps_nodes[near]
        sg = np.where(a1 >= 0, 1.0, -1.0)
        q = -0.5 * (a1 + sg * np.sqrt(disc))               # cancellation-free roots
        with np.errstate(divide="ignore", invalid="ignore"):
            r_plus = C / q                                 # = (-a1 + sg*sqrt)/(2 a2)
            r_minus = q / a2                               # = (-a1 - sg*sqrt)/(2 a2)
        return np.where(eps == sg, r_plus, r_minus)


def l2err(F, f, a, b):
    x, w = quad(a, b, 2000)
    return float(np.sum(w * (F(x) - f(x)) ** 2))


def taylor_cos(deg):
    def T(x):
        out = np.zeros_like(x, dtype=float)
        for k in range(0, deg + 1, 2):
            out += (-1) ** (k // 2) * x ** k / math.factorial(k)
        return out
    return T


results = {}

# ---------- Example 1-3: sin x, exp x, x^2 on [0.1,1] ----------
sq = lambda c, p=0.5: (lambda x: np.abs(x - c) ** p)
ex13 = [("sin", np.sin, sq(0.1, 0.5)), ("exp", np.exp, sq(0.1, 0.5)), ("sq", lambda x: x ** 2, sq(0.1, 1.5))]
fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
axes = list(axes) + [None]
for ax, (name, f, W) in zip(axes, ex13):
    A = TEMPR(f, 0.1, 1, order=1, W=W)
    C = TEMPR(f, 0.1, 1, order=2, W=W)
    results[f"ex_{name}"] = dict(aff_err=l2err(A, f, 0.1, 1), con_err=l2err(C, f, 0.1, 1),
                                 aff_sigma=float(A.sigma0), con_sigma=float(C.sigma0))
    if ax is None:
        continue
    xx = np.linspace(0.1, 1, 400)
    ax.semilogy(xx, np.abs(A(xx) - f(xx)) + 1e-18, "--", label="ATEMPR")
    ax.semilogy(xx, np.abs(C(xx) - f(xx)) + 1e-18, "-", label="CTEMPR")
    ax.set_title({"sin": r"$f(x)=\sin x$", "exp": r"$f(x)=e^x$", "sq": r"$f(x)=x^2$"}[name])
    ax.set_xlabel("$x$")
axes[0].set_ylabel("absolute error")
axes[0].legend()
fig.tight_layout(); fig.savefig(FIG + "fig_ex123_errors.pdf"); plt.close(fig)

# ---------- Example 4: sin(pi x), affine & conic, growing intervals ----------
f4 = lambda x: np.sin(np.pi * x)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
results["ex4"] = {}
for ax, (lab, b) in zip(axes, [("[0,1]", 1.0), ("[0,pi]", np.pi), ("[0,2pi]", 2 * np.pi)]):
    A = TEMPR(f4, 0, b, order=1)
    C = TEMPR(f4, 0, b, order=2)
    results["ex4"][lab] = dict(aff_err=l2err(A, f4, 0, b), con_err=l2err(C, f4, 0, b),
                               aff_sigma=float(A.sigma0), con_sigma=float(C.sigma0))
    xx = np.linspace(0, b, 800)
    ax.plot(xx, f4(xx), "k-", lw=1.4, label=r"$\sin(\pi x)$")
    Ay = A(xx); Ay[np.abs(Ay) > 3] = np.nan
    Cy = C(xx); Cy[np.abs(Cy) > 3] = np.nan
    ax.plot(xx, Ay, "--", label="ATEMPR")
    ax.plot(xx, Cy, ":", lw=2, label="CTEMPR")
    ax.set_ylim(-2, 2); ax.set_xlabel("$x$")
    ax.set_title({"[0,1]": "$[0,1]$", "[0,pi]": r"$[0,\pi]$", "[0,2pi]": r"$[0,2\pi]$"}[lab])
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG + "fig_ex4_sinpi.pdf"); plt.close(fig)

# ---------- Example 5: log(1+x), W=1, s=sqrt(x) ----------
f5 = lambda x: np.log1p(x)
s5 = lambda x: np.sqrt(x)
results["ex5"] = {}
models5 = {}
for lab, (a, b) in {"[0.1,0.5]": (0.1, 0.5), "[0.1,1]": (0.1, 1.0), "[0.1,2pi]": (0.1, 2 * np.pi)}.items():
    A = TEMPR(f5, a, b, order=1, s=s5)
    C = TEMPR(f5, a, b, order=2, s=s5)
    models5[lab] = (A, C)
    results["ex5"][lab] = dict(aff_err=l2err(A, f5, a, b), con_err=l2err(C, f5, a, b),
                               aff_sigma=float(A.sigma0), con_sigma=float(C.sigma0))
A, C = models5["[0.1,0.5]"]
pts = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
results["ex5_table"] = [dict(x=float(p), f=float(f5(p)), aff=float(A(p)), con=float(C(p)),
                             aff_abs=float(abs(A(p) - f5(p))), con_abs=float(abs(C(p) - f5(p)))) for p in pts]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
for ax, lab in zip(axes, ["[0.1,0.5]", "[0.1,1]", "[0.1,2pi]"]):
    A, C = models5[lab]
    xx = np.linspace(A.a, A.b, 600)
    ax.semilogy(xx, np.abs(A(xx) - f5(xx)) + 1e-18, "--", label="ATEMPR")
    ax.semilogy(xx, np.abs(C(xx) - f5(xx)) + 1e-18, "-", label="CTEMPR")
    ax.set_xlabel("$x$")
    ax.set_title({"[0.1,0.5]": "$[0.1,0.5]$", "[0.1,1]": "$[0.1,1]$", "[0.1,2pi]": r"$[0.1,2\pi]$"}[lab])
axes[0].set_ylabel("absolute error"); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG + "fig_ex5_log.pdf"); plt.close(fig)

# ---------- Example 6: cos x vs Taylor ----------
f6 = np.cos
results["ex6"] = {}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, (lab, a, b) in zip(axes, [("[0,1]", 0.0, 1.0), ("[1,pi]", 1.0, np.pi)]):
    W = sq(a, 0.5)
    s = lambda x, a=a, b=b: np.sqrt(2) * np.sin(np.pi * (x - a) / (b - a))   # rescaled internally
    C = TEMPR(f6, a, b, order=2, W=W, s=s)
    A = TEMPR(f6, a, b, order=1, W=W, s=s)
    T4, T8 = taylor_cos(4), taylor_cos(8)
    results["ex6"][lab] = dict(con_err=l2err(C, f6, a, b), aff_err=l2err(A, f6, a, b),
                               taylor4_err=l2err(T4, f6, a, b), taylor8_err=l2err(T8, f6, a, b),
                               con_sigma=float(C.sigma0))
    xx = np.linspace(a, b, 600)
    ax.plot(xx, f6(xx), "k-", lw=1.4, label=r"$\cos x$")
    ax.plot(xx, C(xx), "--", lw=1.6, label="CTEMPR")
    ax.plot(xx, T4(xx), ":", lw=1.6, label="Taylor (deg. 4)")
    ax.set_xlabel("$x$"); ax.set_title("$[0,1]$" if lab == "[0,1]" else r"$[1,\pi]$")
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG + "fig_ex6_cos.pdf"); plt.close(fig)


# ---------- Example 7: bivariate e^{x+y} on [0,1]^2 ----------
def tempr2d(f, order=2, n=60):
    t, w = leggauss(n)
    t = 0.5 * (t + 1); w = 0.5 * w
    X, Y = np.meshgrid(t, t, indexing="ij")
    Wq = np.outer(w, w).ravel()
    x, y = X.ravel(), Y.ravel()
    fx = f(x, y)
    P = np.stack([np.ones_like(x), x, y, x * y], 1)
    Phi = np.hstack([P * fx[:, None] ** k for k in range(order + 1)])
    ip = lambda A, B: A.T @ (Wq[:, None] * B)
    S2 = np.ones_like(x)
    V = np.hstack([S2[:, None], P])
    Phi1 = Phi[:, 4:]
    c1, full, _ = solve_constancy(V, Phi1, Wq)
    c0 = -full[1:]
    c = np.concatenate([c0, c1])
    g = Phi @ c
    g0 = (Wq @ g) / Wq.sum()
    sigma = g0 ** 2 * Wq.sum() / (Wq @ g ** 2)

    a_nodes = [P @ c[4 * k:4 * k + 4] for k in range(order + 1)]
    eps_nodes = np.sign(2 * a_nodes[-1] * fx + a_nodes[1]) if order == 2 else None
    tree = cKDTree(np.stack([x, y], 1))

    def F(xx, yy):
        xx, yy = np.asarray(xx, float), np.asarray(yy, float)
        Pp = np.stack([np.ones_like(xx), xx, yy, xx * yy], -1)
        a = [Pp @ c[4 * k:4 * k + 4] for k in range(order + 1)]
        if order == 1:
            return (g0 - a[0]) / a[1]
        Cc = a[0] - g0
        disc = np.clip(a[1] ** 2 - 4 * a[2] * Cc, 0, None)
        sg = np.where(a[1] >= 0, 1.0, -1.0)
        q = -0.5 * (a[1] + sg * np.sqrt(disc))
        eps = eps_nodes[tree.query(np.stack([xx.ravel(), yy.ravel()], 1))[1]].reshape(xx.shape)
        return np.where(eps == sg, Cc / q, q / a[2])
    nsw = 0 if order == 1 else int(len(np.unique(eps_nodes)) - 1)
    return F, float(sigma), c, nsw


f7 = lambda x, y: np.exp(x + y)
C7, sig7, c7, nsw7 = tempr2d(f7, 2)
A7, sigA7, _, _ = tempr2d(f7, 1)
T7 = lambda x, y: sum((x + y) ** k / math.factorial(k) for k in range(6))
t, w = leggauss(200); t = 0.5 * (t + 1); w = 0.5 * w
X, Y = np.meshgrid(t, t, indexing="ij"); WW = np.outer(w, w)
results["ex7"] = dict(con_err=float(np.sum(WW * (C7(X, Y) - f7(X, Y)) ** 2)),
                      aff_err=float(np.sum(WW * (A7(X, Y) - f7(X, Y)) ** 2)),
                      taylor5_err=float(np.sum(WW * (T7(X, Y) - f7(X, Y)) ** 2)),
                      con_sigma=sig7, branch_switch=nsw7, f11=float(f7(1., 1.)), con11=float(C7(np.array(1.), np.array(1.))),
                      aff11=float(A7(np.array(1.), np.array(1.))), tay11=float(T7(1., 1.)),
                      con_maxerr=float(np.max(np.abs(C7(X, Y) - f7(X, Y)))),
                      tay_maxerr=float(np.max(np.abs(T7(X, Y) - f7(X, Y)))))
g = np.linspace(0, 1, 80); GX, GY = np.meshgrid(g, g)
fig = plt.figure(figsize=(9.5, 3.6))
for i, (Fn, ttl) in enumerate([(C7, "CTEMPR"), (T7, "Taylor (total degree 5)")]):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    E = np.abs(Fn(GX, GY) - f7(GX, GY))
    ax.plot_surface(GX, GY, E, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_title(ttl + " absolute error", fontsize=9)
    ax.ticklabel_format(axis="z", style="sci", scilimits=(0, 0))
fig.tight_layout(); fig.savefig(FIG + "fig_ex7_errors.pdf"); plt.close(fig)

# ---------- sigma0 vs polynomial degree (log(1+x) on [0.1,1]) ----------
deg = list(range(0, 5))
rows = []
for m in deg:
    A = TEMPR(f5, 0.1, 1, order=1, m=m, s=s5)
    C = TEMPR(f5, 0.1, 1, order=2, m=m, s=s5)
    rows.append(dict(m=m, aff_err=l2err(A, f5, 0.1, 1), con_err=l2err(C, f5, 0.1, 1),
                     aff_1ms=float(1 - A.sigma0), con_1ms=float(1 - C.sigma0)))
results["degree_study"] = rows
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.semilogy(deg, [r["aff_err"] for r in rows], "o--", label="ATEMPR")
ax.semilogy(deg, [r["con_err"] for r in rows], "s-", label="CTEMPR")
ax.set_xlabel("degree $m$ of coefficient polynomials"); ax.set_ylabel(r"$\|f-\tilde f\|_2^2$")
ax.set_xticks(deg); ax.legend()
fig.tight_layout(); fig.savefig(FIG + "fig_degree.pdf"); plt.close(fig)

with open(os.path.join(HERE, "results.json"), "w") as fh:
    json.dump(results, fh, indent=1)
print(json.dumps(results, indent=1))
