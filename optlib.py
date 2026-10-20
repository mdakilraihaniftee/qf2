"""
optlib.py  --  Unconstrained optimization toolkit for Project 2.

Contains
  * the three test problems (value / gradient / Hessian)
  * line searches: Armijo backtracking and strong-Wolfe (bracket + zoom)
  * Gradient Descent, (damped / modified) Newton, and Quasi-Newton (BFGS, DFP)

All solvers share one return type (`Result`) that records the full history
(f, ||g||, step length, optionally the iterates) so that convergence curves can
be plotted afterwards.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.linalg as sla

def _find_data_dir() -> Path:
    """Folder holding fun2_A/b/c.txt: $FUN2_DIR, the upload folder, or the current directory."""
    for d in (os.environ.get("FUN2_DIR", ""), "/mnt/user-data/uploads", "."):
        if d and (Path(d) / "fun2_A.txt").exists():
            return Path(d)
    raise FileNotFoundError("fun2_A.txt not found: put fun2_*.txt next to the code or set FUN2_DIR")


DATA_DIR = _find_data_dir()


# --------------------------------------------------------------------------- #
#  Problems
# --------------------------------------------------------------------------- #
@dataclass
class Problem:
    name: str
    n: int
    fg: callable                  # x -> (f, g);  f = +inf (g = None) outside the domain
    hess: callable                # x -> H
    x0: np.ndarray
    f_star: float | None = None   # known optimal value (if any)
    x_star: np.ndarray | None = None

    def f(self, x):
        return self.fg(x)[0]

    def grad(self, x):
        return self.fg(x)[1]


def make_fun1(n: int = 100, x0=None) -> Problem:
    """f(x) = sum_i i * x_i^2  -- separable convex quadratic, kappa(H) = n."""
    w = np.arange(1, n + 1, dtype=float)

    def fg(x):
        return float(w @ (x * x)), 2.0 * w * x

    def hess(x):
        return np.diag(2.0 * w)

    return Problem("Function 1", n, fg, hess,
                   np.ones(n) if x0 is None else np.asarray(x0, float),
                   f_star=0.0, x_star=np.zeros(n))


def load_fun2_data():
    """Read b, c, A exactly like the MATLAB snippet in ReadMe (column-major fscanf)."""
    A = np.loadtxt(DATA_DIR / "fun2_A.txt").ravel().reshape((500, 100), order="F")
    b = np.loadtxt(DATA_DIR / "fun2_b.txt").ravel()
    c = np.loadtxt(DATA_DIR / "fun2_c.txt").ravel()
    return A, b, c


def make_fun2(x0=None, data=None) -> Problem:
    """f(x) = c'x - sum log(b - Ax).  Domain: b - Ax > 0 (an open polytope)."""
    A, b, c = load_fun2_data() if data is None else data
    n = A.shape[1]

    def fg(x):
        s = b - A @ x
        if np.any(s <= 0.0):
            return np.inf, None
        return float(c @ x - np.sum(np.log(s))), c + A.T @ (1.0 / s)

    def hess(x):
        s = b - A @ x
        return A.T @ ((1.0 / s**2)[:, None] * A)

    return Problem("Function 2", n, fg, hess,
                   np.zeros(n) if x0 is None else np.asarray(x0, float))


def make_fun3(x0=None) -> Problem:
    """Rosenbrock: f = 100 (x2 - x1^2)^2 + (1 - x1)^2,  minimum f = 0 at (1, 1)."""

    def fg(x):
        a = x[1] - x[0] ** 2
        f = 100.0 * a * a + (1.0 - x[0]) ** 2
        g = np.array([-400.0 * x[0] * a - 2.0 * (1.0 - x[0]), 200.0 * a])
        return float(f), g

    def hess(x):
        return np.array([[1200.0 * x[0] ** 2 - 400.0 * x[1] + 2.0, -400.0 * x[0]],
                         [-400.0 * x[0], 200.0]])

    return Problem("Function 3", 2, fg, hess,
                   np.array([-1.2, 1.0]) if x0 is None else np.asarray(x0, float),
                   f_star=0.0, x_star=np.array([1.0, 1.0]))


def check_derivatives(prob: Problem, x, h=1e-6):
    """Compare analytic gradient/Hessian with central differences (course notes, Sec 1)."""
    f0, g = prob.fg(x)
    n = len(x)
    gn = np.zeros(n)
    Hn = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        gn[i] = (prob.f(x + e) - prob.f(x - e)) / (2 * h)
        Hn[:, i] = (prob.grad(x + e) - prob.grad(x - e)) / (2 * h)
    H = prob.hess(x)
    return (np.linalg.norm(g - gn) / max(1.0, np.linalg.norm(g)),
            np.linalg.norm(H - Hn) / max(1.0, np.linalg.norm(H)))


# --------------------------------------------------------------------------- #
#  Result container
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    method: str
    problem: str
    x: np.ndarray
    f: float
    gnorm: float
    iters: int
    nfev: int                      # function evaluations (incl. line search)
    time: float
    converged: bool
    reason: str
    f_hist: list = field(default_factory=list)
    g_hist: list = field(default_factory=list)
    a_hist: list = field(default_factory=list)
    x_hist: list = field(default_factory=list)   # only stored for n == 2


# --------------------------------------------------------------------------- #
#  Line searches
# --------------------------------------------------------------------------- #
FTOL = 1e-15   # *relative* slack on f-comparisons: f is only known to ~1e-16*|f| (fun 2: |f|~2400).
               # Must stay relative: for F1/F3 f -> 0 and an absolute slack would swamp f itself.


def backtracking(prob, x, f0, g0, p, alpha0=1.0, c=1e-4, rho=0.5, max_halvings=400):
    """Armijo backtracking (notes Sec 2.2.2).  Returns (alpha, f, g, nfev) or None."""
    slope = g0 @ p
    slack = FTOL * abs(f0)
    a = alpha0
    for k in range(max_halvings):
        f, g = prob.fg(x + a * p)
        if f <= f0 + c * a * slope + slack:        # inf compares False -> keeps shrinking
            return a, f, g, k + 1
        a *= rho
    return None


def strong_wolfe(prob, x, f0, g0, p, alpha0=1.0, c1=1e-4, c2=0.9,
                 alpha_max=1e6, max_iter=40):
    """Strong-Wolfe search, bracketing + zoom (Nocedal & Wright, Alg. 3.5/3.6).

    Points outside the domain (f = inf) are simply treated as 'too far'.
    Returns (alpha, f, g, nfev) or None.
    """
    d0 = g0 @ p
    slack = FTOL * abs(f0)
    nfev = 0

    def ev(a):
        nonlocal nfev
        nfev += 1
        return prob.fg(x + a * p)

    def zoom(lo, flo, dlo, glo, hi, fhi):
        for _ in range(max_iter):
            d = hi - lo
            if np.isfinite(fhi):                       # safeguarded quadratic interpolation
                denom = 2.0 * (fhi - flo - dlo * d)
                a = lo - dlo * d * d / denom if denom != 0 else lo + 0.5 * d
                lo_b, hi_b = min(lo + 0.1 * d, hi - 0.1 * d), max(lo + 0.1 * d, hi - 0.1 * d)
                if not (lo_b <= a <= hi_b):
                    a = lo + 0.5 * d
            else:                                      # hi is outside the domain: bisect
                a = lo + 0.5 * d
            f, g = ev(a)
            if (not np.isfinite(f)) or f > f0 + c1 * a * d0 + slack or f >= flo + slack:
                hi, fhi = a, f
            else:
                da = g @ p
                if abs(da) <= -c2 * d0:
                    return a, f, g, nfev
                if da * (hi - lo) >= 0:
                    hi, fhi = lo, flo
                lo, flo, dlo, glo = a, f, da, g
            if abs(hi - lo) < 1e-16 * max(1.0, abs(lo)):
                break
        if lo > 0:                                     # best Armijo-satisfying point found
            return lo, flo, glo, nfev
        return None

    a_prev, f_prev, d_prev, g_prev = 0.0, f0, d0, g0
    a = alpha0
    for i in range(max_iter):
        f, g = ev(a)
        if (not np.isfinite(f)) or f > f0 + c1 * a * d0 + slack or (i > 0 and f >= f_prev + slack):
            return zoom(a_prev, f_prev, d_prev, g_prev, a, f)
        d = g @ p
        if abs(d) <= -c2 * d0:
            return a, f, g, nfev
        if d >= 0:
            return zoom(a, f, d, g, a_prev, f_prev)
        a_prev, f_prev, d_prev, g_prev = a, f, d, g
        a = min(2.0 * a, alpha_max)
        if a_prev >= alpha_max:
            break
    return (a_prev, f_prev, g_prev, nfev) if a_prev > 0 else None


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _finish(res_kwargs, hist, t0):
    return Result(time=time.perf_counter() - t0, **res_kwargs, **hist)


def _new_hist(prob, x, f, g):
    h = dict(f_hist=[f], g_hist=[np.linalg.norm(g)], a_hist=[], x_hist=[])
    if prob.n == 2:
        h["x_hist"].append(x.copy())
    return h


def _push(h, prob, x, f, g, a):
    h["f_hist"].append(f)
    h["g_hist"].append(np.linalg.norm(g))
    h["a_hist"].append(a)
    if prob.n == 2:
        h["x_hist"].append(x.copy())


# --------------------------------------------------------------------------- #
#  Gradient descent
# --------------------------------------------------------------------------- #
def gradient_descent(prob, x0=None, c=1e-4, rho=0.5, alpha0=1.0, adaptive_alpha0=False,
                     linesearch="backtracking", c2=0.1, gtol=1e-8, max_iter=200_000):
    """Steepest descent  p = -grad f  with Armijo backtracking.

    adaptive_alpha0=True uses the notes' heuristic
        alpha0_k = alpha_{k-1} * (p_{k-1}' g_{k-1}) / (p_k' g_k)
    (first-order change equal to the previous step's); otherwise alpha0 is fixed.
    linesearch = 'backtracking' (Armijo, notes) or 'wolfe' (strong Wolfe with a small c2,
    i.e. an inexact-but-accurate line search; uses gradient information).
    """
    t0 = time.perf_counter()
    x = (prob.x0 if x0 is None else np.asarray(x0, float)).copy()
    f, g = prob.fg(x)
    h = _new_hist(prob, x, f, g)
    nfev, a_prev, slope_prev, reason, ok = 1, None, None, "max_iter", False
    k = 0
    for k in range(1, max_iter + 1):
        gn = np.linalg.norm(g, np.inf)
        if gn <= gtol:
            reason, ok = "gtol", True
            k -= 1
            break
        p = -g
        slope = g @ p
        a0 = alpha0
        if adaptive_alpha0 and a_prev is not None:
            a0 = a_prev * slope_prev / slope
        if linesearch == "wolfe":
            ls = strong_wolfe(prob, x, f, g, p, alpha0=a0, c1=c, c2=c2)
        else:
            ls = backtracking(prob, x, f, g, p, alpha0=a0, c=c, rho=rho)
        if ls is None:
            reason = "line search failed"
            k -= 1
            break
        a, f, g, ne = ls
        nfev += ne
        x = x + a * p
        a_prev, slope_prev = a, slope
        _push(h, prob, x, f, g, a)
    return _finish(dict(method="Gradient Descent" + (" (Wolfe)" if linesearch == "wolfe" else ""),
                        problem=prob.name, x=x, f=f,
                        gnorm=float(np.linalg.norm(g)), iters=k, nfev=nfev,
                        converged=ok, reason=reason), h, t0)


# --------------------------------------------------------------------------- #
#  Newton
# --------------------------------------------------------------------------- #
def _newton_direction(H, g, beta=1e-3):
    """Solve H p = -g.  If H is not safely positive definite, add tau*I
    (Nocedal & Wright Alg. 3.3, i.e. a Levenberg-Marquardt style modification)."""
    n = len(g)
    diag_min = H.diagonal().min()
    tau = 0.0 if diag_min > 0 else beta - diag_min
    for _ in range(60):
        try:
            L = sla.cho_factor(H + tau * np.eye(n) if tau > 0 else H, lower=True)
            return sla.cho_solve(L, -g), tau
        except (np.linalg.LinAlgError, sla.LinAlgError):
            tau = max(2.0 * tau, beta)
    return -g, np.inf                                   # fall back to steepest descent


def newton(prob, x0=None, c=1e-4, rho=0.5, alpha0=1.0, gtol=1e-8, max_iter=500):
    """Newton's method with Armijo backtracking (alpha0 = 1) and a modified-Hessian
    safeguard so the direction is always a descent direction."""
    t0 = time.perf_counter()
    x = (prob.x0 if x0 is None else np.asarray(x0, float)).copy()
    f, g = prob.fg(x)
    h = _new_hist(prob, x, f, g)
    nfev, n_mod, reason, ok = 1, 0, "max_iter", False
    k = 0
    for k in range(1, max_iter + 1):
        if np.linalg.norm(g, np.inf) <= gtol:
            reason, ok = "gtol", True
            k -= 1
            break
        p, tau = _newton_direction(prob.hess(x), g)
        n_mod += tau > 0
        if g @ p >= 0:                                   # safety net
            p = -g
        ls = backtracking(prob, x, f, g, p, alpha0=alpha0, c=c, rho=rho)
        if ls is None:
            reason = "line search failed"
            k -= 1
            break
        a, f, g, ne = ls
        nfev += ne
        x = x + a * p
        _push(h, prob, x, f, g, a)
    r = _finish(dict(method="Newton", problem=prob.name, x=x, f=f,
                     gnorm=float(np.linalg.norm(g)), iters=k, nfev=nfev,
                     converged=ok, reason=reason), h, t0)
    r.n_modified = int(n_mod)
    return r


# --------------------------------------------------------------------------- #
#  Quasi-Newton (BFGS / DFP) -- inverse-Hessian approximation C
# --------------------------------------------------------------------------- #
def quasi_newton(prob, x0=None, update="bfgs", linesearch="wolfe", c=1e-4, c2=0.9, rho=0.5,
                 gtol=1e-8, max_iter=20_000):
    """BFGS / DFP with an inverse-Hessian approximation C ~ H^-1.

        BFGS: C+ = (I - r s y') C (I - r y s') + r s s',      r = 1/(y's)
        DFP : C+ = C - C y y' C /(y' C y) + s s' / (y' s)

    linesearch = 'wolfe'        -> strong Wolfe (keeps y's > 0 => C stays p.d.)
                 'backtracking' -> Armijo backtracking (update skipped if y's <= 0)
    alpha0 = 1 (as recommended for Newton-like methods).  The first step uses
    alpha0 = 1/||g|| and C is rescaled after the first step by y's / y'y.
    """
    t0 = time.perf_counter()
    n = prob.n
    x = (prob.x0 if x0 is None else np.asarray(x0, float)).copy()
    f, g = prob.fg(x)
    h = _new_hist(prob, x, f, g)
    C = np.eye(n)
    nfev, n_skip, n_reset, reason, ok = 1, 0, 0, "max_iter", False
    k = 0
    for k in range(1, max_iter + 1):
        if np.linalg.norm(g, np.inf) <= gtol:
            reason, ok = "gtol", True
            k -= 1
            break
        p = -C @ g
        if g @ p >= 0:                                   # lost descent -> reset
            C = np.eye(n)
            p = -g
            n_reset += 1
        a0 = min(1.0, 1.0 / np.linalg.norm(g)) if k == 1 else 1.0
        ls = (strong_wolfe(prob, x, f, g, p, alpha0=a0, c1=c, c2=c2)
              if linesearch == "wolfe"
              else backtracking(prob, x, f, g, p, alpha0=a0, c=c, rho=rho))
        if ls is None:
            if np.allclose(C, np.eye(n)):                # even steepest descent fails
                reason = "line search failed"
                k -= 1
                break
            C = np.eye(n)                                # retry once with steepest descent
            n_reset += 1
            p = -g
            ls = (strong_wolfe(prob, x, f, g, p, alpha0=min(1.0, 1.0 / np.linalg.norm(g)), c1=c, c2=c2)
                  if linesearch == "wolfe"
                  else backtracking(prob, x, f, g, p, alpha0=1.0, c=c, rho=rho))
            if ls is None:
                reason = "line search failed"
                k -= 1
                break
        a, f_new, g_new, ne = ls
        nfev += ne
        s = a * p
        y = g_new - g
        x = x + s
        f, g_old, g = f_new, g, g_new
        _push(h, prob, x, f, g, a)

        sy = s @ y
        if sy <= 1e-12 * np.linalg.norm(s) * np.linalg.norm(y):
            n_skip += 1                                  # curvature condition violated
            continue
        if k == 1:                                       # initial scaling (N&W eq. 6.20)
            C = (sy / (y @ y)) * np.eye(n)
        if update == "bfgs":
            r = 1.0 / sy
            Cy = C @ y
            # expanded form of (I - r s y')C(I - r y s') + r s s'  (O(n^2))
            C = C - r * (np.outer(s, Cy) + np.outer(Cy, s)) + (r * r * (y @ Cy) + r) * np.outer(s, s)
        elif update == "dfp":
            Cy = C @ y
            C = C - np.outer(Cy, Cy) / (y @ Cy) + np.outer(s, s) / sy
        else:
            raise ValueError(update)
    name = {"bfgs": "BFGS", "dfp": "DFP"}[update]
    r = _finish(dict(method=name, problem=prob.name, x=x, f=f,
                     gnorm=float(np.linalg.norm(g)), iters=k, nfev=nfev,
                     converged=ok, reason=reason), h, t0)
    r.n_skipped, r.n_reset = n_skip, n_reset
    return r
