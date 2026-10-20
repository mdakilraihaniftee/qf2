"""
run_experiments.py -- reproduces every table / figure of the Project 2 report.

    python run_experiments.py            # writes ./results/{tables.md, *.csv, *.png, solutions/*}

Requires optlib.py in the same folder and the fun2_*.txt files (see DATA_DIR in optlib.py).
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optlib import (check_derivatives, gradient_descent, make_fun1, make_fun2, make_fun3,
                    newton, quasi_newton, load_fun2_data)

OUT = Path("results")
(OUT / "solutions").mkdir(parents=True, exist_ok=True)
GTOL = 1e-10
MD: list[str] = []          # accumulated markdown tables


def md(s=""):
    MD.append(s)
    print(s)


def table(headers, rows):
    md("| " + " | ".join(headers) + " |")
    md("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        md("| " + " | ".join(str(c) for c in r) + " |")
    md()


def sci(v, d=2):
    return f"{v:.{d}e}"


# --------------------------------------------------------------------------- #
#  Solver configurations used in the main comparison
# --------------------------------------------------------------------------- #
CONFIGS = {
    "GD (backtracking)": lambda p, x0=None, gtol=GTOL, mi=400_000: gradient_descent(
        p, x0, c=0.1, rho=0.5, adaptive_alpha0=True, gtol=gtol, max_iter=mi),
    "GD (Wolfe)": lambda p, x0=None, gtol=GTOL, mi=400_000: gradient_descent(
        p, x0, c=1e-4, c2=0.1, adaptive_alpha0=True, linesearch="wolfe", gtol=gtol, max_iter=mi),
    "Newton": lambda p, x0=None, gtol=GTOL, mi=500: newton(p, x0, c=0.1, rho=0.5, gtol=gtol, max_iter=mi),
    "BFGS (Wolfe)": lambda p, x0=None, gtol=GTOL, mi=20_000: quasi_newton(
        p, x0, update="bfgs", linesearch="wolfe", c=1e-4, c2=0.9, gtol=gtol, max_iter=mi),
    "BFGS (backtracking)": lambda p, x0=None, gtol=GTOL, mi=20_000: quasi_newton(
        p, x0, update="bfgs", linesearch="backtracking", c=0.1, rho=0.5, gtol=gtol, max_iter=mi),
    "DFP (Wolfe, c2=0.1)": lambda p, x0=None, gtol=GTOL, mi=20_000: quasi_newton(
        p, x0, update="dfp", linesearch="wolfe", c=1e-4, c2=0.1, gtol=gtol, max_iter=mi),
}


def reference_solution(prob):
    """High-accuracy reference: Newton to ||g||_inf <= 1e-12 (problem 2 is strictly convex
    => unique global minimiser)."""
    if prob.x_star is not None:
        return prob.x_star, prob.f_star
    r = newton(prob, gtol=1e-12)
    return r.x, r.f


# --------------------------------------------------------------------------- #
#  0. derivative check
# --------------------------------------------------------------------------- #
def exp_derivative_check(problems):
    md("## 0. Analytic derivative check (relative error vs. central differences, h = 1e-6)\n")
    rng = np.random.default_rng(0)
    rows = []
    for p in problems:
        x = p.x0 + 0.01 * rng.standard_normal(p.n)
        eg, eh = check_derivatives(p, x)
        rows.append([p.name, sci(eg), sci(eh)])
    table(["Function", "gradient rel. err", "Hessian rel. err"], rows)


# --------------------------------------------------------------------------- #
#  1. main comparison
# --------------------------------------------------------------------------- #
def exp_main(problems, refs):
    md("## 1. Solutions of every method on every function\n")
    md(f"Stopping rule: ||grad f||_inf <= {GTOL:g}.  Start points: F1 x0 = ones(100); "
       "F2 x0 = 0 (strictly feasible); F3 x0 = (-1.2, 1).\n")
    results = {}
    for p in problems:
        xs, fs = refs[p.name]
        md(f"### {p.name}   (reference f* = {fs:.12g})\n")
        rows = []
        results[p.name] = {}
        for name, run in CONFIGS.items():
            r = run(p)
            results[p.name][name] = r
            rows.append([name, f"{r.f:.12e}", sci(abs(r.f - fs)), sci(np.linalg.norm(r.x - xs)),
                         sci(r.gnorm), r.iters, r.nfev, f"{r.time:.3f}",
                         "yes" if r.converged else f"NO ({r.reason})"])
        table(["Method", "f(x_final)", "|f - f*|", "||x - x*||", "||grad||", "iters", "f-evals",
               "time [s]", "converged"], rows)
    return results


# --------------------------------------------------------------------------- #
#  2. convergence curves
# --------------------------------------------------------------------------- #
STY = {"GD (backtracking)": ("tab:blue", "-"), "GD (Wolfe)": ("tab:cyan", "-"),
       "Newton": ("tab:red", "-"), "BFGS (Wolfe)": ("tab:green", "-"),
       "BFGS (backtracking)": ("tab:olive", "--"), "DFP (Wolfe, c2=0.1)": ("tab:purple", "-")}


def fig_convergence(results, refs, problems):
    for p in problems:
        xs, fs = refs[p.name]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
        for name, r in results[p.name].items():
            col, ls = STY[name]
            floor = max(1e-18, 2.2e-16 * abs(fs))      # f is only resolved to ~eps*|f*|
            err = np.maximum(np.array(r.f_hist) - fs, floor)
            ax[0].semilogy(err, color=col, ls=ls, label=name, lw=1.6)
            ax[1].semilogy(np.maximum(r.g_hist, 1e-18), color=col, ls=ls, label=name, lw=1.6)
        for a, yl in zip(ax, [r"$f(x_k)-f^*$", r"$\|\nabla f(x_k)\|_2$"]):
            a.set_xlabel("iteration")
            a.set_ylabel(yl)
            a.grid(alpha=.3, which="both")
        if p.name == "Function 1":
            ax[0].set_xscale("symlog", linthresh=10)
            ax[1].set_xscale("symlog", linthresh=10)
        else:
            ax[0].set_xscale("symlog", linthresh=10)
            ax[1].set_xscale("symlog", linthresh=10)
        ax[0].legend(fontsize=8)
        fig.suptitle(f"{p.name}: convergence of the algorithms (x-axis linear up to 10, log beyond; "
                     "f - f* plotted down to its floating-point resolution)")
        fig.tight_layout()
        fig.savefig(OUT / f"fig_convergence_{p.name.split()[-1]}.png", dpi=150)
        plt.close(fig)


def fig_newton_phases(results):
    """Newton on F2/F3: step length and gradient norm => damped phase then quadratic phase."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, col in (("Function 2", "tab:red"), ("Function 3", "tab:orange")):
        r = results[name]["Newton"]
        ax[0].semilogy(np.maximum(r.g_hist, 1e-18), "o-", color=col, label=name, ms=4)
        ax[1].plot(np.arange(1, len(r.a_hist) + 1), r.a_hist, "o-", color=col, label=name, ms=4)
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel(r"$\|\nabla f\|_2$"); ax[0].grid(alpha=.3)
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel(r"accepted step length $\alpha_k$"); ax[1].grid(alpha=.3)
    ax[0].legend(); ax[1].legend()
    ax[0].set_title("Newton: gradient norm (quadratic phase = digits double)")
    ax[1].set_title("Newton: backtracking tames the first steps, then alpha = 1")
    fig.tight_layout()
    fig.savefig(OUT / "fig_newton_phases.png", dpi=150)
    plt.close(fig)


def fig_rosenbrock_paths(results):
    p = make_fun3()
    X, Y = np.meshgrid(np.linspace(-2, 2, 500), np.linspace(-1.5, 3, 500))
    Z = 100 * (Y - X**2) ** 2 + (1 - X) ** 2
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a, xlim, ylim, title in ((ax[0], (-2, 2), (-1.5, 3), "full view"),
                                 (ax[1], (0.2, 1.1), (0.0, 1.25), "zoom near the solution")):
        a.contour(X, Y, Z, levels=np.logspace(-2, 3.5, 28), cmap="Blues_r", linewidths=.9, alpha=.8)
        for name in ("GD (backtracking)", "GD (Wolfe)", "Newton", "BFGS (Wolfe)"):
            xh = np.array(results["Function 3"][name].x_hist)
            col, _ = STY[name]
            if name.startswith("GD (back"):
                xh = xh[:: max(1, len(xh) // 400)]    # thin the 23k-point path for plotting
            a.plot(xh[:, 0], xh[:, 1], "-o" if len(xh) < 60 else "-", color=col, ms=3, lw=1.1,
                   label=f"{name} ({results['Function 3'][name].iters} it)")
        a.plot(*p.x0, "ks"); a.plot(1, 1, "r*", ms=12)
        a.set_xlim(*xlim); a.set_ylim(*ylim); a.set_title(f"Rosenbrock: {title}"); a.set_xlabel("x1"); a.set_ylabel("x2")
    ax[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_rosenbrock_paths.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  3. condition-number scaling on F1
# --------------------------------------------------------------------------- #
def gd_exact_quadratic(p, x0, gtol=1e-8, max_iter=400_000):
    """Steepest descent with the closed-form exact step alpha = g'g / g'Hg (valid for quadratics,
    notes Sec 2.2).  Used only as the textbook reference for F1."""
    H = p.hess(x0)
    x, k = x0.copy(), 0
    g = p.grad(x)
    while np.linalg.norm(g, np.inf) > gtol and k < max_iter:
        a = (g @ g) / (g @ (H @ g))
        x = x - a * g
        g = p.grad(x)
        k += 1
    class R: iters = k
    return R


def exp_condition_scaling():
    md("## 3. Effect of conditioning: F1 with n = 10 ... 1000 (kappa(H) = n)\n")
    md("Median iterations over 5 random starts x0 ~ N(0, I) to ||g||_inf <= 1e-8.\n")
    ns = [10, 20, 50, 100, 200, 500, 1000]
    runs = {
        "GD (bt)": lambda p, x0: gradient_descent(p, x0, c=0.1, rho=0.5, adaptive_alpha0=True, gtol=1e-8, max_iter=400_000),
        "GD (Wolfe)": lambda p, x0: gradient_descent(p, x0, c=1e-4, c2=0.1, adaptive_alpha0=True, linesearch="wolfe",
                                                     gtol=1e-8, max_iter=400_000),
        "GD (exact LS)": lambda p, x0: gd_exact_quadratic(p, x0),
        "Newton": lambda p, x0: newton(p, x0, c=0.1, gtol=1e-8),
        "BFGS": lambda p, x0: quasi_newton(p, x0, update="bfgs", gtol=1e-8),
        "DFP": lambda p, x0: quasi_newton(p, x0, update="dfp", c2=0.1, gtol=1e-8),
    }
    med = {k: [] for k in runs}
    for n in ns:
        p = make_fun1(n)
        for k, run in runs.items():
            its = [run(p, np.random.default_rng(100 * n + s).standard_normal(n)).iters for s in range(5)]
            med[k].append(int(np.median(its)))
    table(["n = kappa"] + list(runs), [[n] + [med[k][i] for k in runs] for i, n in enumerate(ns)])
    sty = {"GD (bt)": ("o-", "tab:blue"), "GD (Wolfe)": ("s-", "tab:cyan"), "GD (exact LS)": ("*-", "tab:gray"), "BFGS": ("^-", "tab:green"),
           "DFP": ("v-", "tab:purple"), "Newton": ("d-", "tab:red")}
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for k, (m, col) in sty.items():
        ax.loglog(ns, med[k], m, color=col, label=k)
    ax.loglog(ns, np.array(ns) * med["GD (bt)"][0] / ns[0], "k:", lw=1, label=r"$\propto\kappa$ (slope 1)")
    ax.set_xlabel(r"problem size $n$ ( = condition number $\kappa$ )")
    ax.set_ylabel("iterations to ||g||_inf <= 1e-8 (median of 5)")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    ax.set_title("F1: iterations vs conditioning")
    fig.tight_layout(); fig.savefig(OUT / "fig_condition_scaling.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- #
#  4. line-search parameter study
# --------------------------------------------------------------------------- #
CS = [1e-4, 1e-2, 0.1, 0.4]
RHOS = [0.1, 0.3, 0.5, 0.7, 0.9]


def plot_param_grids(grids, names):
    """Heat-maps of GD iterations over (c, rho).  grids[(name, adaptive)] -> array (nan = failed)."""
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 6.8))
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#d9d9d9")
    for col, name in enumerate(names):
        vals = np.concatenate([grids[(name, a)].ravel() for a in (False, True)])
        L = np.log10(vals[~np.isnan(vals)])
        vmin, vmax = L.min(), L.max()                     # shared colour scale for both rows
        for row, adaptive in enumerate((False, True)):
            G = grids[(name, adaptive)]
            ax = axs[row, col]
            ax.imshow(np.ma.masked_invalid(np.log10(G)), cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
            for i in range(len(CS)):
                for j in range(len(RHOS)):
                    if np.isnan(G[i, j]):
                        ax.text(j, i, "not\nconv.", ha="center", va="center", color="k", fontsize=8)
                    else:
                        dark = (np.log10(G[i, j]) - vmin) / max(vmax - vmin, 1e-9) > 0.55
                        ax.text(j, i, int(G[i, j]), ha="center", va="center", fontsize=8,
                                color="w" if dark else "k")
            ax.set_xticks(range(len(RHOS))); ax.set_xticklabels(RHOS)
            ax.set_yticks(range(len(CS))); ax.set_yticklabels([f"{c:g}" for c in CS])
            ax.set_xlabel("rho"); ax.set_ylabel("c")
            ax.set_title(f"{name}: " + ("adaptive alpha0" if adaptive else "alpha0 = 1"), fontsize=10)
    fig.suptitle("Gradient descent: iterations to ||g||_inf <= 1e-8   (yellow = fast, dark = slow, "
                 "grey = not converged in 20 000;  colour scale shared by the two rows of a column)")
    fig.tight_layout(); fig.savefig(OUT / "fig_linesearch_params.png", dpi=150); plt.close(fig)


def exp_linesearch_params(problems):
    md("## 4. Influence of the backtracking parameters (c, rho, alpha0)\n")
    md("Gradient descent, stopping at ||g||_inf <= 1e-8, iteration cap 20 000 (`>20000` = not converged).\n"
       "Each cell: iterations / function evaluations.\n")
    CAP = 20_000
    grids = {}
    for p in problems:
        for adaptive in (False, True):
            tag = "adaptive alpha0" if adaptive else "alpha0 = 1"
            rows, G = [], np.full((len(CS), len(RHOS)), np.nan)
            for i, c in enumerate(CS):
                row = [f"c={c:g}"]
                for j, rho in enumerate(RHOS):
                    r = gradient_descent(p, c=c, rho=rho, adaptive_alpha0=adaptive, gtol=1e-8, max_iter=CAP)
                    row.append(f"{r.iters}/{r.nfev}" if r.converged else
                               (f">{CAP}/{r.nfev}" if r.reason == "max_iter" else f"ls-fail/{r.nfev}"))
                    G[i, j] = r.iters if r.converged else np.nan
                rows.append(row)
            grids[(p.name, adaptive)] = G
            md(f"**{p.name} - GD, {tag}**\n")
            table([""] + [f"rho={r}" for r in RHOS], rows)

    plot_param_grids(grids, [p.name for p in problems])

    # Newton / BFGS(bt): c and rho have almost no effect, because alpha = 1 is (almost) always accepted
    md("**Newton and BFGS (backtracking), same grid -- iterations (function evaluations)**\n")
    for p in problems:
        for label, fn in (("Newton", lambda c, rho: newton(p, c=c, rho=rho, gtol=1e-8)),
                          ("BFGS-bt", lambda c, rho: quasi_newton(p, update="bfgs", linesearch="backtracking",
                                                                  c=c, rho=rho, gtol=1e-8))):
            rows = []
            for c in CS:
                row = [f"c={c:g}"]
                for rho in RHOS:
                    r = fn(c, rho)
                    row.append(f"{r.iters} ({r.nfev})" if r.converged else f"FAIL")
                rows.append(row)
            md(f"*{p.name} - {label}*\n")
            table([""] + [f"rho={r}" for r in RHOS], rows)

    # c2 study for quasi-Newton
    md("**Wolfe curvature constant c2 for BFGS / DFP / GD (iterations to 1e-10)**\n")
    rows = []
    for p in problems:
        for label, mk in (("BFGS", lambda c2: quasi_newton(p, update="bfgs", c2=c2, gtol=GTOL)),
                          ("DFP", lambda c2: quasi_newton(p, update="dfp", c2=c2, gtol=GTOL, max_iter=5000)),
                          ("GD", lambda c2: gradient_descent(p, linesearch="wolfe", c2=c2, adaptive_alpha0=True,
                                                             gtol=GTOL, max_iter=100_000))):
            row = [p.name, label]
            for c2 in (0.9, 0.5, 0.1, 0.01):
                r = mk(c2)
                row.append(f"{r.iters}" if r.converged else f"FAIL ({r.iters} it, |g|={r.gnorm:.1e})")
            rows.append(row)
    table(["Function", "Method", "c2=0.9", "c2=0.5", "c2=0.1", "c2=0.01"], rows)


# --------------------------------------------------------------------------- #
#  5. initial conditions
# --------------------------------------------------------------------------- #
def random_feasible_start(data, seed, frac):
    A, b, _ = data
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(A.shape[1])
    d /= np.linalg.norm(d)
    ad = A @ d
    tmax = np.min(b[ad > 0] / ad[ad > 0])           # distance to the polytope boundary along d
    return frac * tmax * d


def exp_initial_conditions(data):
    md("## 5. Influence of the initial point (iterations to ||g||_inf <= 1e-8; `FAIL` = not converged)\n")
    methods = {"GD (bt)": CONFIGS["GD (backtracking)"], "GD (Wolfe)": CONFIGS["GD (Wolfe)"],
               "Newton": CONFIGS["Newton"], "BFGS": CONFIGS["BFGS (Wolfe)"]}

    def run_all(p, x0, label, rows):
        row = [label]
        for name, run in methods.items():
            mi = 100_000 if name.startswith("GD") else None
            r = run(p, x0, 1e-8, mi) if mi else run(p, x0, 1e-8)
            row.append(f"{r.iters}  (f={r.f:.3g})" if r.converged else f"FAIL (f={r.f:.3g})")
        rows.append(row)

    p = make_fun1()
    rng = np.random.default_rng(1)
    rows = []
    run_all(p, np.ones(100), "x0 = 1", rows)
    run_all(p, 1e-3 * np.ones(100), "x0 = 1e-3", rows)
    run_all(p, 1e3 * np.ones(100), "x0 = 1e3", rows)
    for s in range(3):
        run_all(p, 10 * rng.standard_normal(100), f"x0 ~ N(0,10^2) seed {s}", rows)
    md("**Function 1**\n")
    table(["start"] + list(methods), rows)

    p = make_fun2(data=data)
    rows = []
    run_all(p, np.zeros(100), "x0 = 0 (analytic-centre-ish)", rows)
    for frac in (0.5, 0.9, 0.99):
        for seed in (0, 1):
            run_all(p, random_feasible_start(data, seed, frac), f"{int(frac*100)}% to boundary, seed {seed}", rows)
    md("**Function 2** (every start strictly feasible; the 'x% to boundary' starts lie on a random ray, "
       "x% of the way to the polytope boundary)\n")
    table(["start"] + list(methods), rows)

    p = make_fun3()
    rows = []
    for x0 in [(-1.2, 1.0), (0, 0), (2, 2), (-2, 2), (1.5, -1.5), (-3, -4), (5, -5), (10, 10), (-10, 10)]:
        run_all(p, np.array(x0, float), f"x0 = {x0}", rows)
    md("**Function 3 (Rosenbrock)**\n")
    table(["start"] + list(methods), rows)


# --------------------------------------------------------------------------- #
#  6. cross-check with SciPy and save solutions
# --------------------------------------------------------------------------- #
def exp_scipy_crosscheck(problems, refs):
    """Independent check with SciPy's trust-region Newton (steps leaving the domain of F2
    are rejected by returning a huge value)."""
    from scipy.optimize import minimize
    md("## 6. Cross-check against SciPy (independent implementation, `trust-exact`)\n")
    rows = []
    for p in problems:
        xs, fs = refs[p.name]

        def fun(x, p=p):
            f = p.f(x)
            return f if np.isfinite(f) else 1e30

        r = minimize(fun, p.x0, jac=p.grad if p.name != "Function 2" else
                     (lambda x, p=p: p.grad(x) if np.isfinite(p.f(x)) else np.zeros(p.n)),
                     hess=p.hess if p.name != "Function 2" else
                     (lambda x, p=p: p.hess(x) if np.isfinite(p.f(x)) else np.eye(p.n)),
                     method="trust-exact", options=dict(gtol=1e-10, maxiter=5000))
        rows.append([p.name, "scipy trust-exact", f"{r.fun:.12e}", sci(abs(r.fun - fs)),
                     sci(np.linalg.norm(r.x - xs)), r.nit])
    table(["Function", "solver", "f", "|f - f*|", "||x - x*||", "iters"], rows)


def save_solutions(results, refs):
    sol_dir = OUT / "solutions"
    for fname, p_name in (("fun1", "Function 1"), ("fun2", "Function 2"), ("fun3", "Function 3")):
        best = min(results[p_name].values(), key=lambda r: (r.gnorm if r.converged else np.inf, r.iters))
        np.savetxt(sol_dir / f"{fname}_solution_x.txt", best.x, fmt="%.15e")
        np.savetxt(sol_dir / f"{fname}_reference_x.txt", refs[p_name][0], fmt="%.15e")
    with open(OUT / "summary_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["function", "method", "f_final", "f_error", "x_error", "grad_norm", "iters", "nfev", "time_s", "converged"])
        for p_name, d in results.items():
            xs, fs = refs[p_name]
            for m, r in d.items():
                w.writerow([p_name, m, f"{r.f:.15e}", f"{abs(r.f-fs):.3e}", f"{np.linalg.norm(r.x-xs):.3e}",
                            f"{r.gnorm:.3e}", r.iters, r.nfev, f"{r.time:.4f}", r.converged])


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    data = load_fun2_data()
    problems = [make_fun1(), make_fun2(data=data), make_fun3()]
    refs = {p.name: reference_solution(p) for p in problems}
    md(f"# Project 2 -- raw experiment output\n")
    md(f"Reference optimum of F2 (Newton, ||g||_inf <= 1e-12): f* = {refs['Function 2'][1]:.13f}\n")
    exp_derivative_check(problems)
    results = exp_main(problems, refs)
    fig_convergence(results, refs, problems)
    fig_newton_phases(results)
    fig_rosenbrock_paths(results)
    save_solutions(results, refs)
    exp_scipy_crosscheck(problems, refs)
    exp_condition_scaling()
    exp_initial_conditions(data)
    exp_linesearch_params(problems)
    (OUT / "tables.md").write_text("\n".join(MD))
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
