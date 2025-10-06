#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Empirical soundness test for RS Correlated Agreement (CA) vs Johnson-style ετ bounds.

We estimate:
  • Per-alpha acceptance probability  p̂_acc  := Pr_alpha[ list_decode(combo)>0 ]
    (averaged over many random instances and random alphas)
  • Per-instance "any alpha" success rate := Pr_instance[ ∃ alpha among sampled alphas s.t. list_decode(combo)>0 ]

We compare p̂_acc against ετ for τ ∈ {4,5,6,7}:
    ε_τ(q,n,k,δ) = (k+1)^2 / ( (2 * min{η, sqrt(ρ)/20})^τ * q ),
    ρ = k/n, η = 1 - sqrt(ρ) - δ, require η>0 (else ετ = +∞).

Notes
-----
• Domain is an n-th root-of-unity in F_p, so we REQUIRE n | (p-1).
• Decoding is brute-force via trying all k-subsets (choose(n,k)); keep n small.
• Use --no_force_flip to allow zero flips when s = ceil((1-δ)n) equals n (otherwise raw acceptance is impossible).
"""

import itertools, math, random, json, statistics
from typing import Optional, Dict, Any, List, Tuple

# ---------------- Field & poly utils ----------------

def poly_eval(coeffs: List[int], x: int, p: int) -> int:
    y = 0
    for c in reversed(coeffs):
        y = (y * x + c) % p
    return y

def eval_poly_vector(coeffs: List[int], xs: List[int], p: int) -> List[int]:
    return [poly_eval(coeffs, x, p) for x in xs]

def poly_add(a: List[int], b: List[int], p: int) -> List[int]:
    m = max(len(a), len(b))
    out = [0] * m
    for i in range(m):
        if i < len(a): out[i] = (out[i] + a[i]) % p
        if i < len(b): out[i] = (out[i] + b[i]) % p
    return out

def poly_mul(a: List[int], b: List[int], p: int) -> List[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] = (out[i + j] + ai * bj) % p
    return out

def poly_scale(a: List[int], s: int, p: int) -> List[int]:
    return [(ai * s) % p for ai in a]

def interpolate_lagrange(xs: List[int], ys: List[int], k: int, p: int) -> List[int]:
    """Unique degree<k polynomial through k points (xs, ys)."""
    coeffs = [0] * k
    for i in range(k):
        xi, yi = xs[i] % p, ys[i] % p
        num = [1]
        denom = 1
        for m in range(k):
            if m == i: continue
            num = poly_mul(num, [(-xs[m]) % p, 1], p)   # (x - x_m)
            denom = (denom * ((xi - xs[m]) % p)) % p
        inv_denom = pow(denom, p - 2, p)
        li = poly_scale(num, (yi * inv_denom) % p, p)  # yi * L_i(x)
        coeffs = poly_add(coeffs, li, p)
    coeffs = (coeffs + [0] * k)[:k]
    return coeffs

# ---------------- Domain ----------------

def primitive_root(p: int) -> int:
    if p == 2: return 1
    phi = p - 1
    fac = []
    m = phi; d = 2
    while d * d <= m:
        while m % d == 0:
            fac.append(d); m //= d
        d += 1
    if m > 1: fac.append(m)
    for g in range(2, p):
        ok = True
        for q in set(fac):
            if pow(g, phi // q, p) == 1:
                ok = False; break
        if ok: return g
    raise RuntimeError("no primitive root found")

def root_of_unity_domain(p: int, n: int) -> List[int]:
    if (p - 1) % n != 0:
        raise ValueError("n must divide p-1")
    g = primitive_root(p)
    omega = pow(g, (p - 1) // n, p)
    xs = [1]
    for _ in range(1, n):
        xs.append((xs[-1] * omega) % p)
    return xs

# ---------------- Generators ----------------

def rand_poly(k: int, p: int) -> List[int]:
    return [random.randrange(p) for _ in range(k)]

def gen_instance(xs: List[int], p: int, k: int, ell: int, delta: float,
                 aligned: bool = True, force_at_least_one_flip: bool = True) -> List[List[int]]:
    """
    Generate ℓ δ-close words; 'aligned' => same flipped indices across provers.
    """
    n = len(xs)
    fs: List[List[int]] = []

    def pick_t():
        t = math.floor(delta * n)
        if force_at_least_one_flip:
            t = max(1, t)
        else:
            t = max(0, t)
        return t

    if aligned:
        t = pick_t()
        flips = sorted(random.sample(range(n), t)) if t > 0 else []
        for _ in range(ell):
            coeffs = rand_poly(k, p)
            c = eval_poly_vector(coeffs, xs, p)
            y = c[:]
            for i in flips:
                y[i] = (y[i] + 1 + random.randrange(p - 1)) % p
            fs.append(y)
    else:
        for _ in range(ell):
            t = pick_t()
            flips = sorted(random.sample(range(n), t)) if t > 0 else []
            coeffs = rand_poly(k, p)
            c = eval_poly_vector(coeffs, xs, p)
            y = c[:]
            for i in flips:
                y[i] = (y[i] + 1 + random.randrange(p - 1)) % p
            fs.append(y)
    return fs

# ---------------- List decoding & witnesses ----------------

def list_decode(y: List[int], xs: List[int], k: int, p: int, delta: float) -> Tuple[int, List[Tuple[int,...]]]:
    """Return threshold s and *all* RS codewords agreeing with y on ≥ s coords."""
    n = len(xs)
    s = math.ceil((1 - delta) * n)
    codewords: Dict[Tuple[int,...], int] = {}
    for T in itertools.combinations(range(n), k):
        coeffs = interpolate_lagrange([xs[i] for i in T], [y[i] for i in T], k, p)
        cw = tuple(eval_poly_vector(coeffs, xs, p))
        agree = sum(cw[i] == y[i] for i in range(n))
        if agree > codewords.get(cw, -1):
            codewords[cw] = agree
    good = [cw for cw, a in codewords.items() if a >= s]
    return s, good

# ---------------- CA (single alpha) ----------------

def gen_linear_combo(fs: List[List[int]], alpha: int, p: int) -> List[int]:
    ell = len(fs); n = len(fs[0])
    weights = [pow(alpha, j, p) for j in range(ell)]
    return [(sum(weights[j] * fs[j][i] for j in range(ell)) % p) for i in range(n)]

def ca_accepts_for_alpha(xs: List[int], fs: List[List[int]], k: int, p: int, alpha: int, delta: float) -> bool:
    combo = gen_linear_combo(fs, alpha, p)
    _, good_list = list_decode(combo, xs, k, p, delta)
    return len(good_list) > 0

# ---------------- ετ ----------------

def epsilon_tau(q: int, n: int, k: int, delta: float, tau: int) -> float:
    """
    ε_τ(q,n,k,δ) = (k+1)^2 / ((2 * min{eta, sqrt(rho)/20})^τ * q)
    with ρ=k/n and η=1 - sqrt(ρ) - δ. If η<=0, return +inf.
    """
    if not (0 < k < n):
        return float("inf")
    rho = k / n
    sr = math.sqrt(rho)
    eta = 1 - sr - delta
    if eta <= 0:
        return float("inf")
    m = min(eta, sr / 20.0)
    base = 2.0 * m
    return ((k + 1) ** 2) / ((base ** tau) * q)

# ---------------- Experiment runner ----------------

def run_soundness(
    p: int,
    n: int,
    k: int,
    ell: int = 2,
    instances: int = 200,
    alphas_per_instance: int = 200,
    aligned: bool = True,
    force_at_least_one_flip: bool = True,
    delta: Optional[float] = None,
    seed: Optional[int] = None,
):
    if seed is not None:
        random.seed(seed)

    # Domain & basic params
    xs = root_of_unity_domain(p, n)
    rho = k / n
    default_delta = 1 - 1.01 * math.sqrt(rho)
    delta = default_delta if delta is None else delta
    s = math.ceil((1 - delta) * n)

    # Precompute theorem epsilons
    eps = {tau: epsilon_tau(p, n, k, delta, tau) for tau in (4,5,6,7)}

    # Sampling
    total_alpha = 0
    accept_alpha = 0
    per_instance_any_alpha_accept = 0

    # For optional diagnostics
    agree_sizes: List[int] = []

    # Sample loop
    for _ in range(instances):
        fs = gen_instance(xs, p, k, ell, delta, aligned, force_at_least_one_flip)
        # sample alphas uniformly from F_p^* (exclude 0)
        any_accept = False
        for _j in range(alphas_per_instance):
            alpha = random.randrange(1, p)  # 1..p-1 inclusive
            ok = ca_accepts_for_alpha(xs, fs, k, p, alpha, delta)
            total_alpha += 1
            if ok:
                accept_alpha += 1
                any_accept = True
        if any_accept:
            per_instance_any_alpha_accept += 1

    p_hat_alpha = accept_alpha / total_alpha if total_alpha > 0 else 0.0
    p_hat_any = per_instance_any_alpha_accept / instances if instances > 0 else 0.0

    # Report
    out = {
        "params": {
            "p": p, "n": n, "k": k, "ell": ell, "rho": rho,
            "delta": delta, "s": s, "aligned": aligned,
            "force_at_least_one_flip": force_at_least_one_flip,
            "instances": instances, "alphas_per_instance": alphas_per_instance
        },
        "epsilons": eps,
        "empirical": {
            "per_alpha_accept_prob": p_hat_alpha,
            "per_instance_any_alpha_rate": p_hat_any,
            "total_alpha": total_alpha,
            "accepted_alpha": accept_alpha
        }
    }
    print(json.dumps(out, indent=2))

    # Human-readable summary
    print("========== SUMMARY ==========")
    print(f"delta={delta:.6g}  sqrt(rho)={math.sqrt(rho):.6g}  s=ceil((1-δ)n)={s}")
    for tau in (4,5,6,7):
        e = eps[tau]
        e_txt = f"{e:.6g}" if e != float('inf') else "inf"
        print(f"ε_{tau} = {e_txt}")
    print(f"\nEmpirical per-alpha acceptance  p̂_acc = {p_hat_alpha:.6g} "
          f"({accept_alpha}/{total_alpha})")
    print(f"Empirical per-instance any-α rate = {p_hat_any:.6g} "
          f"({per_instance_any_alpha_accept}/{instances})")

    # Compare empirical against bounds
    # replace compare loop
    print("\n-- Comparison: is p̂_acc > ε_τ? --")
    for tau in (4,5,6,7):
        e = eps[tau]
        vacuous = (e > 1)
        e_clamped = min(1.0, e)
        verdict = "EXCEEDS (potentially refutes τ)" if (e_clamped < 1.0 and p_hat_alpha > e_clamped) else "≤ (within bound)"
        note = " [vacuous]" if vacuous else ""
        print(f"τ={tau}: p̂_acc {('>' if p_hat_alpha > e_clamped else '≤')} ε_{tau}={e:.6g}{note}  ==> {verdict}")

# ---------------- CLI ----------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Empirical soundness test for RS-CA vs ετ bounds")
    ap.add_argument("--p", type=int, default=2147483647, help="prime field (must be prime)")
    ap.add_argument("--n", type=int, default=6, help="block length; require n | (p-1)")
    ap.add_argument("--k", type=int, default=4, help="message dimension (0<k<n)")
    ap.add_argument("--ell", type=int, default=2, help="number of provers/words to combine")
    ap.add_argument("--instances", type=int, default=200, help="number of random instances to sample")
    ap.add_argument("--alphas-per-instance", type=int, default=200, help="number of random nonzero alphas per instance")
    ap.add_argument("--aligned", dest="aligned", action="store_true", help="use aligned flips (default)")
    ap.add_argument("--unaligned", dest="aligned", action="store_false", help="use unaligned flips")
    ap.set_defaults(aligned=True)
    ap.add_argument("--seed", type=int, default=None, help="PRNG seed")
    ap.add_argument("--delta", type=float, default=None, help="override δ; default: 1 - 1.01*sqrt(k/n)")
    ap.add_argument("--no_force_flip", action="store_true",
                    help="allow zero flips (useful when s=n); default forces ≥1 flip")
    args = ap.parse_args()

    run_soundness(
        p=args.p,
        n=args.n,
        k=args.k,
        ell=args.ell,
        instances=args.instances,
        alphas_per_instance=args.alphas_per_instance,
        aligned=args.aligned,
        force_at_least_one_flip=not args.no_force_flip,
        delta=args.delta,
        seed=args.seed,
    )
