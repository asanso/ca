#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RS CA/MCA fuzzer with base closeness check (root-of-unity domain).

- Agreement threshold: s = ceil((1 - delta) * n)
- Default delta: 1 - sqrt(rho) - 0.05 with rho = k/n (degree < k)
- CA requires fraction >= beta of alphas to pass, where:
      beta = min(1, (k * n**2) / (p - 1))

Behavior:
  1) Base closeness: each f_j individually δ-close (agreement ≥ s).
  2) Correlated Agreement (CA): for fraction ≥ beta of α, the combo f*_α is δ-close.
  3) MCA: search max common witness set S; counterexample if |S| < s while base+CA hold.

Defaults pick a small toy where beta < 1:
  p=79, n=6, k=2, ell=2, tries=50, delta auto, seed=None
"""

import itertools, math, random, json
from typing import List, Optional

# ---------------- Field & poly utils ----------------

def poly_eval(coeffs: List[int], x: int, p: int) -> int:
    y = 0
    for c in reversed(coeffs):
        y = (y * x + c) % p
    return y

def eval_poly_vector(coeffs: List[int], xs: List[int], p: int) -> List[int]:
    return [poly_eval(coeffs, x, p) for x in xs]

def poly_add(a, b, p):
    m = max(len(a), len(b))
    out = [0] * m
    for i in range(m):
        if i < len(a): out[i] = (out[i] + a[i]) % p
        if i < len(b): out[i] = (out[i] + b[i]) % p
    return out

def poly_mul(a, b, p):
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] = (out[i + j] + ai * bj) % p
    return out

def poly_scale(a, s, p):
    return [(ai * s) % p for ai in a]

def interpolate_lagrange(xs, ys, k, p):
    """
    Return coeffs of the unique degree<k polynomial through k points.
    Coeffs are [c0, c1, ..., c_{k-1}] in power basis.
    """
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

# ---------------- Domain / primitives ----------------

def primitive_root(p: int) -> int:
    if p == 2: return 1
    # factor p-1
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
    raise RuntimeError("no primitive root found (prime field)")

def root_of_unity_domain(p: int, n: int):
    if (p - 1) % n != 0:
        raise ValueError(f"(p-1) must be divisible by n; got p={p}, n={n}")
    g = primitive_root(p)
    omega = pow(g, (p - 1) // n, p)
    xs = [1]
    for _ in range(1, n):
        xs.append((xs[-1] * omega) % p)
    if len(set(xs)) != n:
        raise RuntimeError("domain elements not distinct (unexpected)")
    return xs

# ---------------- Generators ----------------

def rand_poly(k, p): return [random.randrange(p) for _ in range(k)]

def gen_instance(xs, p, k, ell, mode="patchwork", err_frac=0.45):
    """
    mode='patchwork': for each prover j:
       pick two RS codewords cA,cB (deg<k) and splice halves + light noise.
    """
    n = len(xs); fs = []
    if mode != "patchwork":
        raise ValueError("use mode='patchwork'")
    for _ in range(ell):
        cA = eval_poly_vector(rand_poly(k, p), xs, p)
        cB = eval_poly_vector(rand_poly(k, p), xs, p)
        idx = list(range(n)); random.shuffle(idx)
        cut = n // 2; A = set(idx[:cut])
        y = [cA[i] if i in A else cB[i] for i in range(n)]
        flips = random.sample(range(n), max(1, n // 6))
        for i in flips:
            y[i] = (y[i] + 1 + random.randrange(p - 1)) % p
        fs.append(y)
    return fs

# ---------------- Agreement checks ----------------

def best_agreement_exact(y, xs, k, p):
    """
    Max agreement with a degree<k polynomial by brute force:
      try all k-subsets, interpolate, evaluate, count matches.
    """
    n = len(xs); best = -1
    for T in itertools.combinations(range(n), k):
        coeffs = interpolate_lagrange([xs[i] for i in T],
                                      [y[i] for i in T], k, p)
        cw = eval_poly_vector(coeffs, xs, p)
        agree = sum(cw[i] == y[i] for i in range(n))
        if agree > best:
            best = agree
            if best == n: break
    return best

def gen_linear_combo(fs, alpha, p):
    ell = len(fs); n = len(fs[0])
    weights = [pow(alpha, j, p) for j in range(ell)]
    return [(sum(weights[j] * fs[j][i] for j in range(ell)) % p) for i in range(n)]

def check_base_closeness(fs, xs, k, p, s):
    records = []; ok = True
    for j, f in enumerate(fs):
        agree = best_agreement_exact(f, xs, k, p)
        records.append((j, agree))
        if agree < s: ok = False
    return ok, records

def correlated_agreement(xs, fs, k, p, alphas, s, beta):
    good = 0; records = []
    for a in alphas:
        combo = gen_linear_combo(fs, a, p)
        agree = best_agreement_exact(combo, xs, k, p)
        records.append((a, agree))
        if agree >= s: good += 1
    frac = good / len(alphas)
    return frac >= beta, frac, records

def mutual_agreement(xs, fs, k, p):
    """
    MCA: search a single S that works for all provers.
    Build polynomials from a base T of size k for each prover,
    then mark coordinates where ALL match. Return best |S|.
    """
    n = len(xs); ell = len(fs)
    best = -1; bestS = None
    for T in itertools.combinations(range(n), k):
        coeffs_list = []
        for j in range(ell):
            coeffs = interpolate_lagrange([xs[i] for i in T],
                                          [fs[j][i] for i in T], k, p)
            coeffs_list.append(coeffs)
        match = [True] * n
        for j in range(ell):
            cw = eval_poly_vector(coeffs_list[j], xs, p)
            for i in range(n):
                if match[i] and cw[i] != fs[j][i]:
                    match[i] = False
        size = sum(match)
        if size > best:
            best = size; bestS = [i for i in range(n) if match[i]]
            if best == n: break
    return best, bestS

# ---------------- Beta ----------------

def compute_beta(n: int, k: int, q: int) -> float:
    """
    Your requested formula:
        beta = min(1, (k * n^2) / (q - 1))
    Here k is the polynomial length (degree < k).
    """
    if q <= 1: return 1.0
    return min(1.0, (k * (n ** 2)) / (q - 1))

# ---------------- Runner ----------------

def run(p: int = 79, n: int = 6, k: int = 2, ell: int = 2, tries: int = 50,
        delta: Optional[float] = None,
        alphas: Optional[List[int]] = None,
        seed: Optional[int] = None):

    if seed is not None:
        random.seed(seed)

    if (p - 1) % n != 0:
        raise ValueError(f"(p-1) % n != 0; got p={p}, n={n}")

    xs = root_of_unity_domain(p, n)

    # rate (your original convention): rho = k / n  (degree < k)
    rho = k / n
    if delta is None:
        delta = 1 - math.sqrt(rho) - 0.05

    s = math.ceil((1 - delta) * n)
    beta = compute_beta(n, k, p)
    if alphas is None:
        alphas = list(range(p))

    for t in range(tries):
        fs = gen_instance(xs, p, k, ell, "patchwork")
        base_ok, base_records = check_base_closeness(fs, xs, k, p, s)
        CA_ok, CA_frac, CA_records = correlated_agreement(xs, fs, k, p, alphas, s, beta)
        MCA_size, MCA_S = mutual_agreement(xs, fs, k, p)

        if base_ok and CA_ok and MCA_size < s:
            result = {
                "params": {
                    "p": p, "n": n, "k": k, "ell": ell,
                    "rho": rho, "delta": delta, "s": s, "beta": beta
                },
                "domain": xs,
                "fs": fs,
                "base_closeness": {"ok": base_ok, "records": base_records},
                "CA": {"ok": CA_ok, "frac": CA_frac, "records": CA_records},
                "MCA": {"size": MCA_size, "S": MCA_S}
            }
            print(json.dumps(result, indent=2))
            print(f">> COUNTEREXAMPLE: bases close, CA holds, MCA fails (size {MCA_size} < {s})")
            return result

    # No counterexample found: still print params
    params = {
        "p": p, "n": n, "k": k, "ell": ell,
        "rho": rho, "delta": delta, "s": s, "beta": beta
    }
    print(json.dumps({"params": params}, indent=2))
    print("no counterexample found")
    return None

# ---------------- CLI ----------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=79)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--ell", type=int, default=2)
    ap.add_argument("--tries", type=int, default=50)
    ap.add_argument("--delta", type=float, default=None,
                    help="Decoding radius δ; if None, uses 1 - sqrt(rho) - 0.05 with rho=k/n.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducibility.")
    args = ap.parse_args()

    run(p=args.p, n=args.n, k=args.k, ell=args.ell,
        tries=args.tries, delta=args.delta, seed=args.seed)
