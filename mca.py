#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# ---------------- Agreement checks ----------------

def make_delta_close_word(xs, p, k, delta, rng=None):
    """
    Return a single word y that is δ-close to the RS code C.
    """
    if rng is None:
        rng = random
    n = len(xs)

    # pick a random codeword
    coeffs = [rng.randrange(p) for _ in range(k)]
    c = eval_poly_vector(coeffs, xs, p)

    # flip floor(delta * n) positions
    t = max(0, math.floor(delta * n))
    flip_positions = rng.sample(range(n), t)
    y = c[:]
    for i in flip_positions:
        y[i] = (y[i] + 1 + rng.randrange(p - 1)) % p

    return y

def is_delta_close(y, xs, k, p, delta):
    """
    Check if a received word y is δ-close to the RS code C.

    δ-close ⇔ agreement ≥ ceil((1 - δ) * n),
    where agreement = max over codewords of matches with y.
    """
    n = len(xs)
    threshold = math.ceil((1 - delta) * n)
    agreement = best_agreement_exact(y, xs, k, p)
    return agreement >= threshold

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

# ---------------- Runner ----------------

def run(p: int = 79, n: int = 6, k: int = 2, ell: int = 2, tries: int = 50,
        seed: Optional[int] = None):

    if seed is not None:
        random.seed(seed)

    print("ready")

    if (p - 1) % n != 0:
        raise ValueError(f"(p-1) % n != 0; got p={p}, n={n}")

    xs = root_of_unity_domain(p, n)
    # rate under this script's convention: rho = k/n  (degree < k ⇒ dimension = k)
    rho = k / n
    delta = 1 - 1.01*math.sqrt(rho) # just inside Johnson radius

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=13)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--ell", type=int, default=2)
    ap.add_argument("--tries", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducibility.")
    args = ap.parse_args()

    run(p=args.p, n=args.n, k=args.k, ell=args.ell,
        tries=args.tries,seed=args.seed)