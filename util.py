#!/usr/bin/env python3
# util.py
# Core utilities: polynomial ops, domain, generators, linear combos.

import math, random
from typing import List

import itertools, math, json, random
from typing import List, Tuple, Optional, Dict, Any, Set

# ---------- Field & polynomial utils ----------

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
            num = poly_mul(num, [(-xs[m]) % p, 1], p)  # (x - x_m)
            denom = (denom * ((xi - xs[m]) % p)) % p
        inv_denom = pow(denom, p - 2, p)
        li = poly_scale(num, (yi * inv_denom) % p, p)  # yi * L_i(x)
        coeffs = poly_add(coeffs, li, p)
    coeffs = (coeffs + [0] * k)[:k]
    return coeffs

# ---------- Domain ----------

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

# ---------- Generators ----------

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
        return max(1 if force_at_least_one_flip else 0, t)

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

# ---------- Linear combo ----------

def gen_linear_combo(fs: List[List[int]], alpha: int, p: int) -> List[int]:
    ell = len(fs); n = len(fs[0])
    weights = [pow(alpha, j, p) for j in range(ell)]
    return [(sum(weights[j] * fs[j][i] for j in range(ell)) % p) for i in range(n)]

def list_decode(y: List[int], xs: List[int], k: int, p: int, delta: float):
    """Return threshold s and *all* RS codewords agreeing with y on ≥ s coords."""
    n = len(xs)
    s = math.ceil((1 - delta) * n)
    codewords: Dict[Tuple[int, ...], int] = {}
    for T in itertools.combinations(range(n), k):
        coeffs = interpolate_lagrange([xs[i] for i in T], [y[i] for i in T], k, p)
        cw = tuple(eval_poly_vector(coeffs, xs, p))
        agree = sum(cw[i] == y[i] for i in range(n))
        if agree > codewords.get(cw, -1):
            codewords[cw] = agree
    good = [cw for cw, a in codewords.items() if a >= s]
    return s, good