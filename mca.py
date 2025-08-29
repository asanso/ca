#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools, math, random, json
from typing import List, Optional

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
    delta = 1 - math.sqrt(rho) - 0.05  # just inside Johnson radius

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