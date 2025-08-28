#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exact MCA fuzzer (small n) for Reed-Solomon over GF(p) with ROOT-OF-UNITY domain.

- Uses exhaustive k-subset enumeration (exact) — suitable for small n (e.g., n ≤ ~22).
- Evaluation domain can be:
    * "root": multiplicative subgroup <ω> of size n (requires n | p-1)
    * "sequential": {1,2,...,n}
    * "random": n distinct elements from {1,...,p-1}

- Correlated check (strict): for ≥ beta fraction of alphas, there exists a witness set S
  of size ≥ s_threshold where f*_α agrees with some RS codeword.
- Mutual check: search for a single witness set S of size ≥ s_threshold where
  each f_j agrees with some (possibly different) RS codeword.
"""
import argparse, json, math, random
from itertools import combinations
from typing import List, Tuple, Optional

# ----------------------------- Poly / Field utils -----------------------------
def poly_eval(coeffs: List[int], x: int, p: int) -> int:
    y = 0
    for c in reversed(coeffs):
        y = (y * x + c) % p
    return y

def eval_poly_vector(coeffs: List[int], xs: List[int], p: int) -> List[int]:
    return [poly_eval(coeffs, x, p) for x in xs]

def poly_add(a, b, p):
    m = max(len(a), len(b))
    out = [0]*m
    for i in range(m):
        if i < len(a): out[i] = (out[i] + a[i]) % p
        if i < len(b): out[i] = (out[i] + b[i]) % p
    return out

def poly_mul(a, b, p):
    out = [0]*(len(a)+len(b)-1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i+j] = (out[i+j] + ai*bj) % p
    return out

def poly_scale(a, s, p):
    return [(ai*s) % p for ai in a]

def interpolate_lagrange(xs, ys, k, p):
    coeffs = [0]*k
    for i in range(k):
        xi, yi = xs[i] % p, ys[i] % p
        num = [1]
        denom = 1
        for m in range(k):
            if m == i: continue
            num = poly_mul(num, [(-xs[m]) % p, 1], p)
            denom = (denom * ((xi - xs[m]) % p)) % p
        inv_denom = pow(denom, p-2, p)
        li = poly_scale(num, (yi*inv_denom) % p, p)
        coeffs = poly_add(coeffs, li, p)
    coeffs = (coeffs + [0]*k)[:k]
    return coeffs

# ----------------------------- Domain utilities -------------------------------
def factorize(n: int):
    fac = {}
    d = 2
    m = n
    while d * d <= m:
        while m % d == 0:
            fac[d] = fac.get(d, 0) + 1
            m //= d
        d += 1 if d == 2 else 2
    if m > 1:
        fac[m] = fac.get(m, 0) + 1
    return fac

def primitive_root(p: int) -> int:
    if p == 2:
        return 1
    phi = p - 1
    fac = factorize(phi)
    for g in range(2, p):
        ok = True
        for q in fac.keys():
            if pow(g, phi // q, p) == 1:
                ok = False; break
        if ok:
            return g
    raise RuntimeError("failed to find primitive root")

def root_of_unity_domain(p: int, n: int):
    if (p - 1) % n != 0:
        raise ValueError(f"n={n} does not divide p-1={p-1}")
    g = primitive_root(p)
    omega = pow(g, (p - 1) // n, p)
    if pow(omega, n, p) != 1:
        raise RuntimeError("omega^n != 1")
    xs = [1]
    for i in range(1, n):
        xs.append((xs[-1] * omega) % p)
    if len(set(xs)) != n:
        raise RuntimeError("domain elements not distinct")
    return xs

def build_domain(p: int, n: int, kind: str, seed: Optional[int] = None):
    kind = kind.lower()
    if kind == "root":
        return root_of_unity_domain(p, n)
    elif kind == "sequential":
        return list(range(1, n+1))
    elif kind == "random":
        rnd = random.Random(seed)
        return rnd.sample(range(1, p), n)
    else:
        raise ValueError("bad domain kind")

# ------------------------------- Generators ----------------------------------
def rand_poly(k: int, p: int) -> List[int]:
    return [random.randrange(p) for _ in range(k)]

def gen_instance(xs: List[int], p=97, k=6, ell=3, mode="patchwork", err_frac=0.45):
    n = len(xs)
    fs = []
    if mode == "random":
        for _ in range(ell):
            fs.append([random.randrange(p) for _ in range(n)])
    elif mode == "noisy_codewords":
        for _ in range(ell):
            c = eval_poly_vector(rand_poly(k, p), xs, p)
            y = c[:]
            m = max(1, int(round(err_frac * n)))
            flips = random.sample(range(n), m)
            for i in flips:
                y[i] = (y[i] + 1 + random.randrange(p-1)) % p
            fs.append(y)
    elif mode == "patchwork":
        for _ in range(ell):
            cA = eval_poly_vector(rand_poly(k, p), xs, p)
            cB = eval_poly_vector(rand_poly(k, p), xs, p)
            idx = list(range(n)); random.shuffle(idx)
            cut = n // 2
            A = set(idx[:cut])
            y = [cA[i] if i in A else cB[i] for i in range(n)]
            flips = random.sample(range(n), max(1, n//10))
            for i in flips:
                y[i] = (y[i] + 1 + random.randrange(p-1)) % p
            fs.append(y)
    else:
        raise ValueError("unknown mode")
    return fs

# ---------------------------- Agreement oracles -------------------------------
def gen_linear_combo(fs: List[List[int]], alpha: int, p: int) -> List[int]:
    ell = len(fs); n = len(fs[0])
    weights = [pow(alpha, j, p) for j in range(ell)]
    out = [0]*n
    for i in range(n):
        s = 0
        for j in range(ell):
            s = (s + weights[j]*fs[j][i]) % p
        out[i] = s
    return out

def best_agreement_exact(y: List[int], xs: List[int], k: int, p: int):
    """
    Return (max_agree, witness_S) where witness_S is a set of coordinates
    on which y agrees with some RS codeword.
    """
    n = len(xs)
    best = -1; best_S = None
    for T in combinations(range(n), k):
        xT = [xs[i] for i in T]; yT = [y[i] for i in T]
        coeffs = interpolate_lagrange(xT, yT, k, p)
        cw = eval_poly_vector(coeffs, xs, p)
        S = [i for i in range(n) if cw[i] == y[i]]
        if len(S) > best:
            best = len(S); best_S = S
            if best == n: break
    return best, best_S

def strict_correlated_agreement(xs, fs, k, p, alpha_list, s_threshold, beta=0.5):
    """
    Strict CA: For ≥ beta fraction of alphas, there exists a witness set S of size ≥ s_threshold.
    """
    ok = 0; records = []; witness_sets = []
    for a in alpha_list:
        combo = gen_linear_combo(fs, a, p)
        agree, S = best_agreement_exact(combo, xs, k, p)
        records.append({"alpha": a, "agree": agree, "S": S})
        if agree >= s_threshold:
            ok += 1
            witness_sets.append(set(S))
    return (ok >= math.ceil(beta * len(alpha_list))), ok, records, witness_sets

def mutual_agreement_max_witness(xs: List[int], fs: List[List[int]], k: int, p: int):
    """
    MCA: Find max size of a set S such that for each f_j there exists some RS codeword c_j
    with f_j(x) = c_j(x) for all x in S. S is common, but c_j may differ.
    """
    n = len(xs); ell = len(fs)
    best = -1; best_S = None
    for T in combinations(range(n), k):
        xT = [xs[i] for i in T]
        coeffs_list = []
        for j in range(ell):
            yT = [fs[j][i] for i in T]
            coeffs = interpolate_lagrange(xT, yT, k, p)
            coeffs_list.append(coeffs)
        match = [True]*n
        for j in range(ell):
            cw = eval_poly_vector(coeffs_list[j], xs, p)
            for i in range(n):
                if match[i] and cw[i] != fs[j][i]:
                    match[i] = False
        S = [i for i in range(n) if match[i]]
        if len(S) > best:
            best = len(S); best_S = S
            if best == n: break
    return best, best_S

# ---------------------------------- Runner -----------------------------------
def run_fuzz_root(
    tries=80,
    p=73, n=18, k=6, ell=3,
    domain="root",
    mode="patchwork",
    err_frac=0.45,
    beta=0.4,
    alphas_per_field=16,
    seed=None,
    outfile="mca_counterexample_root.json",
):
    if seed is not None:
        random.seed(seed)
    rho = k / n
    s_threshold = math.ceil((1 - math.sqrt(rho)) * n)

    xs = build_domain(p, n, domain, seed=seed)

    print("domain is ", xs)

    all_alphas = list(range(p))
    random.shuffle(all_alphas)
    alpha_list = all_alphas[:alphas_per_field]

    for t in range(tries):
        fs = gen_instance(xs, p=p, k=k, ell=ell, mode=mode, err_frac=err_frac)
        holds, good, recs, wsets = strict_correlated_agreement(xs, fs, k, p, alpha_list, s_threshold, beta)
        mut_size, mut_S = mutual_agreement_max_witness(xs, fs, k, p)

        if holds and mut_size < s_threshold:
            info = {
                "p": p, "n": n, "k": k, "ell": ell, "rho": rho,
                "domain": domain, "mode": mode, "err_frac": err_frac,
                "beta": beta, "alphas": alpha_list,
                "s_threshold": s_threshold,
                "cor_records": recs,
                "mutual_exact": {"size": mut_size, "S": mut_S},
                "seed": seed, "try_index": t
            }
            with open(outfile, "w") as f:
                json.dump({"params": info, "xs": xs, "fs": fs}, f, indent=2)
            print("Found CANDIDATE (CA holds, MCA fails) on domain=", domain, "Saved to", outfile)
            print(json.dumps(info, indent=2))
            return True, outfile, info

    print("No candidate found with current settings.")
    return False, None, None

# ----------------------------------- CLI -------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Exact MCA fuzzer (root-of-unity domain)")
    ap.add_argument("--tries", type=int, default=80)
    ap.add_argument("--p", type=int, default=73)
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--ell", type=int, default=3)
    ap.add_argument("--domain", choices=["root","sequential","random"], default="root")
    ap.add_argument("--mode", choices=["random","noisy_codewords","patchwork"], default="patchwork")
    ap.add_argument("--err-frac", type=float, default=0.45)
    ap.add_argument("--beta", type=float, default=0.4)
    ap.add_argument("--alphas", type=int, default=16)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--outfile", type=str, default="mca_counterexample_root.json")
    args = ap.parse_args()

    ok, path, info = run_fuzz_root(
        tries=args.tries, p=args.p, n=args.n, k=args.k, ell=args.ell,
        domain=args.domain, mode=args.mode, err_frac=args.err_frac,
        beta=args.beta, alphas_per_field=args.alphas, seed=args.seed,
        outfile=args.outfile
    )
    if ok:
        print("SUCCESS:", path)
    else:
        print("DONE: no candidate")

if __name__ == "__main__":
    main()
