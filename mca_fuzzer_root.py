#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exact MCA fuzzer (small n) for Reed-Solomon over GF(p) with ROOT-OF-UNITY domain.

- Uses exhaustive k-subset enumeration (exact) — suitable for small n (e.g., n ≤ ~22).
- Evaluation domain can be:
    * "root": multiplicative subgroup <ω> of size n (requires n | p-1)
    * "sequential": {1,2,...,n}
    * "random": n distinct elements from {1,...,p-1}

- Correlated check: for a list of alphas, check if ≥ beta fraction decode to agreement ≥ s.
- Mutual check: search for a single witness set S (via base T of size k) of size ≥ s for all f_j.

If a candidate (correlated holds, mutual fails) is found, it saves to mca_counterexample_root.json.
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
    """
    Coeffs of the unique degree<k polynomial over GF(p) interpolating k points.
    Returned in power basis: c0 + c1 x + ... + c_{k-1} x^{k-1}.
    """
    coeffs = [0]*k
    for i in range(k):
        xi, yi = xs[i] % p, ys[i] % p
        num = [1]
        denom = 1
        for m in range(k):
            if m == i: continue
            num = poly_mul(num, [(-xs[m]) % p, 1], p)   # (x - x_m)
            denom = (denom * ((xi - xs[m]) % p)) % p
        inv_denom = pow(denom, p-2, p)
        li = poly_scale(num, (yi*inv_denom) % p, p)     # yi * L_i(x)
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
        d += 1 if d == 2 else 2  # skip evens after 2
    if m > 1:
        fac[m] = fac.get(m, 0) + 1
    return fac

def primitive_root(p: int) -> int:
    """Return a primitive root g of GF(p), i.e., generator of F_p^* of order p-1."""
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
    raise RuntimeError("failed to find primitive root (should not happen for prime p)")

def root_of_unity_domain(p: int, n: int):
    """Return domain [ω^0, ω^1, ..., ω^{n-1}] where ω has exact order n. Requires n | p-1."""
    if (p - 1) % n != 0:
        raise ValueError(f"n={n} does not divide p-1={p-1}; cannot build size-n subgroup.")
    g = primitive_root(p)                  # order p-1
    omega = pow(g, (p - 1) // n, p)        # has order dividing n
    # ensure exact order n
    if pow(omega, n, p) != 1:
        raise RuntimeError("constructed omega does not satisfy omega^n=1")
    # check no smaller divisor
    for d in range(1, n):
        if n % d == 0 and pow(omega, d, p) == 1:
            raise RuntimeError("omega order < n (unexpected)")
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
        raise ValueError("domain kind must be one of: root, sequential, random")

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

def best_agreement_exact(y: List[int], xs: List[int], k: int, p: int) -> int:
    n = len(xs)
    best = -1
    for T in combinations(range(n), k):
        xT = [xs[i] for i in T]; yT = [y[i] for i in T]
        coeffs = interpolate_lagrange(xT, yT, k, p)
        cw = eval_poly_vector(coeffs, xs, p)
        agree = sum(int(cw[i] == y[i]) for i in range(n))
        if agree > best:
            best = agree
            if best == n: break
    return best

def correlated_agreement_holds(xs, fs, k, p, alpha_list, s_threshold, beta=0.5):
    ok = 0; records = []
    for a in alpha_list:
        combo = gen_linear_combo(fs, a, p)
        agree = best_agreement_exact(combo, xs, k, p)
        records.append((a, agree))
        if agree >= s_threshold:
            ok += 1
    return (ok >= math.ceil(beta * len(alpha_list))), ok, records

def mutual_agreement_max_witness(xs: List[int], fs: List[List[int]], k: int, p: int):
    n = len(xs); ell = len(fs)
    best = -1; best_T = None
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
        size = sum(match)
        if size > best:
            best = size; best_T = T
            if best == n: break
    return best, best_T

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

    # Preselect alphas (0 is allowed; we just sample)
    all_alphas = list(range(p))
    random.shuffle(all_alphas)
    alpha_list = all_alphas[:alphas_per_field]

    for t in range(tries):
        fs = gen_instance(xs, p=p, k=k, ell=ell, mode=mode, err_frac=err_frac)
        holds, good, recs = correlated_agreement_holds(xs, fs, k, p, alpha_list, s_threshold, beta)
        mut_size, mut_T = mutual_agreement_max_witness(xs, fs, k, p)

        if holds and mut_size < s_threshold:
            info = {
                "p": p, "n": n, "k": k, "ell": ell, "rho": rho,
                "domain": domain, "mode": mode, "err_frac": err_frac,
                "beta": beta, "alphas": alpha_list,
                "s_threshold": s_threshold,
                "cor_records": recs,
                "mutual_exact": {"size": mut_size, "T": mut_T},
                "seed": seed, "try_index": t
            }
            with open(outfile, "w") as f:
                json.dump({"params": info, "xs": xs, "fs": fs}, f, indent=2)
            print("Found CANDIDATE (exact) on domain=", domain, "Saved to", outfile)
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
