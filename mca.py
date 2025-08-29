#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools, math, random, json
from typing import List, Optional, Tuple

# ---------- Field & poly utils ----------

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
    coeffs = [0] * k
    for i in range(k):
        xi, yi = xs[i] % p, ys[i] % p
        num = [1]
        denom = 1
        for m in range(k):
            if m == i: continue
            num = poly_mul(num, [(-xs[m]) % p, 1], p)
            denom = (denom * ((xi - xs[m]) % p)) % p
        inv_denom = pow(denom, p - 2, p)
        li = poly_scale(num, (yi * inv_denom) % p, p)
        coeffs = poly_add(coeffs, li, p)
    return (coeffs + [0]*k)[:k]

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
        raise ValueError(f"(p-1) must be divisible by n; got p={p}, n={n}")
    g = primitive_root(p)
    omega = pow(g, (p - 1) // n, p)
    xs = [1]
    for _ in range(1, n):
        xs.append((xs[-1] * omega) % p)
    if len(set(xs)) != n:
        raise RuntimeError("domain elements not distinct (unexpected)")
    return xs

# ---------- Agreement / CA / MCA ----------

def best_agreement_exact(y: List[int], xs: List[int], k: int, p: int) -> int:
    n = len(xs); best = -1
    for T in itertools.combinations(range(n), k):
        coeffs = interpolate_lagrange([xs[i] for i in T], [y[i] for i in T], k, p)
        cw = eval_poly_vector(coeffs, xs, p)
        agree = sum(cw[i] == y[i] for i in range(n))
        if agree > best:
            best = agree
            if best == n: break
    return best

def gen_linear_combo(fs: List[List[int]], alpha: int, p: int) -> List[int]:
    ell = len(fs); n = len(fs[0])
    weights = [pow(alpha, j, p) for j in range(ell)]
    return [(sum(weights[j] * fs[j][i] for j in range(ell)) % p) for i in range(n)]

def mutual_agreement(xs: List[int], fs: List[List[int]], k: int, p: int) -> Tuple[int, Optional[List[int]]]:
    n = len(xs); ell = len(fs)
    best = -1; bestS = None
    for T in itertools.combinations(range(n), k):
        coeffs_list = []
        for j in range(ell):
            coeffs = interpolate_lagrange([xs[i] for i in T], [fs[j][i] for i in T], k, p)
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

def compute_err_johnson(p: int, n: int, k: int, delta: float, rho: float) -> float:
    if delta <= 0 or delta >= 1:
        return 1.0
    if delta < (1 - rho) / 2:
        err = (k * n) / p
    elif delta < 1 - 1.01 * math.sqrt(rho):
        err = (k * (n ** 2)) / p
    else:
        err = 1.0
    return max(0.0, min(1.0, err))

# ---------- δ-close generator ----------

def make_delta_close_word_with_flips(xs: List[int], p: int, k: int, delta: float,
                                     flips: Optional[List[int]] = None) -> Tuple[List[int], List[int], List[int], List[int]]:
    coeffs = [random.randrange(p) for _ in range(k)]
    c = eval_poly_vector(coeffs, xs, p)
    n = len(xs)
    if flips is None:
        t = max(1, math.floor(delta * n))
        flips = sorted(random.sample(range(n), t))
    y = c[:]
    for i in flips:
        y[i] = (y[i] + 1 + random.randrange(p - 1)) % p
    return y, coeffs, c, flips

def gen_instance_delta_close(xs: List[int], p: int, k: int, ell: int, delta: float,
                             aligned: bool, keep_meta: bool = True):
    fs = []; meta = []
    n = len(xs)
    if aligned:
        t = max(1, math.floor(delta * n))
        common_flips = sorted(random.sample(range(n), t))
        for _ in range(ell):
            y, coeffs, c, flips = make_delta_close_word_with_flips(xs, p, k, delta, flips=common_flips)
            fs.append(y)
            if keep_meta: meta.append({"coeffs": coeffs, "codeword": c, "flips": flips})
    else:
        for _ in range(ell):
            y, coeffs, c, flips = make_delta_close_word_with_flips(xs, p, k, delta, flips=None)
            fs.append(y)
            if keep_meta: meta.append({"coeffs": coeffs, "codeword": c, "flips": flips})
    return fs, meta

# ---------- Runner ----------

def run(p: int = 1009, n: int = 10, k: int = 4, ell: int = 2,
        tries: int = 50, aligned: bool = False,
        alphas: Optional[List[int]] = None,
        seed: Optional[int] = None):
    """
    Find CA_ok ∧ (MCA_size < s) counterexample.

    Use unaligned flips (default) to make MCA likely to fail.
    Increase p (e.g. 1009+) so err is small ⇒ CA is easier to pass.
    """
    if seed is not None:
        random.seed(seed)
    if (p - 1) % n != 0:
        raise ValueError(f"(p-1) % n != 0; got p={p}, n={n}")
    xs = root_of_unity_domain(p, n)
    rho = k / n
    delta = 1 - 1.01 * math.sqrt(rho)             # Johnson regime
    s = math.ceil((1 - delta) * n)

    if alphas is None:
        alphas = list(range(min(p, 64)))          # sample some α for speed

    err = compute_err_johnson(p, n, k, delta, rho)
    needed = math.floor(err * len(alphas)) + 1     # strict '> err'
    print(f"Parameters: p={p}, n={n}, k={k}, ell={ell}, rho={rho:.4f}, delta={delta:.4f}, s={s}, aligned={aligned}")

    for t in range(tries):
        fs, meta = gen_instance_delta_close(xs, p, k, ell, delta, aligned, keep_meta=True)

        # CA strict
        good = 0
        for a in alphas:
            combo = gen_linear_combo(fs, a, p)
            if best_agreement_exact(combo, xs, k, p) >= s:
                good += 1
        CA_ok = (good >= needed)

        # MCA
        MCA_size, MCA_S = mutual_agreement(xs, fs, k, p)

        result = {
            "trial": t,
            "params": {
                "p": p, "n": n, "k": k, "ell": ell,
                "rho": rho, "delta": delta, "s": s,
                "aligned": aligned, "err": err, "needed_for_CA": needed,
                "alphas_total": len(alphas)
            },
            "meta": meta,
            "CA": {"ok": CA_ok, "good": good},
            "MCA": {"size": MCA_size, "S": MCA_S},
        }
        print(json.dumps(result, indent=2))

        if  MCA_size < s:
            print(f">> COUNTEREXAMPLE at trial {t}: CA_ok=True but MCA_size={MCA_size} < s={s}")
            return result

    print("no counterexample found")
    return None

# ---------- CLI ----------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=79)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--ell", type=int, default=2)
    ap.add_argument("--tries", type=int, default=50)
    ap.add_argument("--aligned", action="store_true", help="Force same flipped indices across provers")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--alphas", type=str, default=None,
                    help="Comma list like '0,3,5' or a single integer N meaning alphas=0..N-1")
    args = ap.parse_args()

    def parse_alphas(arg: Optional[str], p: int) -> Optional[List[int]]:
        if arg is None: return None
        s = arg.strip()
        if "," in s:
            vals = []
            for tok in s.split(","):
                tok = tok.strip()
                if tok == "": continue
                vals.append(int(tok) % p)
            seen = set(); out = []
            for v in vals:
                if v not in seen:
                    seen.add(v); out.append(v)
            return out
        N = int(s)
        if N <= 0: return []
        if N > p: N = p
        return list(range(N))

    parsed_alphas = parse_alphas(args.alphas, args.p)
    run(p=args.p, n=args.n, k=args.k, ell=args.ell, tries=args.tries,
        aligned=args.aligned, alphas=parsed_alphas, seed=args.seed)
