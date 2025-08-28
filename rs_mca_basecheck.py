#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RS CA/MCA fuzzer with base closeness enforcement (root-of-unity domain).

Theorem mapping (BCIKS'20, Thm. 6.2), in this script's convention (deg < k ⇒ dim=k, ρ=k/n):
  If Pr_r[ u^(r) is δ-close ] > err  ⇒ every base u_j is δ-close.

We test δ-closeness via agreement threshold:
    s = ceil((1 - δ) * n), i.e., "δ-close" ⇔ agreement ≥ s.

err is modeled by the two regimes (clamped to [0,1], |F| = p):
  - 0 < δ < (1-ρ)/2         : err = c1 * (k n)/p
  - (1-ρ)/2 < δ < 1-1.01√ρ  : err = c2 * (k n^2)/p

CA decision (strict theorem form, no extra knobs):
  CA_ok  ⇔  #good  ≥  floor(err * total) + 1
(We also report has_witness = (#good ≥ 1), but it does NOT affect CA_ok.)

Counterexample condition (printed & returned):
  base_closeness holds, CA_ok is True (strict), but MCA witness size < s.

Defaults (toy):
  p=79, n=6, k=2, ell=2, tries=50, delta auto (just inside Johnson), seed=None, c1=c2=1.0
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

def gen_instance(xs, p, k, ell, mode="noisy_codewords", err_frac=0.45):
    """
    mode='patchwork': for each prover j:
       pick two RS codewords cA,cB (deg<k) and splice halves + light noise.
    """
    n = len(xs); fs = []
    if mode == "noisy_codewords":
        for _ in range(ell):
            c = eval_poly_vector(rand_poly(k, p), xs, p)
            y = c[:]
            m = int(err_frac * n)  # #errors
            flips = random.sample(range(n), m)
            for i in flips:
                y[i] = (y[i] + 1 + random.randrange(p-1)) % p
            fs.append(y)
        return fs  # return AFTER the loop
    if mode != "patchwork":
        raise ValueError("use mode='patchwork' or 'noisy_codewords'")
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

def correlated_agreement(xs, fs, k, p, alphas, s):
    """
    Strict CA counting: how many alphas give agreement >= s.
    Returns: (frac, good, total, records)
    """
    good = 0
    records = []
    for a in alphas:
        combo = gen_linear_combo(fs, a, p)
        agree = best_agreement_exact(combo, xs, k, p)
        records.append((a, agree))
        if agree >= s:
            good += 1
    total = len(alphas)
    frac = good / total if total > 0 else 0.0
    return frac, good, total, records

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

# ---------------- err ----------------

def compute_err(p: int, n: int, k: int, delta: float, rho: float, c1: float, c2: float) -> float:
    """
    |F| = p. Use the theorem's regimes:
      - 0 < δ < (1-ρ)/2         : err = c1 * (k n)/p
      - (1-ρ)/2 < δ < 1-1.01√ρ  : err = c2 * (k n^2)/p
    Clamp to [0,1].
    """
    if delta <= 0 or delta >= 1:  # out of meaningful range
        return 1.0
    if delta < (1 - rho) / 2:
        err = c1 * (k * n) / p
    elif delta < 1 - 1.01 * math.sqrt(rho):
        err = c2 * (k * (n ** 2)) / p
    else:
        # outside the stated regimes; treat as vacuous (err = 1)
        err = 1.0
    return max(0.0, min(1.0, err))

# ---------------- Runner ----------------

def run(p: int = 79, n: int = 6, k: int = 2, ell: int = 2, tries: int = 50,
        delta: Optional[float] = None,
        c1: float = 1.0, c2: float = 1.0,
        alphas: Optional[List[int]] = None,
        seed: Optional[int] = None,
        mode: str = "noisy_codewords",
        err_frac: float = 0.45):

    if seed is not None:
        random.seed(seed)

    if (p - 1) % n != 0:
        raise ValueError(f"(p-1) % n != 0; got p={p}, n={n}")

    xs = root_of_unity_domain(p, n)

    # rate under this script's convention: rho = k/n  (degree < k ⇒ dimension = k)
    rho = k / n
    if delta is None:
        delta = 1 - math.sqrt(rho) - 0.05  # just inside Johnson radius

    s = math.ceil((1 - delta) * n)

    # err per theorem (strict form)
    err = compute_err(p, n, k, delta, rho, c1, c2)

    if alphas is None:
        alphas = list(range(p))

    # Track only PASSING base-closeness trials
    last_CA_summary = None
    last_base_summary = None
    skipped_due_to_base = 0
    tested_after_base = 0

    for _ in range(tries):
        fs = gen_instance(xs, p, k, ell, mode=mode, err_frac=err_frac)

        base_ok, base_records = check_base_closeness(fs, xs, k, p, s)
        if not base_ok:
            skipped_due_to_base += 1
            continue  # enforce base closeness: skip this trial entirely

        # Record only passing base instances
        last_base_summary = {"ok": True, "records": base_records}
        tested_after_base += 1

        CA_frac, CA_good, CA_total, CA_records = correlated_agreement(
            xs, fs, k, p, alphas, s
        )

        # strict theorem requirement: Pr[u^(r) δ-close] > err ⇒ strict '>'
        passes_needed_err = int(math.floor(err * CA_total)) + 1
        CA_ok = (CA_good >= passes_needed_err)
        has_witness = (CA_good >= 1)

        last_CA_summary = {
            "ok": CA_ok,
            "good": CA_good,
            "total": CA_total,
            "frac": CA_frac,
            "passes_needed_err_strict": passes_needed_err,
            "meets_err_requirement": CA_ok,
            "has_witness": has_witness
        }

        MCA_size, MCA_S = mutual_agreement(xs, fs, k, p)

        if base_ok and CA_ok and MCA_size < s:
            result = {
                "params": {
                    "p": p, "n": n, "k": k, "ell": ell,
                    "rho": rho, "delta": delta, "s": s,
                    "err": err, "c1": c1, "c2": c2,
                    "mode": mode, "err_frac": err_frac
                },
                "domain": xs,
                "fs": fs,
                "base_closeness": last_base_summary,
                "CA": {
                    "ok": CA_ok,
                    "good": CA_good,
                    "total": CA_total,
                    "frac": CA_frac,
                    "passes_needed_err_strict": passes_needed_err,
                    "meets_err_requirement": CA_ok,
                    "has_witness": has_witness,
                    "records": CA_records
                },
                "MCA": {"size": MCA_size, "S": MCA_S},
                "trial_stats": {
                    "tries_requested": tries,
                    "skipped_due_to_base": skipped_due_to_base,
                    "tested_after_base": tested_after_base
                }
            }
            print(json.dumps(result, indent=2))
            print(f">> COUNTEREXAMPLE: bases close, strict-CA holds, MCA fails (size {MCA_size} < {s})")
            return result

    # No counterexample found: still print params + last PASSING base + CA summary + stats
    params = {
        "p": p, "n": n, "k": k, "ell": ell,
        "rho": rho, "delta": delta, "s": s,
        "err": err, "c1": c1, "c2": c2,
        "mode": mode, "err_frac": err_frac
    }
    out = {"params": params}

    if last_CA_summary is not None:
        out["CA_summary_last_try"] = last_CA_summary
    else:
        out["CA_summary_last_try"] = {
            "note": "No CA was evaluated because all trials were skipped by base closeness."
        }

    if last_base_summary is not None:
        out["base_closeness_last_checked"] = last_base_summary
    else:
        out["base_closeness_last_checked"] = {
            "ok": False,
            "note": "no trials had all bases δ-close (all were skipped)"
        }

    out["trial_stats"] = {
        "tries_requested": tries,
        "skipped_due_to_base": skipped_due_to_base,
        "tested_after_base": tested_after_base
    }

    print(json.dumps(out, indent=2))
    print("no counterexample found")
    return None

# ---------------- CLI ----------------

def _parse_alphas(arg: Optional[str], p: int) -> Optional[List[int]]:
    """
    --alphas parsing:
      * Comma-separated list: "0,3,5"
      * Single integer N: use list(range(N))
      * Omitted/None: return None (caller will default to 0..p-1)
    """
    if arg is None:
        return None
    s = arg.strip()
    if "," in s:
        vals = []
        for tok in s.split(","):
            tok = tok.strip()
            if tok == "": continue
            vals.append(int(tok) % p)
        # dedupe but preserve order
        seen = set(); out = []
        for v in vals:
            if v not in seen:
                seen.add(v); out.append(v)
        return out
    # single integer
    N = int(s)
    if N <= 0:
        return []
    if N > p:
        N = p
    return list(range(N))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=13)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--ell", type=int, default=2)
    ap.add_argument("--tries", type=int, default=50)
    ap.add_argument("--delta", type=float, default=None,
                    help="Decoding radius δ; if None, uses 1 - sqrt(rho) - 0.05 with rho=k/n.")
    ap.add_argument("--c1", type=float, default=1.0,
                    help="Constant for err = c1 * (k n)/|F| in the small-δ regime.")
    ap.add_argument("--c2", type=float, default=1.0,
                    help="Constant for err = c2 * (k n^2)/|F| in the Johnson regime.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducibility.")
    # Existing CLI flags
    ap.add_argument("--mode", type=str, default="noisy_codewords",
                    choices=["noisy_codewords", "patchwork"],
                    help="Instance generation mode.")
    ap.add_argument("--err-frac", type=float, default=0.45,
                    help="Error fraction for 'noisy_codewords' mode.")
    ap.add_argument("--alphas", type=str, default=None,
                    help="Comma list like '0,3,5' or a single integer N meaning alphas=0..N-1.")

    args = ap.parse_args()

    # Parse/normalize alphas
    parsed_alphas = _parse_alphas(args.alphas, args.p)

    run(p=args.p, n=args.n, k=args.k, ell=args.ell,
        tries=args.tries, delta=args.delta,
        c1=args.c1, c2=args.c2, seed=args.seed,
        alphas=parsed_alphas,
        mode=args.mode, err_frac=args.err_frac)
