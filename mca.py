#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RS MCA/CA fuzzer (Johnson regime, root-of-unity domain) with CA_witness_check.

- δ = 1 - 1.01*sqrt(rho), s = ceil((1-δ)*n)
- δ-close generators (aligned/unaligned)
- CA uses Johnson list decoding; records per-α list_size and, for the first passing α,
  remembers (alpha, combo, decoded codeword, agree_indices) as a CA witness.
- MCA reports size and witness set S, plus per-prover candidate & agree-indices.

On a counterexample (CA_ok and MCA_size < s), prints:
  1) the full JSON result, and
  2) a CA_witness_check block showing the combo word, decoded codeword, and per-index matches.
"""

import itertools, math, random, json
from typing import List, Tuple, Optional

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

def gen_instance(xs: List[int], p: int, k: int, ell: int, delta: float, aligned: bool = True) -> List[List[int]]:
    """Generate ℓ δ-close words; aligned flips => same flipped indices."""
    n = len(xs)
    fs: List[List[int]] = []
    if aligned:
        t = max(1, math.floor(delta * n))
        flips = sorted(random.sample(range(n), t))
        for _ in range(ell):
            coeffs = rand_poly(k, p)
            c = eval_poly_vector(coeffs, xs, p)
            y = c[:]
            for i in flips:
                y[i] = (y[i] + 1 + random.randrange(p - 1)) % p
            fs.append(y)
    else:
        for _ in range(ell):
            t = max(1, math.floor(delta * n))
            flips = sorted(random.sample(range(n), t))
            coeffs = rand_poly(k, p)
            c = eval_poly_vector(coeffs, xs, p)
            y = c[:]
            for i in flips:
                y[i] = (y[i] + 1 + random.randrange(p - 1)) % p
            fs.append(y)
    return fs

# ---------------- List decoding & witnesses ----------------

def list_decode(y: List[int], xs: List[int], k: int, p: int, delta: float):
    """Return threshold s and *all* RS codewords agreeing with y on ≥ s coords."""
    n = len(xs)
    s = math.ceil((1 - delta) * n)
    codewords = {}
    for T in itertools.combinations(range(n), k):
        coeffs = interpolate_lagrange([xs[i] for i in T], [y[i] for i in T], k, p)
        cw = tuple(eval_poly_vector(coeffs, xs, p))
        agree = sum(cw[i] == y[i] for i in range(n))
        codewords[cw] = max(codewords.get(cw, 0), agree)
    good = [cw for cw, a in codewords.items() if a >= s]
    return s, good

def best_agreement_witness(y: List[int], xs: List[int], k: int, p: int):
    """Return (max_agree, coeffs_mod_p, codeword, indices_agree, mismatches)."""
    n = len(xs); best = -1; best_data = None
    for T in itertools.combinations(range(n), k):
        coeffs = interpolate_lagrange([xs[i] for i in T], [y[i] for i in T], k, p)
        cw = eval_poly_vector(coeffs, xs, p)
        agree = sum(cw[i] == y[i] for i in range(n))
        if agree > best:
            S_star = [i for i in range(n) if cw[i] == y[i]]
            mism = [i for i in range(n) if cw[i] != y[i]]
            best = agree
            best_data = ([c % p for c in coeffs], cw, S_star, mism)
            if best == n: break
    coeffs_mod_p, cw, S_star, mism = best_data
    return best, coeffs_mod_p, cw, S_star, mism

# ---------------- CA / MCA ----------------

def gen_linear_combo(fs: List[List[int]], alpha: int, p: int) -> List[int]:
    ell = len(fs); n = len(fs[0])
    weights = [pow(alpha, j, p) for j in range(ell)]
    return [(sum(weights[j] * fs[j][i] for j in range(ell)) % p) for i in range(n)]

def correlated_agreement(xs: List[int], fs: List[List[int]], k: int, p: int,
                         alphas: List[int], delta: float):
    """
    CA with list decoding + debug: per-α list_size and, for the first α that passes,
    return a witness (alpha, combo, decoded codeword, agree_indices).
    """
    good = 0; records = []
    witness = None  # dict with keys: alpha, combo_word, decoded_codeword, agree_indices
    for a in alphas:
        if a == 0:  # skip alpha=0 as requested
            continue
        combo = gen_linear_combo(fs, a, p)
        s_needed, good_list = list_decode(combo, xs, k, p, delta)
        list_size = len(good_list)
        rec = {"alpha": a, "list_size": list_size}
        if list_size == 0:
            max_agree, coeffs, cw, S_star, mism = best_agreement_witness(combo, xs, k, p)
            rec["why_no_list"] = {
                "threshold_s": s_needed,
                "max_agreement": max_agree,
                "deficit": s_needed - max_agree,
                "best_poly_coeffs_mod_p": coeffs,
                "best_codeword": cw,
                "match_indices_best": S_star,
                "mismatch_indices": mism,
                "combo_word": combo
            }
        else:
            cw = list(good_list[0])
            agree_indices = [i for i in range(len(xs)) if cw[i] == combo[i]]
            rec["first_codeword"] = cw
            rec["agree_indices"] = agree_indices
            if witness is None:
                witness = {
                    "alpha": a,
                    "combo_word": combo,
                    "decoded_codeword": cw,
                    "agree_indices": agree_indices
                }
            good += 1
        records.append(rec)
    return good, records, witness

def mutual_agreement_debug(xs: List[int], fs: List[List[int]], k: int, p: int,
                           delta: float, s: int):
    """
    MCA with diagnostics.
    For each prover, take first list-decoded candidate (if any) and record agree-indices.
    Compute intersection; report clear pass/fail reason.
    """
    n = len(xs); ell = len(fs)
    per_prover = []
    empty = False
    per_sizes = []

    for j in range(ell):
        _, good_list = list_decode(fs[j], xs, k, p, delta)
        if not good_list:
            per_prover.append({
                "prover": j,
                "has_candidate": False,
                "candidate_codeword": None,
                "agree_indices": [],
                "word": fs[j]
            })
            per_sizes.append(0)
            empty = True
        else:
            cw = good_list[0]
            agree_set = [i for i in range(n) if cw[i] == fs[j][i]]
            per_prover.append({
                "prover": j,
                "has_candidate": True,
                "candidate_codeword": cw,
                "agree_indices": agree_set,
                "word": fs[j]
            })
            per_sizes.append(len(agree_set))

    # Intersection of agree sets (only meaningful if no empty list)
    if empty:
        intersection = []
    else:
        mask = [True] * n
        for rec in per_prover:
            cw = rec["candidate_codeword"]
            for i in range(n):
                if mask[i] and cw[i] != rec["word"][i]:
                    mask[i] = False
        intersection = [i for i in range(n) if mask[i]]

    size = len(intersection)
    passes = (not empty) and (size >= s)

    if empty:
        reason = f"MCA fails: at least one prover has empty Johnson list, cannot reach s={s}."
    elif passes:
        reason = f"MCA passes: intersection size = {size}, meets threshold s={s}. S = {intersection}"
    else:
        reason = f"MCA fails: intersection size = {size}, below threshold s={s}. S = {intersection}"

    debug = {
        "passes": passes,
        "needed_s": s,
        "intersection_size": size,
        "intersection_indices": intersection,
        "per_prover_agree_sizes": per_sizes,
        "per_prover": per_prover,
        "reason": reason,
    }
    return size, intersection, debug

# ---------------- Runner ----------------

def run(p: int = 13, n: int = 6, k: int = 2, ell: int = 2, tries: int = 10,
        aligned: bool = False, alphas: Optional[List[int]] = None, seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)
    xs = root_of_unity_domain(p, n)
    rho = k / n
    delta = 1 - 1.01 * math.sqrt(rho)  # Johnson-ish radius
    if alphas is None:
        alphas = list(range(min(p, 16)))
    s = math.ceil((1 - delta) * n)

    for t in range(tries):
        fs = gen_instance(xs, p, k, ell, delta, aligned)

        # CA (Johnson)
        good, ca_records, ca_witness = correlated_agreement(xs, fs, k, p, alphas, delta)
        CA_ok = (good >= 1)  # ∃ α with nonempty list

        # MCA (diagnostic)
        MCA_size, MCA_S, MCA_debug = mutual_agreement_debug(xs, fs, k, p, delta, s)

        if CA_ok and MCA_size < s:
            # Prepare result JSON
            result = {
                "trial": t,
                "params": {"p": p, "n": n, "k": k, "ell": ell, "rho": rho,
                           "delta": delta, "s": s, "aligned": aligned},
                "fs": fs,
                "CA": {
                    "ok": CA_ok, "good": good,
                    "records": ca_records,
                    "witness": ca_witness  # may be None if CA_ok came from later alphas skipped? (unlikely)
                },
                "MCA": {"size": MCA_size, "S": MCA_S, "debug": MCA_debug}
            }
            print(json.dumps(result, indent=2))

            # Extra clarity: CA_witness_check (only if we actually have a witness)
            if ca_witness is not None:
                u = ca_witness["combo_word"]
                cw = ca_witness["decoded_codeword"]
                per_index = [{"i": i, "combo": u[i], "decoded": cw[i], "match": (u[i] == cw[i])}
                             for i in range(len(u))]
                print(json.dumps({
                    "CA_witness_check": {
                        "alpha": ca_witness["alpha"],
                        "combo_word_u_alpha": u,
                        "decoded_codeword": cw,
                        "agree_indices": ca_witness["agree_indices"],
                        "per_index_match": per_index
                    }
                }, indent=2))

            print(f">> COUNTEREXAMPLE: CA holds via alpha={ca_witness['alpha'] if ca_witness else '<?>'} "
                  f"(agree on indices {ca_witness['agree_indices'] if ca_witness else '<?>'}), "
                  f"but MCA fails: size {MCA_size} < s={s}, S={MCA_S}")
            return result

    print("no counterexample found")
    return None

# ---------------- CLI ----------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=13)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--ell", type=int, default=2)
    ap.add_argument("--tries", type=int, default=10)
    ap.add_argument("--aligned", action="store_true", help="Use same flipped indices across provers")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--alphas", type=str, default=None, help="Comma list '0,3,5' or integer N => 0..N-1")
    args = ap.parse_args()

    def parse_alphas(arg: Optional[str], p: int) -> Optional[List[int]]:
        if arg is None: return None
        s = arg.strip()
        if "," in s:
            vals = [int(tok) % p for tok in s.split(",") if tok.strip() != ""]
            out = []; seen = set()
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
