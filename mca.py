#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RS MCA/CA fuzzer (Johnson regime, root-of-unity domain) with proper multi-candidate handling.

- δ = 1 - 1.01*sqrt(rho), s = ceil((1-δ)*n)
- δ-close generators (aligned/unaligned); optional strict mode without forced flips
- CA records ALL candidates per α and returns the best-agreement witness
- MCA v2: like CA per α, but require all alphas to agree on the same subset S

On a counterexample (CA_ok and MCA_size < s), prints:
  1) the full JSON result, and
  2) a CA_witness_check block showing the combo word, decoded codeword, and per-index matches.
"""

import itertools, math, random, json
from typing import List, Tuple, Optional, Dict, Any

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

def list_decode(y: List[int], xs: List[int], k: int, p: int, delta: float):
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

def best_agreement_witness(y: List[int], xs: List[int], k: int, p: int):
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

def decode_word_debug(word: List[int], xs: List[int], k: int, p: int, delta: float) -> Dict[str, Any]:
    s_needed, good_list = list_decode(word, xs, k, p, delta)
    rec: Dict[str, Any] = {"threshold_s": s_needed, "list_size": len(good_list)}
    if not good_list:
        max_agree, coeffs, cw, S_star, mism = best_agreement_witness(word, xs, k, p)
        rec["why_no_list"] = {
            "max_agreement": max_agree,
            "deficit": s_needed - max_agree,
            "best_poly_coeffs_mod_p": coeffs,
            "best_codeword": cw,
            "match_indices_best": S_star,
            "mismatch_indices": mism,
            "word": word
        }
        rec["candidates"] = []
        return rec
    cands = []
    for cw in good_list:
        agree = [i for i in range(len(xs)) if cw[i] == word[i]]
        cands.append({
            "codeword": list(cw),
            "agree_indices": agree,
            "agree_size": len(agree),
        })
    cands.sort(key=lambda e: e["agree_size"], reverse=True)
    rec["candidates"] = cands
    return rec

# ---------------- CA / MCA ----------------

def gen_linear_combo(fs: List[List[int]], alpha: int, p: int) -> List[int]:
    ell = len(fs); n = len(fs[0])
    weights = [pow(alpha, j, p) for j in range(ell)]
    return [(sum(weights[j] * fs[j][i] for j in range(ell)) % p) for i in range(n)]

def correlated_agreement(xs: List[int], fs: List[List[int]], k: int, p: int,
                         alphas: List[int], delta: float):
    """
    CA (v2): for each alpha, decode and keep ALL candidates in the record.
    For the global 'witness', pick the candidate (over ALL alphas) with maximum agree_size.
    """
    good = 0
    records = []
    best_witness = None
    best_agree_size = -1

    for a in alphas:
        if a == 0:
            continue
        combo = gen_linear_combo(fs, a, p)
        rec = decode_word_debug(combo, xs, k, p, delta)
        rec["alpha"] = a
        rec["combo_word"] = combo

        if rec["list_size"] > 0:
            good += 1
            # consider ALL candidates for this alpha
            for cand in rec["candidates"]:
                if cand["agree_size"] > best_agree_size:
                    best_agree_size = cand["agree_size"]
                    best_witness = {
                        "alpha": a,
                        "combo_word": combo,
                        "decoded_codeword": cand["codeword"],
                        "agree_indices": cand["agree_indices"],
                        "agree_size": cand["agree_size"],
                        "threshold_s": rec["threshold_s"],
                    }

        records.append(rec)

    return good, records, best_witness

def _max_intersection_over_candidates(per_alpha: List[Dict[str, Any]], n: int,
                                      exact_cap: int = 500_000, beam: int = 64):
    """
    Given per_alpha records (each has rec['candidates']), find the combination
    (choose exactly one candidate per alpha) that maximizes the size of the
    intersection of agree sets.

    Returns (best_size, best_S, best_choice_idxs), where:
      - best_S is a set of indices (intersection)
      - best_choice_idxs is a list of candidate indices (one per alpha)

    Strategy:
      - If product of list sizes <= exact_cap: exact DFS with pruning.
      - Else: beam search (greedy) to keep runtime bounded.
    """
    cand_sets = []
    list_sizes = []
    for rec in per_alpha:
        if rec["list_size"] == 0:
            return 0, set(), []  # impossible
        sets = [set(c["agree_indices"]) for c in rec["candidates"]]
        cand_sets.append(sets)
        list_sizes.append(len(sets))

    # exact search if feasible
    total = 1
    for m in list_sizes:
        total *= m

    if total <= exact_cap:
        best_size = -1
        best_S = set()
        best_choice = []

        def dfs(i, cur_set, choice):
            nonlocal best_size, best_S, best_choice
            if i == len(cand_sets):
                sz = len(cur_set)
                if sz > best_size:
                    best_size = sz
                    best_S = cur_set
                    best_choice = choice[:]
                return
            # order candidates by descending set size for better pruning
            order = sorted(range(len(cand_sets[i])),
                           key=lambda j: len(cand_sets[i][j]),
                           reverse=True)
            for j in order:
                nxt = cur_set & cand_sets[i][j]
                # pruning: can't beat current best
                if len(nxt) <= best_size:
                    continue
                choice.append(j)
                dfs(i + 1, nxt, choice)
                choice.pop()

        dfs(0, set(range(n)), [])
        if best_size < 0:
            return 0, set(), []
        return best_size, best_S, best_choice

    # beam search fallback
    BeamItem = tuple  # (intersection_set, choices_list)
    beam_items: List[BeamItem] = [(set(range(n)), [])]

    for i, sets in enumerate(cand_sets):
        next_beam: List[BeamItem] = []
        # try larger sets first
        order = sorted(range(len(sets)), key=lambda j: len(sets[j]), reverse=True)
        for S, choices in beam_items:
            for j in order:
                inter = S & sets[j]
                if not inter:
                    continue
                next_beam.append((inter, choices + [j]))
        if not next_beam:
            return 0, set(), []
        # keep top 'beam' by intersection size
        next_beam.sort(key=lambda item: len(item[0]), reverse=True)
        beam_items = next_beam[:beam]

    best_inter, best_choices = max(beam_items, key=lambda item: len(item[0]))
    return len(best_inter), best_inter, best_choices



def mutual_agreement(xs: List[int], fs: List[List[int]], k: int, p: int,
                     alphas: List[int], delta: float, s: int):
    """
    MCA v2: For each alpha, decode to a list. Then choose ONE candidate per alpha
    (search over all candidates, exact with pruning or beam fallback) to maximize
    the intersection size of the agree sets. Report pass/fail vs s and the
    chosen combination.
    """
    per_alpha = []
    n = len(xs)
    empty = False

    for a in alphas:
        if a == 0:
            continue
        combo = gen_linear_combo(fs, a, p)
        rec = decode_word_debug(combo, xs, k, p, delta)
        rec["alpha"] = a
        rec["combo_word"] = combo
        per_alpha.append(rec)
        if rec["list_size"] == 0:
            empty = True

    if empty:
        return 0, [], {
            "passes": False,
            "reason": f"MCA fails: at least one alpha had empty list; need s={s}",
            "per_alpha": per_alpha
        }

    best_size, best_S, best_choice_idxs = _max_intersection_over_candidates(per_alpha, n)

    passes = best_size >= s
    reason = (f"MCA passes: intersection size={best_size}, meets s={s}"
              if passes else
              f"MCA fails: intersection size={best_size}, below s={s}")

    # annotate which candidate was chosen per alpha (if any)
    chosen = []
    if best_choice_idxs:
        for rec, j in zip(per_alpha, best_choice_idxs):
            cand = rec["candidates"][j]
            chosen.append({
                "alpha": rec["alpha"],
                "chosen_candidate_index": j,
                "agree_size": cand["agree_size"],
                "agree_indices": cand["agree_indices"],
                "codeword": cand["codeword"],
            })

    debug = {
        "passes": passes,
        "reason": reason,
        "per_alpha": per_alpha,
        "chosen_per_alpha": chosen
    }
    return best_size, sorted(best_S), debug

# ---------------- Runner ----------------

def run(p: int = 13, n: int = 6, k: int = 2, ell: int = 2, tries: int = 10,
        aligned: bool = True, alphas: Optional[List[int]] = None, seed: Optional[int] = None,
        force_at_least_one_flip: bool = True):
    if seed is not None:
        random.seed(seed)
    xs = root_of_unity_domain(p, n)
    rho = k / n
    delta = 1 - 1.01 * math.sqrt(rho)
    if alphas is None:
        alphas = list(range(min(p, 16)))
    s = math.ceil((1 - delta) * n)
    for t in range(tries):
        fs = gen_instance(xs, p, k, ell, delta, aligned, force_at_least_one_flip=force_at_least_one_flip)
        good, ca_records, ca_witness = correlated_agreement(xs, fs, k, p, alphas, delta)
        CA_ok = (good >= 1)
        MCA_size, MCA_S, MCA_debug = mutual_agreement(xs, fs, k, p, alphas, delta, s)
        if CA_ok and MCA_size < s:
            result = {
                "trial": t,
                "params": {"p": p, "n": n, "k": k, "ell": ell, "rho": rho,
                           "delta": delta, "s": s, "aligned": aligned},
                "fs": fs,
                "CA": {"ok": CA_ok, "good": good, "records": ca_records, "witness": ca_witness},
                "MCA": {"size": MCA_size, "S": MCA_S, "debug": MCA_debug}
            }
            print(json.dumps(result, indent=2))
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
            print(f">> COUNTEREXAMPLE: CA holds via alpha={ca_witness['alpha'] if ca_witness else '<?>'}, "
                  f"but MCA fails: size {MCA_size} < s={s}, S={MCA_S}")
            return result
    print("no counterexample found")
    return None

# ---------------- CLI ----------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=29)
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--ell", type=int, default=2)
    ap.add_argument("--tries", type=int, default=10)
    # Default to aligned; allow opting out with --unaligned
    ap.add_argument(
        "--aligned",
        dest="aligned",
        action="store_true",
        help="Use aligned flips (default)."
    )
    ap.add_argument("--unaligned", dest="aligned", action="store_false", help="Use unaligned flips.")
    ap.set_defaults(aligned=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--alphas", type=str, default=None, help="Comma list '0,3,5' or integer N => 0..N-1")
    ap.add_argument("--no_force_flip", action="store_true")
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
        aligned=args.aligned, alphas=parsed_alphas, seed=args.seed,
        force_at_least_one_flip=not args.no_force_flip)
