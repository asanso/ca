#!/usr/bin/env python3
# algo.py
# Decoding, CA/MCA algorithms, and experiment runner.

import itertools, math, json, random
from typing import List, Tuple, Optional, Dict, Any, Set

from util import (
    interpolate_lagrange, eval_poly_vector,
    root_of_unity_domain, gen_instance, gen_linear_combo
)

# ---------- List decoding & witnesses ----------
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

# ---------- Correlated agreement (CA) ----------

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

# ---------- Intersection search for MCA ----------

def _max_intersection_over_candidates(per_alpha: List[Dict[str, Any]], n: int,
                                      exact_cap: int = 500_000, beam: int = 64):
    """
    Given per_alpha records (each has rec['candidates']), find the combination
    (choose exactly one candidate per alpha) that maximizes the size of the
    intersection of agree sets.
    """
    cand_sets = []
    list_sizes = []
    for rec in per_alpha:
        if rec["list_size"] == 0:
            return 0, set(), []  # impossible
        sets = [set(c["agree_indices"]) for c in rec["candidates"]]
        cand_sets.append(sets)
        list_sizes.append(len(sets))

    # exact feasible?
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
            order = sorted(range(len(cand_sets[i])),
                           key=lambda j: len(cand_sets[i][j]),
                           reverse=True)
            for j in order:
                nxt = cur_set & cand_sets[i][j]
                if len(nxt) <= best_size:
                    continue
                choice.append(j)
                dfs(i + 1, nxt, choice)
                choice.pop()

        dfs(0, set(range(n)), [])
        if best_size < 0:
            return 0, set(), []
        return best_size, best_S, best_choice

    # beam fallback
    BeamItem = tuple  # (intersection_set, choices_list)
    beam_items: List[BeamItem] = [(set(range(n)), [])]

    for sets in cand_sets:
        next_beam: List[BeamItem] = []
        order = sorted(range(len(sets)), key=lambda j: len(sets[j]), reverse=True)
        for S, choices in beam_items:
            for j in order:
                inter = S & sets[j]
                if not inter:
                    continue
                next_beam.append((inter, choices + [j]))
        if not next_beam:
            return 0, set(), []
        next_beam.sort(key=lambda item: len(item[0]), reverse=True)
        beam_items = next_beam[:beam]

    best_inter, best_choices = max(beam_items, key=lambda item: len(item[0]))
    return len(best_inter), best_inter, best_choices

# ---------- Mutual agreement (MCA) ----------

def mutual_agreement(xs: List[int], fs: List[List[int]], k: int, p: int,
                     alphas: List[int], delta: float, s: int):
    """
    MCA v2: For each alpha, decode to a list. Then choose ONE candidate per alpha
    to maximize the intersection size of the agree sets; check vs s.
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

# ---------- Runner ----------

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
        fs = gen_instance(xs, p, k, ell, delta, aligned,
                          force_at_least_one_flip=force_at_least_one_flip)
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
