#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan F1, F2 over the FULL ambient space (F_p)^n (not just codewords).

For each pair (F1, F2) in (F_p)^n × (F_p)^n:
  If F1 is δ-far OR F2 is δ-far from the RS(n, k) code (over F_p w.r.t. the root-of-unity domain),
    counter = |{ alpha in F_p : F1 + alpha * F2 is δ-close }|,
    print(counter) and (if counter > 128) also print "counter > 128".

Options:
- Defaults: p=257, n=8, k=4, δ=0.31.
- --require-f1-far: change proceed rule to ONLY proceed when F1 is δ-far.
- Sharding/caps: --stride*/--offset* for F1/F2 enumeration, --limit1/--limit2 and --max-pairs.

WARNING: For p=257, n=8 there are 257^8 ≈ 1.78e19 vectors; pairs are ≈ 10^38 — you must use caps/sharding.
"""

import argparse
import itertools
import math
import sys
from typing import Dict, Iterator, List, Optional, Tuple
from util import *

# ---- Closeness ----
def is_delta_close(y: List[int], xs: List[int], k: int, p: int, delta: float) -> bool:
    _, good = list_decode(y, xs, k, p, delta)
    #print(f"good: {good}")
    return len(good) > 0

# ---- Enumerating all vectors in (F_p)^n ----
def index_to_vector(idx: int, p: int, n: int) -> List[int]:
    """Base-p expansion of idx of length n (least significant digit at position 0)."""
    v = [0] * n
    for i in range(n):
        v[i] = idx % p
        idx //= p
    return v

def iterate_vectors(p: int, n: int, start: int = 0, step: int = 1, limit: Optional[int] = None) -> Iterator[List[int]]:
    """Stream vectors v in (F_p)^n by counting in base p.
       - start: first *index* (0..p^n-1)
       - step:  produce indices start, start+step, start+2*step, ...
       - limit: if given, stop after yielding this many vectors
    """
    total = p ** n
    produced = 0
    idx = start % total
    while idx < total:
        if limit is not None and produced >= limit:
            return
        yield index_to_vector(idx, p, n)
        produced += 1
        idx += step

# ---- Scan a single pair ----
def scan_pair(F1: List[int], F2: List[int], xs: List[int], p: int, k: int, delta: float, E:float, require_f1_far: bool = False) -> int:
    if not (len(F1) == len(F2) == len(xs)):
        raise ValueError("F1, F2, xs must have length n.")
    F1_close = is_delta_close(F1, xs, k, p, delta)
    F2_close = is_delta_close(F2, xs, k, p, delta)
    # Proceed rule:
    # - default: proceed if (F1 is δ-far) OR (F2 is δ-far)
    # - --require-f1-far: proceed only if F1 is δ-far
    if require_f1_far:
        proceed = not F1_close
    else:
        proceed = not (F1_close and F2_close)
 
    if not proceed:
        return 0

    counter = 0
    for alpha in range(p):
        combo = [(F1[i] + (alpha * F2[i]) % p) % p for i in range(len(xs))]
        if is_delta_close(combo, xs, k, p, delta):
            counter += 1
 
    if counter > E:
        print("counter > E")
    return counter

# ---- Driver ----
def main():
    ap = argparse.ArgumentParser(description="Scan ALL F1,F2 in (F_p)^n with sharding/limits (not just codewords).")
    ap.add_argument("--p", type=int, default=257)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--delta", type=float, default=0.31)

    # proceed rule toggle
    ap.add_argument("--require-f1-far", action="store_true",
                    help="Proceed only if F1 is δ-far (instead of (F1 far) OR (F2 far)).")

    # sharding for F1 and F2 spaces
    ap.add_argument("--stride1", type=int, default=1, help="stride over indices for F1")
    ap.add_argument("--offset1", type=int, default=0, help="offset in [0, stride1-1] for F1")
    ap.add_argument("--stride2", type=int, default=1, help="stride over indices for F2")
    ap.add_argument("--offset2", type=int, default=0, help="offset in [0, stride2-1] for F2")

    # caps
    ap.add_argument("--limit1", type=int, default=None, help="limit number of F1 vectors produced")
    ap.add_argument("--limit2", type=int, default=None, help="limit number of F2 vectors per F1")
    ap.add_argument("--max-pairs", type=int, default=None, help="stop after this many pairs")

    # safety switch: require explicit ack to traverse huge spaces
    ap.add_argument("--i-know-what-im-doing", action="store_true",
                    help="Required to run when search space exceeds 1e7 pairs and no --max-pairs given.")
    args = ap.parse_args()

    if not (0 < args.k < args.n):
        raise SystemExit("Require 0 < k < n.")
    if (args.p - 1) % args.n != 0:
        raise SystemExit("Require n | (p - 1).")

    xs = root_of_unity_domain(args.p, args.n)

    rho = args.k/args.n
    eta = 1 -rho - args.delta
    E = args.n/(rho*eta)
    # Estimate workload to guard accidental huge runs
    total_vecs = args.p ** args.n
    total1 = (total_vecs - args.offset1 + args.stride1 - 1) // args.stride1
    total2 = (total_vecs - args.offset2 + args.stride2 - 1) // args.stride2
    eff1 = total1 if args.limit1 is None else min(total1, args.limit1)
    eff2 = total2 if args.limit2 is None else min(total2, args.limit2)
    est_pairs = eff1 * eff2

    if args.max_pairs is None and est_pairs > 1e7 and not args.i_know_what_im_doing:
        sys.exit(f"Refusing to launch a scan of ~{est_pairs:.2e} pairs without --max-pairs or --i-know-what-im-doing.")

    pairs = 0
    for F1 in iterate_vectors(args.p, args.n, start=args.offset1, step=args.stride1, limit=args.limit1):
        for F2 in iterate_vectors(args.p, args.n, start=args.offset2, step=args.stride2, limit=args.limit2):
            scan_pair(F1, F2, xs, args.p, args.k, args.delta, E, require_f1_far=args.require_f1_far)
            pairs += 1
            if args.max_pairs is not None and pairs >= args.max_pairs:
                print(f"[info] Reached --max-pairs={args.max_pairs}.")
                return
    print(f"[info] Completed {pairs} pairs.")

if __name__ == "__main__":
    main()
