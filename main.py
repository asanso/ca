#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan F1, F2 over an ambient space and count alphas making F1 + α F2 δ-close to RS(n,k) over F_p.

Modes (picked automatically):
- DEFAULT (safe): iterate only over {2,...,p-1}^n using a scrambled order (avoids 0s/1s and bunching).
- FULL (huge): when --i-know-what-im-doing is passed, iterate over the FULL space (F_p)^n in base-p order.

For each pair (F1, F2):
  Proceed rule:
    - default: proceed if (F1 is δ-far) OR (F2 is δ-far)
    - with --require-f1-far: proceed only if F1 is δ-far
  Then compute counter = |{ α in F_p : F1 + α F2 is δ-close }|.
  Print counter; if counter > E (where E = n / (ρ η), ρ=k/n, η=1-ρ-δ), also print "counter > E".

Options (defaults in parentheses):
  --p (257), --n (8), --k (4), --delta (0.31)
  --require-f1-far
  Sharding/caps for both F1 and F2: --stride1/--offset1, --stride2/--offset2
  Caps: --limit1, --limit2, --max-pairs
  Safety switch: --i-know-what-im-doing  (enables FULL space traversal)

WARNING: The FULL space is enormous (e.g., p=257, n=8 ⇒ 257^8 ≈ 1.78e19 vectors; pairs ≈ 1e38).
         Use sharding/caps.
"""

import argparse
import math
import sys
from typing import Dict, Iterator, List, Optional, Tuple
from math import gcd

from util import *  # expects: root_of_unity_domain, list_decode

# ---- Closeness ----
def is_delta_close(y: List[int], xs: List[int], k: int, p: int, delta: float) -> bool:
    _, good = list_decode(y, xs, k, p, delta)
    return len(good) > 0

# ---- Enumerating all vectors ----
def index_to_vector(idx: int, p: int, n: int) -> List[int]:
    """Base-p expansion of idx of length n (least significant digit at position 0)."""
    v = [0] * n
    for i in range(n):
        v[i] = idx % p
        idx //= p
    return v

def iterate_vectors_full(p: int, n: int, start: int = 0, step: int = 1,
                         limit: Optional[int] = None) -> Iterator[List[int]]:
    """Stream vectors v in (F_p)^n by counting in base p.
       - start: first *index* (0..p^n-1)
       - step:  produce indices start, start+step, ...
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

def iterate_vectors_sparse(p: int, n: int, start: int = 0, step: int = 1,
                           limit: Optional[int] = None) -> Iterator[List[int]]:
    """Stream vectors v in (F_p)^n with each coordinate in {2,...,p-1} (no 0s, no 1s).
       Uses a sparsified / scrambled order to avoid bunching at small symbols.
       - start: logical start index (0..(p-2)^n-1)
       - step:  produce indices start, start+step, ...
       - limit: yield at most this many vectors
    """
    assert p >= 3 and n >= 1
    base = p - 2                  # digits map to values {2,...,p-1}
    total = base ** n

    # Pick an index-space permutation: j = (a*i + c) mod total, with gcd(a,total)=1
    def next_coprime(m: int, seed: int) -> int:
        a = max(2, seed % m)
        if a == m:
            a = 2
        while gcd(a, m) != 1:
            a += 1
            if a >= m:
                a = 2
        return a

    a = next_coprime(total, base * base + 1)
    c = (base ** (n // 2) + 1) % total

    # Per-position digit scrambler: d -> (m_pos * (d + o_pos)) mod base
    m_pos, o_pos = [], []
    for pos in range(n):
        m_pos.append(next_coprime(base, 2 * pos + 1))
        o_pos.append((pos * (pos + 3) + 7) % base)

    produced = 0
    i = start % total
    while i < total:
        if limit is not None and produced >= limit:
            return

        j = (a * i + c) % total

        # convert j in base=(p-2), LSB last
        x = j
        d = [0] * n
        for pos in range(n - 1, -1, -1):
            d[pos] = x % base
            x //= base

        # scramble per position, then shift to {2,...,p-1}
        for pos in range(n):
            d[pos] = (m_pos[pos] * (d[pos] + o_pos[pos])) % base
            d[pos] += 2

        yield d
        produced += 1
        i += step

# ---- Scan a single pair ----
def scan_pair(F1: List[int], F2: List[int], xs: List[int], p: int, k: int, delta: float, E: float,
              require_f1_far: bool = False) -> int:
    if not (len(F1) == len(F2) == len(xs)):
        raise ValueError("F1, F2, xs must have length n.")
    F1_close = is_delta_close(F1, xs, k, p, delta)
    F2_close = is_delta_close(F2, xs, k, p, delta)

    # Proceed rule (see module docstring)
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
        print("counter > E ", counter)
    return counter

# ---- Driver ----
def main():
    ap = argparse.ArgumentParser(description="Scan F1,F2 pairs with sharding/limits. Default iterates {2,...,p-1}^n; pass --i-know-what-im-doing for FULL (F_p)^n.")
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

    # safety switch: toggles FULL space traversal
    ap.add_argument("--i-know-what-im-doing", action="store_true",
                    help="Enable FULL (F_p)^n traversal. Without this, iterate only over {2,...,p-1}^n in scrambled order.")

    args = ap.parse_args()

    if not (0 < args.k < args.n):
        raise SystemExit("Require 0 < k < n.")
    if (args.p - 1) % args.n != 0:
        raise SystemExit("Require n | (p - 1).")

    xs = root_of_unity_domain(args.p, args.n)

    rho = args.k / args.n
    eta = 1 - rho - args.delta
    if eta <= 0:
        raise SystemExit(f"eta = 1 - k/n - delta must be positive; got eta={eta:.4f}.")
    E = args.n / (rho * eta)
    err = E / args.p
    space_desc = "FULL (F_p)^n" if args.i_know_what_im_doing else "{2,...,p-1}^n (scrambled)"
    print(f"[info] p={args.p}, n={args.n}, k={args.k}, delta={args.delta}, rho={rho:.3f}, eta={eta:.3f}, E={E:.1f}, err={err:.3f}")
    print(f"[info] iterating space: {space_desc}")

    # Choose iterator + size depending on mode
    iter_fn = iterate_vectors_full if args.i_know_what_im_doing else iterate_vectors_sparse
    total_vecs = (args.p ** args.n) if args.i_know_what_im_doing else ((args.p - 2) ** args.n)

    # Estimate workload to guard accidental huge runs
    total1 = (total_vecs - args.offset1 + args.stride1 - 1) // args.stride1
    total2 = (total_vecs - args.offset2 + args.stride2 - 1) // args.stride2
    eff1 = total1 if args.limit1 is None else min(total1, args.limit1)
    eff2 = total2 if args.limit2 is None else min(total2, args.limit2)
    est_pairs = eff1 * eff2

    if args.max_pairs is None and est_pairs > 1e7 and not args.i_know_what_im_doing:
        sys.exit(f"Refusing to launch a scan of ~{est_pairs:.2e} pairs without --max-pairs or --i-know-what-im-doing.")

    pairs = 0
    for F1 in iter_fn(args.p, args.n, start=args.offset1, step=args.stride1, limit=args.limit1):
        for F2 in iter_fn(args.p, args.n, start=args.offset2, step=args.stride2, limit=args.limit2):
            scan_pair(F1, F2, xs, args.p, args.k, args.delta, E, require_f1_far=args.require_f1_far)
            pairs += 1
            if args.max_pairs is not None and pairs >= args.max_pairs:
                print(f"[info] Reached --max-pairs={args.max_pairs}.")
                return
    print(f"[info] Completed {pairs} pairs.")

if __name__ == "__main__":
    main()
