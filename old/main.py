#!/usr/bin/env python3
# main.py
# CLI entrypoint.

import argparse
from typing import Optional, List
from algo import run

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=257)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--ell", type=int, default=2)
    ap.add_argument("--tries", type=int, default=10)
    ap.add_argument("--aligned", dest="aligned", action="store_true",
                    help="Use aligned flips (default).")
    ap.add_argument("--unaligned", dest="aligned", action="store_false",
                    help="Use unaligned flips.")
    ap.set_defaults(aligned=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--alphas", type=str, default=None,
                    help="Comma list '0,3,5' or integer N => 0..N-1")
    ap.add_argument("--no_force_flip", action="store_true")
    args = ap.parse_args()

    parsed_alphas = parse_alphas(args.alphas, args.p)
    run(p=args.p, n=args.n, k=args.k, ell=args.ell, tries=args.tries,
        aligned=args.aligned, alphas=parsed_alphas, seed=args.seed,
        force_at_least_one_flip=not args.no_force_flip)

if __name__ == "__main__":
    main()
