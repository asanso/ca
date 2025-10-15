#!/usr/bin/env python3

#This script finds suitable set of parameters (p, n, k, delta) for the test in main.py.

import math
import argparse
from typing import Set

def genPrimes(limit: int) -> Set[int]:
    """
    basic implementation of the Sieve of Eratosthenes
    """
    primes = {2}
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(3, limit + 1, 2):
        if is_prime[i]:
            primes.add(i)
            for multiple in range(i * i, limit + 1, i):
                is_prime[multiple] = False
    return primes

def findParams(rho: float, n_start: int, n_end: int, p_factor: float,
                prime_search_multiplier: int, primes: Set[int], sieve_limit: int):
    """
    The main experiment (main.py) tests a property of Reed-Solomon codes by
    taking two vectors "far" from the code and counting how many of their
    linear combinations are "close" to the code. This script finds parameters (p, n, k, delta),
    aiming for the smallest prime `p`.

    Args:
        rho: The desired code rate (k/n).
        n_start: The minimum codeword length (n) to search.
        n_end: The maximum codeword length (n) to search.
        p_factor: A factor to determine the initial search space for the prime p,
                  influencing how much larger p is than the threshold E.
        prime_search_multiplier: A factor to control how many prime candidates are
                                 checked for each (n, delta) pair. The search space
                                 is `p_candidate + prime_search_multiplier * n`.
        primes: A set of pre-computed primes for fast lookups.
        sieve_limit: The upper bound of the pre-computed primes set.
    """
    best_params = None
    smallest_p = float('inf')
    warning_printed = False

    print(f"Searching for best parameters with rho = {rho} in n range [{n_start}, {n_end}]...")
    for n in range(n_start, n_end + 1):
        # rho = k/n is the code rate. For k to be an integer, n*rho must be an integer.
        k_float = n * rho
        if not k_float.is_integer():
            continue
        k = int(k_float)

        # Theoretical bounds for delta from the underlying cryptographic paper.
        delta_min = math.floor((1 - math.sqrt(rho)) * n) + 1
        delta_max = math.ceil((1 - rho) * n) - 1

        for delta in range(delta_min, delta_max + 1):
            # d is the relative distance threshold (delta/n).
            d = delta / n
            # eta is a soundness parameter, must be > 0.
            eta = 1 - rho - d
            if eta <= 0:
                continue

            # E is the soundness threshold for the number of close linear combinations.
            E = n / (eta * rho)

            # Start searching for a prime p that is significantly larger than E.
            # We also need n to divide (p-1) for the Reed-Solomon domain to exist.
            # This is equivalent to p = 1 (mod n).
            p_start = math.ceil(p_factor * E)
            
            rem = (p_start - 1) % n
            if rem == 0:
                p_candidate = p_start
            else:
                p_candidate = p_start + (n - rem)

            # Search for the smallest prime p = 1 (mod n) starting from p_candidate.
            # The search space is limited to keep the script fast.
            for p in range(p_candidate, p_candidate + prime_search_multiplier * n, n):
                if p > sieve_limit:
                    if not warning_printed:
                        print(f"\n[Warning] Prime search for p={p} exceeded sieve limit of {sieve_limit}.")
                        print("Consider increasing --sieve-limit for a more exhaustive search.")
                        warning_printed = True
                    break # Stop searching for this (n, delta) pair

                if p in primes:
                    if p < smallest_p:
                        smallest_p = p
                        best_params = {
                            "n": n, "k": k, "delta": delta, "p": p,
                            "d": d, "eta": eta, "E": E, "p/E": p/E
                        }
                    break # Found the smallest prime for this (n, delta), move on

    if best_params:
        print("\n--- Found best set of parameters! ---")
        print(f"  n (codeword length) = {best_params['n']}")
        print(f"  k (message length)  = {best_params['k']}")
        print(f"  delta (threshold)   = {best_params['delta']}")
        print(f"  p (field size)      = {best_params['p']}")
        print("\n--- Derived values ---")
        print(f"  d (relative dist)   = {best_params['d']:.4f}")
        print(f"  eta (soundness)     = {best_params['eta']:.4f}")
        print(f"  E (threshold)       = {best_params['E']:.4f}")
        print(f"  p/E ratio           = {best_params['p/E']:.4f}")
    else:
        print("No suitable parameters found in the given range.")

def main():
    ap = argparse.ArgumentParser(
        description="Search for parameters (n, k, delta, p) for Reed-Solomon code analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("rho", type=float, help="The desired code rate rho (k/n).")
    ap.add_argument("--n_start", type=int, default=2, help="Starting value for n (codeword length).")
    ap.add_argument("--n_end", type=int, default=1000, help="Ending value for n.")
    ap.add_argument("--p_factor", type=float, default=10.0, help="A factor to control how much larger p should be than the threshold E.")
    ap.add_argument("--prime_search_multiplier", type=int, default=200,
                    help="A factor to control how many prime candidates are checked for each (n, delta) pair.")
    ap.add_argument("--sieve-limit", type=int, default=300000, help="Upper bound for the prime sieve.")
    args = ap.parse_args()

    print(f"Generating primes up to {args.sieve_limit}...")
    primes = genPrimes(args.sieve_limit)
    print("Prime generation complete.")

    findParams(
        rho=args.rho,
        n_start=args.n_start,
        n_end=args.n_end,
        p_factor=args.p_factor,
        prime_search_multiplier=args.prime_search_multiplier,
        primes=primes,
        sieve_limit=args.sieve_limit
    )

if __name__ == "__main__":
    main()
