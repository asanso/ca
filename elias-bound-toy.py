#!/usr/bin/env python3
# rs_elias_regime_demo.py
#
# Tiny RS example in the Elias-only regime.
# Uses a refined Elias-style counting bound on the *average* list size.
#
# Now with CLI options:
#   --mode   {corrupted,random}  how to sample centers y
#   --trials N                   number of trials
#   --seed   S                   RNG seed

import math
import random
import argparse

from util import (
    root_of_unity_domain,
    rand_poly,
    eval_poly_vector,
    list_decode,
)


def q_ary_entropy(delta: float, q: int) -> float:
    """q-ary entropy H_q(delta)."""
    if delta <= 0.0 or delta >= 1.0:
        return 0.0
    return (
        delta * math.log(q - 1, q)
        - delta * math.log(delta, q)
        - (1 - delta) * math.log(1 - delta, q)
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tiny RS example in the Elias-only regime."
    )
    parser.add_argument(
        "--mode",
        choices=["corrupted", "random"],
        default="corrupted",
        help=(
            "How to sample centers y: "
            "'corrupted' = RS codeword + t random errors "
            "'random' = uniform random word in F_q^n"
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of random samples / trials (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    # Parameters in the "Elias-only" regime:
    #   q = 13, n = 12, k = 5  => rho = 5/12
    #   delta = 0.5           => t = 6 errors
    p = 13
    n = 12
    k = 5
    delta = 0.5

    rho = k / n
    H = q_ary_entropy(delta, p)
    cap_rate = 1 - H
    eta = rho + H - 1  # this is rho - 1 + H_q(delta)
    poly_factor = math.sqrt(8 * n * delta * (1 - delta))
    avg_list_lb = (p ** (eta * n)) / poly_factor

    print(f"Field size q = {p}")
    print(f"n = {n}, k = {k}, rate rho = {rho:.6f}")
    print(f"delta = {delta:.6f}")
    print(f"(1 - delta)*n = {(1 - delta) * n:.3f}  vs  k = {k}")
    print("  -> (1 - delta)*n > k  (OUT of the trivial interpolation regime)\n")

    print(f"H_q(delta) ≈ {H:.6f}")
    print(f"1 - H_q(delta) ≈ {cap_rate:.6f}  (capacity rate at this delta)")
    print(f"rho - (1 - H_q(delta)) ≈ {rho - cap_rate:.6f}  (above capacity)")
    print(f"eta = rho + H_q(delta) - 1 ≈ {eta:.6f}\n")

    print("Refined Elias-style counting bound:")
    print("  E_y[|B(y, δn) ∩ C|] ≥ q^{n(ρ + H_q(δ) - 1)} / sqrt(8 n δ (1-δ))")
    print(
        f"For these parameters that is ≥ {p ** (eta * n):.2f} / {poly_factor:.2f}"
        f" ≈ {avg_list_lb:.2f}\n"
    )

    xs = root_of_unity_domain(p, n)

    def sample_one(mode: str):
        """Sample one center y according to `mode`, then list-decode."""
        t = int(delta * n)

        if mode == "corrupted":
            # Sample random codeword and corrupt t positions
            coeffs = rand_poly(k, p)
            c = eval_poly_vector(coeffs, xs, p)

            flips = random.sample(range(n), t)
            y = c[:]
            for i in flips:
                # add a random non-zero error
                y[i] = (y[i] + 1 + random.randrange(p - 1)) % p

        elif mode == "random":
            # Sample y uniformly from F_p^n
            y = [random.randrange(p) for _ in range(n)]

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        s, good = list_decode(y, xs, k, p, delta)
        return s, len(good)

    print(
        "Sampling centers y with mode ="
        f" '{args.mode}'  (trials = {args.trials}, seed = {args.seed})"
    )
    print("Random centers and their list sizes:")

    sizes = []
    for trial in range(args.trials):
        s, L = sample_one(args.mode)
        sizes.append(L)
        print(f"  trial {trial:2d}: s = {s}, distance ≤ {n - s}, list size = {L}")

    print("\nObserved list sizes:", sizes)
    print("max list size =", max(sizes))
    print("avg list size =", sum(sizes) / len(sizes))


if __name__ == "__main__":
    main()
