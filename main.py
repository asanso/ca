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
  If counter > E, print "counter > E <counter>".

E / err selection:
  Let ρ = k/n. If δ < 1 - sqrt(ρ) (the Johnson radius), use Theorem 4.6:
      η_J = 1 - sqrt(ρ) - δ   (> 0)
      m   = min(η_J, sqrt(ρ)/20)
      E   = k^2 / (2*m)^7
      err = E / |F| = E / p
  Otherwise (δ ≥ 1 - sqrt(ρ)), use Conjecture 4.7 with user exponents c1,c2,c3:
      η   = 1 - ρ - δ         (require η > 0, i.e., δ < 1 - ρ)
      E   = n^{c1} / (ρ^{c2} * η^{c3})
      err = E / p

Stats (optional):
  --collect-stats to gather a histogram of `counter`
  --stats-csv path.csv  to dump value,count
  --stats-plot path.png to save a bar chart (and CDF if --stats-cdf);
                  plot also shows run metadata and a marker for E.
  --stats-cdf to overlay empirical CDF (requires --stats-plot)

WARNING: FULL space is enormous (e.g., p=257, n=8 ⇒ 257^8 ≈ 1.78e19 vectors; pairs ≈ 1e38).
"""

import argparse
import csv
import sys
from collections import Counter
from math import gcd, sqrt
from typing import Iterator, List, Optional, Tuple

from util import *  # expects: root_of_unity_domain, list_decode


# ---- Utility ----
def vprint(verbose: bool, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def build_plot_info(
    *,
    p: int,
    n: int,
    k: int,
    delta: float,
    rho: float,
    johnson: float,
    regime: str,
    eta_like: float,
    c1: int,
    c2: int,
    c3: int,
    E: float,
    err: float,
    space_desc: str,
) -> str:
    """Assemble a multi-line string with run metadata to draw on the plot."""
    header = (
        f"p={p}, n={n}, k={k}, delta={delta}, "
        f"rho={rho:.3f}, Johnson(1-sqrt(rho))={johnson:.3f}"
    )
    if regime == "theorem4.6":
        reg = f"regime=Theorem 4.6, eta_J={eta_like:.3f}"
    else:
        reg = f"regime=Conjecture 4.7, c1={c1}, c2={c2}, c3={c3}, eta={eta_like:.3f}"
    tail = f"E={E:.6g}, err=E/p={err:.6g} | iterating space: {space_desc}"
    return header + "\n" + reg + "\n" + tail


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


def iterate_vectors_full(
    p: int,
    n: int,
    start: int = 0,
    step: int = 1,
    limit: Optional[int] = None,
    *,
    verbose: bool = False,
) -> Iterator[List[int]]:
    """Stream vectors v in (F_p)^n by counting in base p."""
    total = p ** n
    vprint(verbose, f"[iter/full] total={total:,} start={start} step={step} limit={limit}")
    produced = 0
    idx = start % total
    while idx < total:
        if limit is not None and produced >= limit:
            return
        yield index_to_vector(idx, p, n)
        produced += 1
        idx += step


def iterate_vectors_sparse(
    p: int,
    n: int,
    start: int = 0,
    step: int = 1,
    limit: Optional[int] = None,
    *,
    verbose: bool = False,
) -> Iterator[List[int]]:
    """Stream vectors v in (F_p)^n with each coordinate in {2,...,p-1} (no 0s, no 1s).
       Uses a sparsified / scrambled order to avoid bunching at small symbols.
    """
    assert p >= 3 and n >= 1
    base = p - 2
    total = base ** n

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

    m_pos, o_pos = [], []
    for pos in range(n):
        m_pos.append(next_coprime(base, 2 * pos + 1))
        o_pos.append((pos * (pos + 3) + 7) % base)

    vprint(
        verbose,
        f"[iter/sparse] base={base} total={total:,} start={start} step={step} limit={limit}\n"
        f"[iter/sparse] perm a={a}, c={c}\n"
        f"[iter/sparse] first multipliers={m_pos[:min(8,n)]}, first offsets={o_pos[:min(8,n)]}",
    )

    produced = 0
    i = start % total
    while i < total:
        if limit is not None and produced >= limit:
            return

        j = (a * i + c) % total

        x = j
        d = [0] * n
        for pos in range(n - 1, -1, -1):
            d[pos] = x % base
            x //= base

        for pos in range(n):
            d[pos] = (m_pos[pos] * (d[pos] + o_pos[pos])) % base
            d[pos] += 2

        yield d
        produced += 1
        i += step


# ---- Parameters -> E, err (Theorem 4.6 vs Conjecture 4.7) ----
def compute_E_and_err(
    p: int,
    n: int,
    k: int,
    delta: float,
    c1: int,
    c2: int,
    c3: int,
) -> Tuple[float, float, float, float, str]:
    """
    Select formula based on δ vs. Johnson radius 1 - sqrt(ρ).

    Returns:
      E, err, rho, eta_like, regime

    - If δ < 1 - sqrt(ρ): regime="theorem4.6"
        eta_like = η_J = 1 - sqrt(ρ) - δ
        E = k^2 / (2*min(η_J, sqrt(ρ)/20))^7
        err = E / p
    - Else: regime="conj4.7"
        eta_like = η = 1 - ρ - δ  (must be > 0)
        E = n^{c1} / (ρ^{c2} * η^{c3})
        err = E / p
    """
    if not (0 < k < n):
        raise ValueError("Require 0 < k < n.")
    if min(c1, c2, c3) < 0:
        raise ValueError("Require c1, c2, c3 >= 0.")

    rho = k / n
    johnson = 1.0 - sqrt(rho)

    if delta < johnson:
        eta_j = 1.0 - sqrt(rho) - delta
        if eta_j <= 0:
            raise ValueError(f"η_J must be positive in theorem-4.6 regime; got η_J={eta_j:.4f}.")
        m = min(eta_j, sqrt(rho) / 20.0)
        if m <= 0:
            raise ValueError("Internal: min(η_J, sqrt(ρ)/20) must be positive.")
        E = (n ** 2) / ((2.0 * m) ** 7)
        err = E / p
        return E, err, rho, eta_j, "theorem4.6"
    else:
        eta = 1.0 - rho - delta
        if eta <= 0:
            raise ValueError(
                f"eta = 1 - k/n - delta must be positive in conjecture-4.7 regime; got eta={eta:.4f}."
            )
        E = (n ** c1) / ((rho ** c2) * (eta ** c3))
        err = E / p
        return E, err, rho, eta, "conj4.7"


# ---- Simple stats collector for `counter` ----
class CounterStats:
    def __init__(self):
        self.freq = Counter()
        self.samples = 0
        self.sum = 0.0
        self.min_v = None
        self.max_v = None
        self.over_E = 0

    def add(self, value: int, E: float):
        self.freq[value] += 1
        self.samples += 1
        self.sum += value
        self.min_v = value if self.min_v is None else min(self.min_v, value)
        self.max_v = value if self.max_v is None else max(self.max_v, value)
        if value > E:
            self.over_E += 1

    def mean(self) -> float:
        return self.sum / self.samples if self.samples else 0.0

    def dump_csv(self, path: str):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["counter", "count"])
            for v in sorted(self.freq):
                w.writerow([v, self.freq[v]])

    def plot(self, path: str, show_cdf: bool = False, *, info_text: str = "", E: Optional[float] = None):
        try:
            import matplotlib  # try headless backend first
            try:
                matplotlib.get_backend()
            except Exception:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception as e:
            # ASCII fallback
            print(f"[warn] matplotlib not available; ASCII histogram instead: {e}")
            if info_text:
                print("[ascii-plot] run info:")
                for line in info_text.splitlines():
                    print("  " + line)
            if E is not None:
                print(f"[ascii-plot] E ≈ {E:.6g}")
            total = self.samples or 1
            maxw = 50
            print("[ascii-plot] counter histogram")
            if self.freq:
                maxfreq = max(self.freq.values())
            else:
                maxfreq = 1
            for v in sorted(self.freq):
                bar = "#" * max(1, int(self.freq[v] * maxw / maxfreq))
                print(f"{v:>4}: {bar} ({self.freq[v]})")
            if show_cdf:
                cum = 0
                print("[ascii-plot] CDF")
                for v in sorted(self.freq):
                    cum += self.freq[v]
                    pct = 100.0 * cum / total
                    print(f"{v:>4}: {pct:6.2f}%")
            return

        xs = sorted(self.freq)
        ys = [self.freq[x] for x in xs]

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.bar(xs, ys)
        ax.set_xlabel("counter")
        ax.set_ylabel("frequency")
        ax.set_title("Distribution of counter")

        # Vertical marker for E (if given)
        if E is not None:
            ax.axvline(E, linestyle="--")
            ymax = max(ys) if ys else 1
            ax.text(E, ymax * 1.02, f"E={E:.3g}", rotation=90, va="bottom", ha="center")

        # CDF overlay
        if show_cdf and self.samples > 0:
            cdf = np.cumsum(ys) / float(self.samples)
            ax2 = ax.twinx()
            ax2.plot(xs, cdf, linestyle="-")
            ax2.set_ylabel("CDF")

        # Run info text box
        if info_text:
            ax.text(
                1.02, 0.98, info_text,
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.7", boxstyle="round,pad=0.4"),
            )

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


# ---- Scan a single pair ----
def scan_pair(
    F1: List[int],
    F2: List[int],
    xs: List[int],
    p: int,
    k: int,
    delta: float,
    E: float,
    *,
    require_f1_far: bool = False,
    verbose: bool = False,
    stats: Optional[CounterStats] = None,
) -> int:
    if not (len(F1) == len(F2) == len(xs)):
        raise ValueError("F1, F2, xs must have length n.")
    F1_close = is_delta_close(F1, xs, k, p, delta)
    F2_close = is_delta_close(F2, xs, k, p, delta)

    # Proceed rule
    if require_f1_far:
        proceed = not F1_close
    else:
        proceed = not (F1_close and F2_close)

    if not proceed:
        vprint(verbose, f"[pair] skipped: F1_close={F1_close}, F2_close={F2_close}")
        return 0

    vprint(verbose, f"[pair] proceeding: F1_close={F1_close}, F2_close={F2_close}")

    counter = 0
    for alpha in range(p):
        combo = [(F1[i] + (alpha * F2[i]) % p) % p for i in range(len(xs))]
        if is_delta_close(combo, xs, k, p, delta):
            counter += 1

    if counter > E:
        print("counter > E", counter)
    vprint(verbose, f"[pair] counter={counter}  (E={E:.6g})")

    if stats is not None:
        stats.add(counter, E)

    return counter


# ---- Driver ----
def main():
    ap = argparse.ArgumentParser(
        description="Scan F1,F2 pairs with sharding/limits. Default iterates {2,...,p-1}^n; pass --i-know-what-im-doing for FULL (F_p)^n."
    )
    ap.add_argument("--p", type=int, default=257)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--delta", type=float, default=0.31)

    # Conjecture 4.7 exponents (used when δ ≥ 1 - sqrt(ρ))
    ap.add_argument("--c1", type=int, default=1, help="Exponent on n in Conjecture 4.7.")
    ap.add_argument("--c2", type=int, default=1, help="Exponent on ρ in Conjecture 4.7 (denominator).")
    ap.add_argument("--c3", type=int, default=1, help="Exponent on η in Conjecture 4.7 (denominator).")

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

    # verbosity
    ap.add_argument("-v", "--verbose", action="store_true", help="Enable verbose diagnostics.")

    # stats
    ap.add_argument("--collect-stats", action="store_true", help="Collect histogram of `counter` across pairs.")
    ap.add_argument("--stats-csv", type=str, default=None, help="Write histogram CSV (value,count).")
    ap.add_argument("--stats-plot", type=str, default=None, help="Save a PNG bar chart of the histogram.")
    ap.add_argument("--stats-cdf", action="store_true", help="Overlay empirical CDF on the plot (requires --stats-plot).")

    args = ap.parse_args()

    if (args.p - 1) % args.n != 0:
        raise SystemExit("Require n | (p - 1).")

    xs = root_of_unity_domain(args.p, args.n)

    # E/err from Theorem 4.6 vs Conjecture 4.7
    try:
        E, err, rho, eta_like, regime = compute_E_and_err(
            args.p, args.n, args.k, args.delta, args.c1, args.c2, args.c3
        )
    except ValueError as e:
        raise SystemExit(str(e))

    johnson = 1.0 - sqrt(rho)
    space_desc = "FULL (F_p)^n" if args.i_know_what_im_doing else "{2,...,p-1}^n (scrambled)"
    print(
        f"[info] p={args.p}, n={args.n}, k={args.k}, delta={args.delta}, "
        f"rho={rho:.3f}, Johnson(1-sqrt(rho))={johnson:.3f}"
    )
    if regime == "theorem4.6":
        print(f"[info] regime=Theorem 4.6, eta_J={eta_like:.3f}")
    else:
        print(f"[info] regime=Conjecture 4.7, c1={args.c1}, c2={args.c2}, c3={args.c3}, eta={eta_like:.3f}")
    print(f"[info] E={E:.6g}, err=E/p={err:.6g}")
    print(f"[info] iterating space: {space_desc}")

    # iterator + size
    iter_fn = iterate_vectors_full if args.i_know_what_im_doing else iterate_vectors_sparse
    total_vecs = (args.p ** args.n) if args.i_know_what_im_doing else ((args.p - 2) ** args.n)

    # workload estimate
    total1 = (total_vecs - args.offset1 + args.stride1 - 1) // args.stride1
    total2 = (total_vecs - args.offset2 + args.stride2 - 1) // args.stride2
    eff1 = total1 if args.limit1 is None else min(total1, args.limit1)
    eff2 = total2 if args.limit2 is None else min(total2, args.limit2)
    est_pairs = eff1 * eff2

    if args.max_pairs is None and est_pairs > 1e7 and not args.i_know-what_im_doing:
        sys.exit(
            f"Refusing to launch a scan of ~{est_pairs:.2e} pairs without --max-pairs or --i-know-what-im-doing."
        )

    vprint(args.verbose, f"[info] shard1: offset={args.offset1} stride={args.stride1} limit={args.limit1} (eff ~ {eff1:,})")
    vprint(args.verbose, f"[info] shard2: offset={args.offset2} stride={args.stride2} limit={args.limit2} (eff ~ {eff2:,})")
    vprint(args.verbose, f"[info] estimated pairs: {est_pairs:,}")

    stats = CounterStats() if args.collect_stats else None

    # Build plot info string to pass into the plot (if requested)
    plot_info = build_plot_info(
        p=args.p, n=args.n, k=args.k, delta=args.delta, rho=rho, johnson=johnson,
        regime=regime, eta_like=eta_like, c1=args.c1, c2=args.c2, c3=args.c3,
        E=E, err=err, space_desc=space_desc
    )

    pairs = 0
    for F1 in iter_fn(args.p, args.n, start=args.offset1, step=args.stride1, limit=args.limit1, verbose=args.verbose):
        for F2 in iter_fn(args.p, args.n, start=args.offset2, step=args.stride2, limit=args.limit2, verbose=args.verbose):
            scan_pair(
                F1, F2, xs, args.p, args.k, args.delta, E,
                require_f1_far=args.require_f1_far,
                verbose=args.verbose,
                stats=stats,
            )
            pairs += 1
            if args.max_pairs is not None and pairs >= args.max_pairs:
                print(f"[info] Reached --max-pairs={args.max_pairs}.")
                # finalize stats before returning
                if stats and stats.samples:
                    print(f"[stats] samples={stats.samples:,} mean={stats.mean():.4f} min={stats.min_v} max={stats.max_v} "
                          f"frac(counter>E)={stats.over_E / stats.samples:.6f}")
                    if args.stats_csv:
                        stats.dump_csv(args.stats_csv)
                        print(f"[stats] wrote CSV to {args.stats_csv}")
                    if args.stats_plot:
                        stats.plot(args.stats_plot, show_cdf=args.stats_cdf, info_text=plot_info, E=E)
                        print(f"[stats] saved plot to {args.stats_plot}")
                return

            if args.verbose and (pairs & ((1 << 12) - 1)) == 0:  # every 4096 pairs
                print(f"[progress] processed pairs: {pairs:,}")

    print(f"[info] Completed {pairs} pairs.")

    # finalize stats
    if stats and stats.samples:
        print(f"[stats] samples={stats.samples:,} mean={stats.mean():.4f} min={stats.min_v} max={stats.max_v} "
              f"frac(counter>E)={stats.over_E / stats.samples:.6f}")
        if args.stats_csv:
            stats.dump_csv(args.stats_csv)
            print(f"[stats] wrote CSV to {args.stats_csv}")
        if args.stats_plot:
            stats.plot(args.stats_plot, show_cdf=args.stats_cdf, info_text=plot_info, E=E)
            print(f"[stats] saved plot to {args.stats_plot}")


if __name__ == "__main__":
    main()
