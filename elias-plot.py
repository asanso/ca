import numpy as np
import math
import matplotlib.pyplot as plt

# Parameters
q = 13
rho_min, rho_max = 0.01, 0.5
num_points = 400

def H_q(delta, q):
    if delta <= 0.0 or delta >= 1.0:
        return 0.0
    return (
        delta * math.log(q - 1, q)
        - delta * math.log(delta,   q)
        - (1 - delta) * math.log(1 - delta, q)
    )

def delta_elias_for_rho(rho, q, tol=1e-8):
    lo, hi = 0.0, 1.0 - 1.0 / q
    target = 1.0 - rho          # solve H_q(delta) = 1 - rho
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = H_q(mid, q)
        if val > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

rhos = np.linspace(rho_min, rho_max, num_points)
delta_conj = 1.0 - rhos                     # black line δ = 1 − ρ
delta_elias = np.array([delta_elias_for_rho(r, q) for r in rhos])

fig, ax = plt.subplots(figsize=(6, 6))

# Regions
ax.fill_between(
    rhos, 0, delta_elias,
    color="lightgreen", alpha=0.8,
    label="Below Elias capacity (small lists possible)",
)

ax.fill_between(
    rhos, delta_elias, delta_conj,
    color="lightcoral", alpha=0.6,
    label="Above Elias, below δ = 1 − ρ (Elias/entropy explosion region)",
)

ax.fill_between(
    rhos, delta_conj, 1,
    color="lightgray", alpha=0.8,
    label="Trivial interpolation regime  (δ ≥ 1 − ρ)",
)

# Curves
ax.plot(
    rhos, delta_conj,
    color="black", linewidth=2,
    label='"Up to capacity" line  δ = 1 − ρ',
)
ax.plot(
    rhos, delta_elias,
    linestyle="--", color="gold", linewidth=2,
    label="Elias list-decoding capacity  ρ = 1 − H_q(δ)",
)

# Toy RS example point: (rho, delta) = (5/12, 0.5)
rho_example = 5 / 12
delta_example = 0.5
ax.scatter([rho_example], [delta_example], color="blue", s=60, zorder=5)
ax.annotate(
    "toy RS example",
    xy=(rho_example, delta_example),
    xytext=(rho_example + 0.02, delta_example - 0.02),
    arrowprops=dict(arrowstyle="->", lw=1),
    fontsize=9,
)

ax.set_xlabel("Rate ρ")
ax.set_ylabel("Error fraction δ")
ax.set_xlim(rho_min, rho_max)
ax.set_ylim(0, 1.0)
ax.set_title('"Up to capacity" folklore vs Elias and trivial regimes  (q = 13)')
ax.legend(loc="lower left", fontsize=8)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()
