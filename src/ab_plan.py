"""Simple planning and analysis functions for a future two-arm product A/B test."""

from __future__ import annotations

import math
from statistics import NormalDist


def sample_size_per_arm(baseline: float, mde: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Normal approximation for two independent proportions with equal arms."""
    if not (0 < baseline < 1 and 0 < baseline + mde < 1 and 0 < alpha < 1 and 0 < power < 1):
        raise ValueError("Probabilities must be between zero and one")
    p1, p2 = baseline, baseline + mde
    pooled = (p1 + p2) / 2
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1 - alpha / 2)
    z_power = normal.inv_cdf(power)
    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_power * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return math.ceil(numerator / (mde * mde))


def two_proportion_test(success_a: int, size_a: int, success_b: int, size_b: int) -> dict[str, float]:
    """Two-sided pooled z test and unpooled 95% interval for p_b - p_a."""
    if not (0 <= success_a <= size_a and 0 <= success_b <= size_b and size_a > 0 and size_b > 0):
        raise ValueError("Invalid group counts")
    p_a, p_b = success_a / size_a, success_b / size_b
    pooled = (success_a + success_b) / (size_a + size_b)
    pooled_se = math.sqrt(pooled * (1 - pooled) * (1 / size_a + 1 / size_b))
    if pooled_se == 0:
        raise ValueError("No variation in the binary outcome")
    z = (p_b - p_a) / pooled_se
    p_value = 2 * (1 - NormalDist().cdf(abs(z)))
    unpooled_se = math.sqrt(p_a * (1 - p_a) / size_a + p_b * (1 - p_b) / size_b)
    margin = NormalDist().inv_cdf(0.975) * unpooled_se
    return {
        "rate_a": p_a,
        "rate_b": p_b,
        "difference": p_b - p_a,
        "p_value_two_sided": p_value,
        "ci_95_low": p_b - p_a - margin,
        "ci_95_high": p_b - p_a + margin,
    }


if __name__ == "__main__":
    print(f"Planning example: {sample_size_per_arm(0.60, 0.08)} users per arm")
    print("Synthetic count example:", two_proportion_test(600, 1000, 680, 1000))
