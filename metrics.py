"""Statistics used by evaluate.py. No I/O, so each function is testable in isolation."""
import math


def percentile(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a proportion. Returns (low, high).

    Wald is unusable at n=40: it produces bounds outside [0, 1] and zero width at p=1.
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar. b = A right and B wrong, c = the reverse.

    Concordant items are excluded. Exact binomial rather than chi-square, which is
    unreliable below roughly 25 discordant pairs.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def discordant_pairs(correct_a, correct_b):
    """Count (b, c) from two aligned lists of per-item booleans."""
    if len(correct_a) != len(correct_b):
        raise ValueError("paired comparison needs equal-length, aligned lists")
    b = sum(1 for a, x in zip(correct_a, correct_b) if a and not x)
    c = sum(1 for a, x in zip(correct_a, correct_b) if x and not a)
    return b, c


def per_class_prf(confusion, categories):
    """Precision, recall and F1 per category from confusion[gold][predicted]."""
    out = {}
    for cat in categories:
        # INVALID counts toward gold_total but not pred_total: an unparseable response
        # failed to retrieve the item, but is not a claim that it belongs to a class.
        gold_total = sum(confusion[cat].values())
        pred_total = sum(confusion[g].get(cat, 0) for g in categories)
        tp = confusion[cat].get(cat, 0)
        precision = tp / pred_total if pred_total else 0.0
        recall = tp / gold_total if gold_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[cat] = {"precision": precision, "recall": recall, "f1": f1,
                    "support": gold_total, "predicted": pred_total}
    return out


def macro_f1(prf):
    return sum(v["f1"] for v in prf.values()) / len(prf) if prf else 0.0


def mean_and_sd(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)
