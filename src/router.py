"""FRAME / CLIP / VIDEO router: features, scoring, routing policies and the Expected Routing Cost metric.

Expected Routing Cost (ERC) prices a routing decision in one unit. For required routes G and predicted routes P:
    cost = sum of c_k over k in P   +   miss_cost if G is not a subset of P   +   router_ms / ms_per_unit
Routing regret = ERC / oracle - 1, where the oracle routes exactly G (0% = perfect). Default costs are
placeholders in relative units (each step frame -> clip -> video holds ~4x more evidence); pass measured ones.
"""
from itertools import product

import numpy as np
from scipy.sparse import hstack

LABELS = ("FRAME", "CLIP", "VIDEO")
ROUTES = {"FRAME": "frame_index", "CLIP": "clip_index", "VIDEO": "video_index"}
DEFAULT_COSTS = (1.0, 4.0, 16.0)   # frame_index, clip_index, video_index
DEFAULT_MISS = 32.0                # cost of answering without the required evidence (must be >= sum of costs)
DEFAULT_MS_PER_UNIT = 100.0        # downstream compute one unit represents, to price router latency
ROUTE_SETS = np.array([s for s in product((0, 1), repeat=3) if any(s)])  # the 7 non-empty route sets


def normalize(text):
    return " ".join(str(text).lower().split())


def featurize(texts, bundle):
    texts = [normalize(t) for t in texts]
    parts = [bundle["word"].transform(texts)]
    if bundle.get("char") is not None:
        parts.append(bundle["char"].transform(texts))
    return hstack(parts, format="csr")


def scores(texts, bundle):
    """Probability that each question needs FRAME, CLIP and VIDEO evidence (n x 3)."""
    return bundle["model"].predict_proba(featurize(texts, bundle))


def decide(probabilities, thresholds=(0.5, 0.5, 0.5)):
    """Every label over its own threshold; never empty (falls back to the nearest miss)."""
    p, t = np.asarray(probabilities, dtype=float), np.asarray(thresholds, dtype=float)
    chosen = [label for label, hit in zip(LABELS, p >= t) if hit]
    return chosen or [LABELS[int(np.argmax(p - t))]]


def predict_sets(probabilities, bundle):
    """0/1 route sets (n x 3) under the bundle's policy: per-class thresholds (default) or minimum cost."""
    p = np.atleast_2d(np.asarray(probabilities, dtype=float))
    if bundle.get("policy") == "min_cost":
        return min_cost_routes(calibrate(bundle["calibrators"], p), bundle["route_costs"], bundle["miss_cost"])
    return np.array([[int(label in decide(row, bundle["thresholds"])) for label in LABELS] for row in p])


def route(question, bundle):
    """Routing decision for one question, as returned to the video system."""
    p = scores([question], bundle)[0]
    labels = [label for label, on in zip(LABELS, predict_sets(p, bundle)[0]) if on]
    return {"intent": labels, "routes": [ROUTES[label] for label in labels],
            "scores": {label: round(float(s), 3) for label, s in zip(LABELS, p)},
            "policy": bundle.get("policy", "thresholds"),
            "low_confidence": bool(np.all(p < np.asarray(bundle["thresholds"])))}


# ---------------------------------------------------------------- Expected Routing Cost
def check(costs, miss):
    if miss < sum(costs):
        raise ValueError(f"miss cost {miss} must be >= total route cost {sum(costs)}: otherwise skipping "
                         "required evidence looks cheaper than retrieving it")


def per_question_cost(y_true, y_pred, costs=DEFAULT_COSTS, miss=DEFAULT_MISS, router_ms=0.0,
                      ms_per_unit=DEFAULT_MS_PER_UNIT):
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    missed = ~np.all(y_pred >= y_true, axis=1)
    return y_pred @ np.asarray(costs, dtype=float) + miss * missed + router_ms / ms_per_unit


def expected_routing_cost(y_true, y_pred, costs=DEFAULT_COSTS, miss=DEFAULT_MISS, router_ms=0.0,
                          ms_per_unit=DEFAULT_MS_PER_UNIT):
    erc = float(per_question_cost(y_true, y_pred, costs, miss, router_ms, ms_per_unit).mean())
    oracle = float((np.asarray(y_true, dtype=int) @ np.asarray(costs, dtype=float)).mean())
    return {"erc": round(erc, 3), "oracle_cost": round(oracle, 3), "regret_pct": round(100 * (erc / oracle - 1), 1)}


def fit_calibrators(scores_, y):
    """Per-route isotonic calibration, fitted on out-of-template scores so it reflects unseen phrasings."""
    from sklearn.isotonic import IsotonicRegression
    y = np.asarray(y, dtype=int)
    return [IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(scores_[:, k], y[:, k])
            for k in range(y.shape[1])]


def calibrate(calibrators, scores_):
    scores_ = np.asarray(scores_, dtype=float)
    return np.column_stack([cal.predict(scores_[:, k]) for k, cal in enumerate(calibrators)])


def min_cost_routes(probabilities, costs=DEFAULT_COSTS, miss=DEFAULT_MISS):
    """Bayes rule: the route set S with the lowest expected cost, sum_{k in S} c_k + miss * P(a needed route is
    outside S), treating calibrated route probabilities as independent."""
    p = np.clip(np.asarray(probabilities, dtype=float), 0, 1)
    covered = np.where(ROUTE_SETS[None, :, :] == 1, 1.0, 1 - p[:, None, :]).prod(axis=2)
    return ROUTE_SETS[np.argmin(ROUTE_SETS @ np.asarray(costs, dtype=float) + miss * (1 - covered), axis=1)]
