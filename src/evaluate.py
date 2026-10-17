"""Blind human labelling and scoring of FRAME / CLIP / VIDEO routing on Video-MME.

    python src/evaluate.py --annotate --max 30   # label questions (whole videos, fixed random order)
    python src/evaluate.py --split dev           # first third of videos: tuning and error analysis
    python src/evaluate.py --split test          # the rest: the number to report
    python src/evaluate.py                       # everything verified so far
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

import router as C  # Expected Routing Cost lives in router.py

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["frame", "clip", "video"]
RUBRIC = ("f = FRAME  static visual fact at one moment (colour, clothing, on-screen text, object, layout)\n"
          "c = CLIP   action/event in a short segment (what happens, before/after, why, what is said)\n"
          "v = VIDEO  whole-video understanding (topic, purpose, counting/ordering events across the video)")


ANNOTATION_SEED = 13


def video_order(video_ids, seed=ANNOTATION_SEED):
    """Fixed random order of whole videos. Annotation follows it, and its first third is the development split,
    so labelling from the start fills the development videos first."""
    videos = sorted(set(video_ids))
    random.Random(seed).shuffle(videos)
    return videos


def split_videos(video_ids, split):
    order = video_order(video_ids)
    dev = set(order[: len(order) // 3])
    return {"dev": dev, "test": set(order) - dev}.get(split, set(order))


def video_bootstrap(video_ids, y_true, y_pred, route_costs, miss_cost, n=2000, seed=0):
    """95% intervals that resample whole videos, because questions about the same video are correlated."""
    vids = np.asarray(video_ids)
    groups = [np.where(vids == v)[0] for v in np.unique(vids)]
    rng = np.random.default_rng(seed)
    draws = {"macro_f1": [], "exact_match_accuracy": [], "full_query_coverage": [], "routing_regret_pct": []}
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        t, p = y_true[idx], y_pred[idx]
        tp, fp, fn = (t & p).sum(0), ((1 - t) & p).sum(0), (t & (1 - p)).sum(0)
        denom = 2 * tp + fp + fn
        draws["macro_f1"].append(np.nanmean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), np.nan)))
        draws["exact_match_accuracy"].append(np.all(t == p, axis=1).mean())
        draws["full_query_coverage"].append(np.all(p >= t, axis=1).mean())
        draws["routing_regret_pct"].append(C.expected_routing_cost(t, p, route_costs, miss_cost)["regret_pct"])
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in draws.items()}


def annotate(path, limit=None, seed=ANNOTATION_SEED):
    """Blind, resumable labelling. Whole videos come in a fixed random order so any partial
    session is a fair sample rather than the first few videos in the file."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)  # strings in, strings out: IDs keep leading zeros
    videos = video_order(df.video_id.unique(), seed)
    dev = set(videos[: len(videos) // 3])
    rank = {v: i for i, v in enumerate(videos)}
    order = sorted(df.index[df.verified != "1"], key=lambda i: (rank[df.at[i, "video_id"]], i))
    print(RUBRIC)
    count = 0
    for i in order:
        if limit is not None and count >= limit:
            break
        part = "dev" if df.at[i, "video_id"] in dev else "test"
        print(f"\n[{count + 1}] video {df.at[i, 'video_id']} ({part} split): {df.at[i, 'question']}")
        answer = input("Labels (e.g. f, c, v, fc, cv, fv); s=skip, q=quit: ").strip().lower()
        if answer == "q":
            break
        if answer == "s":
            continue
        if not answer or any(char not in "fcv" for char in answer):
            print("Invalid entry; leaving this question unverified.")
            continue
        for label, short in zip(LABELS, "fcv"):
            df.at[i, label] = "1" if short in answer else "0"
        df.at[i, "verified"] = "1"
        df.to_csv(path, index=False)  # safe to stop and resume
        count += 1
    print(f"Verified: {(df.verified == '1').sum()} / {len(df)}")


def evaluate(predictions_path, annotations_path, output_path, route_costs=C.DEFAULT_COSTS, miss_cost=C.DEFAULT_MISS,
             split="all"):
    predicted = pd.read_csv(predictions_path, dtype={"video_id": str, "question_id": str})
    gold = pd.read_csv(annotations_path, dtype={"video_id": str, "question_id": str})
    if not {"verified", *LABELS}.issubset(gold.columns):
        raise ValueError("Annotation CSV must include verified, frame, clip, video columns.")
    for dataset in (predicted, gold):
        if dataset.duplicated(["video_id", "question_id"]).any():
            raise ValueError("Duplicate video_id/question_id pairs; cannot evaluate safely.")

    keep = split_videos(predicted.video_id.unique(), split)  # dev = first third of the annotation order
    predicted = predicted[predicted.video_id.isin(keep)]
    checked = gold[pd.to_numeric(gold["verified"], errors="coerce").eq(1) & gold.video_id.isin(keep)].copy()
    result = {
        "status": "pending_verified_labels",
        "split": split,
        "videos_in_split": len(keep),
        "verified_videos": int(checked.video_id.nunique()),
        "verified_questions": len(checked),
        "total_questions": len(predicted),
        "note": "Unverified model suggestions are NOT benchmark ground truth.",
    }
    if len(checked):
        for label in LABELS:
            checked[label] = pd.to_numeric(checked[label], errors="raise")
            if not checked[label].isin([0, 1]).all():
                raise ValueError(f"Invalid {label} label: expected 0 or 1 for verified rows.")
        merged = predicted.merge(
            checked[["video_id", "question_id", *LABELS]],
            on=["video_id", "question_id"], how="inner", validate="one_to_one"
        )
        if len(merged) != len(checked):
            raise ValueError("Some verified IDs are missing from predictions; regenerate the CSV.")
        y_true = merged[LABELS].to_numpy(dtype=int)
        if (y_true.sum(axis=1) == 0).any():
            raise ValueError("Each verified question needs at least one positive evidence class.")
        y_pred = np.array([
            [int(label.upper() in str(intent).split("+")) for label in LABELS]
            for intent in merged["intent"]
        ], dtype=int)
        tp = np.logical_and(y_true, y_pred).sum(axis=0)
        fn = np.logical_and(y_true, 1 - y_pred).sum(axis=0)
        fp = np.logical_and(1 - y_true, y_pred).sum(axis=0)
        # A query is fully covered if every required retrieval index was activated.
        coverage = np.all(y_pred >= y_true, axis=1)
        result.update({
            "status": "verified_subset" if len(checked) < len(predicted) else "verified_complete",
            # Classes with no gold positives and no predictions are left out of the macro average
            # instead of counting as F1 = 0 (which would penalise perfect predictions on small subsets).
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=np.nan)), 4),
            "micro_f1": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
            "route_recall": round(float(recall_score(y_true, y_pred, average="micro", zero_division=0)), 4),
            "route_precision": round(float(precision_score(y_true, y_pred, average="micro", zero_division=0)), 4),
            "full_query_coverage": round(float(coverage.mean()), 4),
            "exact_match_accuracy": round(float(np.all(y_true == y_pred, axis=1).mean()), 4),
            "extra_routes_per_query": round(float(np.logical_and(y_pred, 1 - y_true).sum(axis=1).mean()), 4),
            # Prices misses, over-routing and retrieval in one unit; 0% regret = perfect routing (src/costs.py).
            "expected_routing_cost": {**C.expected_routing_cost(y_true, y_pred, route_costs, miss_cost),
                                      "route_costs": list(route_costs), "miss_cost": miss_cost},
            "per_class": {
                name.upper(): {
                    "support": int(y_true[:, i].sum()),
                    "precision": round(float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)), 4),
                    "recall": round(float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)), 4),
                    "f1": round(float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)), 4),
                    "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(fn[i]),
                } for i, name in enumerate(LABELS)
            }
        })
    if len(checked) and checked.video_id.nunique() >= 5:
        result["confidence_intervals_95"] = video_bootstrap(merged.video_id, y_true, y_pred, route_costs, miss_cost)
        result["ci_method"] = "bootstrap over whole videos (questions from one video stay together), 2,000 resamples"
    elif len(checked):
        result["ci_method"] = "fewer than 5 verified videos: intervals not computed"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate and evaluate Video-MME intent labels")
    parser.add_argument("--annotate", action="store_true", help="Review the questions interactively")
    parser.add_argument("--max", type=int, help="Maximum questions to review this session")
    parser.add_argument("--predictions", type=Path, default=ROOT / "artifacts/predictions.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "src/data/annotations.csv")
    parser.add_argument("--output", type=Path, help="Default: artifacts/evaluation.json (or evaluation_<split>.json)")
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all",
                        help="dev = first third of videos in annotation order (for tuning); test = the rest")
    parser.add_argument("--route-costs", type=float, nargs=3, default=list(C.DEFAULT_COSTS),
                        metavar=("FRAME", "CLIP", "VIDEO"), help="Retrieval cost of each route (relative units)")
    parser.add_argument("--miss-cost", type=float, default=C.DEFAULT_MISS,
                        help="Cost of answering without the required evidence")
    args = parser.parse_args()
    if args.annotate:
        annotate(args.labels, args.max)
    else:
        output = args.output or ROOT / "artifacts" / ("evaluation.json" if args.split == "all" else f"evaluation_{args.split}.json")
        evaluate(args.predictions, args.labels, output, tuple(args.route_costs), args.miss_cost, args.split)
