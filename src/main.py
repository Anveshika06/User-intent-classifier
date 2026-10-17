"""Train the FRAME / CLIP / VIDEO router and route the Video-MME questions.

    python src/main.py                                    # train (~10 s), validate, route all questions -> artifacts/
    python src/main.py --ask "What color is the car when it crashes?"   # one question (no argument: interactive)
    python src/main.py --policy min-cost                  # cost-aware routing; --route-costs F C V --miss-cost M
    python src/main.py --threshold-scale 0.3              # higher coverage, more retrievals
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier

import router as R

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "data"
MODEL = ROOT / "artifacts" / "intent_model.joblib"
HF_URL = "hf://datasets/aintropy-ai/video-mme-sample10/qa/test.jsonl"
COLS = ["frame", "clip", "video"]
ID_DTYPES = {"video_id": str, "question_id": str}
CONFIG = {"word_ngrams": (1, 2), "char_ngrams": (2, 5), "char_min_df": 2, "C": 8.0}  # chosen in docs/report.md
CV_FOLDS, CV_REPEATS = 5, 5


def fit_bundle(questions, y, cfg=CONFIG):
    texts = [R.normalize(q) for q in questions]
    word = TfidfVectorizer(ngram_range=cfg["word_ngrams"], sublinear_tf=True, dtype=np.float32).fit(texts)
    char = None
    if cfg.get("char_ngrams"):
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=cfg["char_ngrams"], min_df=cfg["char_min_df"],
                               sublinear_tf=True, dtype=np.float32).fit(texts)
        char.stop_words_ = None  # introspection-only attribute; large when pickled
    word.stop_words_ = None
    bundle = {"word": word, "char": char, "thresholds": [0.5, 0.5, 0.5], "config": dict(cfg)}
    bundle["model"] = OneVsRestClassifier(LogisticRegression(C=cfg["C"], max_iter=3000)).fit(
        R.featurize(texts, bundle), np.asarray(y))
    return bundle


def template_folds(template_ids, seed, folds=CV_FOLDS):
    """Whole templates assigned to folds by a seeded permutation, so every machine gets the same split."""
    ids = sorted(set(template_ids))
    fold_of = {ids[i]: n % folds for n, i in enumerate(np.random.RandomState(seed).permutation(len(ids)))}
    assignment = np.array([fold_of[t] for t in template_ids])
    return [(np.where(assignment != k)[0], np.where(assignment == k)[0]) for k in range(folds)]


def out_of_template_scores(train, cfg=CONFIG, repeats=CV_REPEATS):
    """Scores from models that never saw a question's template, averaged over `repeats` seeded splits."""
    oof = np.zeros((len(train), len(COLS)))
    for seed in range(repeats):
        for fit_idx, val_idx in template_folds(train.template_id.tolist(), seed):
            bundle = fit_bundle(train.question.iloc[fit_idx], train[COLS].to_numpy()[fit_idx], cfg)
            oof[val_idx] += R.scores(train.question.iloc[val_idx].tolist(), bundle)
    return oof / repeats


def tune_thresholds(probabilities, y_true, grid=np.arange(0.20, 0.81, 0.02), tolerance=0.005):
    """Per class: the middle of the threshold range whose F1 is within `tolerance` of the best (stable)."""
    best = []
    for k in range(y_true.shape[1]):
        f1s = np.array([f1_score(y_true[:, k], probabilities[:, k] >= t, zero_division=0) for t in grid])
        best.append(round(float(np.median(grid[f1s >= f1s.max() - tolerance])), 2))
    return best


def predict_labels(probabilities, thresholds):
    return np.array([[int(label in R.decide(p, thresholds)) for label in R.LABELS] for p in probabilities])


def routing_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    return {"macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=np.nan)), 3),
            "exact_match": round(float(np.all(y_true == y_pred, axis=1).mean()), 3),
            "full_query_coverage": round(float(np.all(y_pred >= y_true, axis=1).mean()), 3),
            "extra_routes_per_query": round(float(((y_pred == 1) & (y_true == 0)).sum(axis=1).mean()), 3),
            "routing_regret_pct": R.expected_routing_cost(y_true, y_pred)["regret_pct"],
            "questions": int(len(y_true))}


def train(cfg=CONFIG, threshold_scale=1.0, policy="f1", route_costs=None, miss_cost=None, verbose=True):
    train_df = pd.read_csv(DATA / "train.csv")
    y = train_df[COLS].to_numpy()
    oof = out_of_template_scores(train_df, cfg)
    bundle = fit_bundle(train_df.question, y, cfg)
    bundle["thresholds"] = [round(t * threshold_scale, 3) for t in tune_thresholds(oof, y)]
    oof_pred = predict_labels(oof, bundle["thresholds"])
    if policy == "min-cost" or route_costs is not None or miss_cost is not None:
        costs, miss = tuple(route_costs or R.DEFAULT_COSTS), R.DEFAULT_MISS if miss_cost is None else miss_cost
        R.check(costs, miss)
        bundle.update(policy="min_cost", calibrators=R.fit_calibrators(oof, y), route_costs=costs, miss_cost=miss)
        oof_pred = np.zeros_like(y)  # cross-fitted calibration: never score on the calibration data itself
        for fit_idx, val_idx in template_folds(train_df.template_id.tolist(), seed=99):
            cal = R.fit_calibrators(oof[fit_idx], y[fit_idx])
            oof_pred[val_idx] = R.min_cost_routes(R.calibrate(cal, oof[val_idx]), costs, miss)
    report = {"policy": bundle.get("policy", "thresholds"), "thresholds": dict(zip(R.LABELS, bundle["thresholds"])),
              "out_of_template_cv": routing_metrics(y, oof_pred)}
    for name, file in (("heldout_handwritten", "heldout.csv"), ("original_seed_examples", "seeds.csv")):
        test = pd.read_csv(DATA / file)
        report[name] = routing_metrics(test[COLS].to_numpy(), R.predict_sets(R.scores(test.question.tolist(), bundle), bundle))
    if verbose:
        print("Validation on authored questions (not Video-MME ground truth):")
        for name, values in report.items():
            print(f"  {name}: {values}")
    return bundle, report


def load_questions(source):
    if source == "auto":
        local = ROOT / "qa" / "test.jsonl"
        source = str(local) if local.exists() else HF_URL
    print(f"Loading questions from: {source}")
    try:
        data = pd.read_json(source, lines=True, dtype=ID_DTYPES)  # keep IDs like "001" as text
    except Exception:
        if source != HF_URL:
            raise
        from huggingface_hub import hf_hub_download  # fallback if fsspec's hf:// handler differs
        path = hf_hub_download("aintropy-ai/video-mme-sample10", "qa/test.jsonl", repo_type="dataset")
        data = pd.read_json(path, lines=True, dtype=ID_DTYPES)
    missing = {"video_id", "question_id", "question"} - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing: {missing}")
    return data


def latency_ms(questions, bundle):
    for q in questions[:20]:
        R.scores([q], bundle)
    timings = []
    for q in questions:
        start = time.perf_counter()
        R.predict_sets(R.scores([q], bundle), bundle)
        timings.append((time.perf_counter() - start) * 1000)
    return {"p50_single_query_ms": round(float(np.percentile(timings, 50)), 3),
            "p95_single_query_ms": round(float(np.percentile(timings, 95)), 3)}


def show(question, bundle):
    result = R.route(question, bundle)
    print(f"  => {' + '.join(result['routes'])}" + ("   (low confidence)" if result["low_confidence"] else ""))
    for label, threshold in zip(R.LABELS, bundle["thresholds"]):
        score = result["scores"][label]
        print(f"  {label:5s} {score:5.3f} |{'#' * round(score * 30):<30s}| threshold {threshold:.2f}")


def ask(question):
    bundle = joblib.load(MODEL) if MODEL.exists() else train(verbose=False)[0]
    if question:
        return show(question, bundle)
    print("FRAME = one moment, CLIP = short event, VIDEO = whole video. Empty line or q to quit.")
    while (q := input("\n> ").strip()) and q.lower() not in {"q", "quit", "exit"}:
        show(q, bundle)


def run(args):
    qa = load_questions(args.source)
    bundle, report = train(threshold_scale=args.threshold_scale, policy=args.policy, route_costs=args.route_costs,
                           miss_cost=args.miss_cost)
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL, compress=3)
    start = time.perf_counter()
    probabilities = R.scores(qa.question.tolist(), bundle)
    batch_ms = (time.perf_counter() - start) * 1000

    result = qa[[c for c in ["video_id", "question_id", "duration", "domain", "task_type", "question"] if c in qa]].copy()
    for i, name in enumerate(COLS):
        result[f"p_{name}"] = probabilities[:, i].round(3)
    result["intent"] = ["+".join(l for l, on in zip(R.LABELS, row) if on) for row in R.predict_sets(probabilities, bundle)]
    result["route"] = result.intent.map(lambda s: "+".join(R.ROUTES[x] for x in s.split("+")))
    result.to_csv(MODEL.parent / "predictions.csv", index=False)

    labels_path = DATA / "annotations.csv"  # blank, blind sheet for human labels (python src/evaluate.py --annotate)
    if not labels_path.exists():
        sheet = result[["video_id", "question_id", "question"]].copy()
        sheet[COLS] = ""
        sheet["verified"] = 0
        sheet.to_csv(labels_path, index=False)

    summary = {"questions": len(qa), "batch_ms": round(batch_ms, 2), **latency_ms(qa.question.tolist(), bundle),
               "model_kb": round(MODEL.stat().st_size / 1024, 1),
               "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest()[:12], "validation": report}
    (MODEL.parent / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("Run summary:", {k: v for k, v in summary.items() if k != "validation"})
    print("Saved: artifacts/predictions.csv, artifacts/metrics.json, src/data/annotations.csv (if new)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="auto", help="Local JSONL, hf:// URL, or auto")
    parser.add_argument("--ask", nargs="?", const="", metavar="QUESTION",
                        help="Route one question with the trained model (no question: interactive prompt)")
    parser.add_argument("--threshold-scale", type=float, default=1.0, help="Multiply tuned thresholds (0.3 = high coverage)")
    parser.add_argument("--policy", choices=["f1", "min-cost"], default="f1",
                        help="f1: per-class thresholds (default); min-cost: lowest expected routing cost")
    parser.add_argument("--route-costs", type=float, nargs=3, metavar=("FRAME", "CLIP", "VIDEO"),
                        help="Measured retrieval cost of each route (implies --policy min-cost)")
    parser.add_argument("--miss-cost", type=float, help="Cost of answering without the required evidence")
    args = parser.parse_args()
    ask(args.ask) if args.ask is not None else run(args)
