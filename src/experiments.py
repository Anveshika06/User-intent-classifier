"""Every comparison in docs/report.md and docs/slides.pptx, plus docs/tradeoffs.png.

    python src/experiments.py        # ~1-2 minutes on a laptop CPU; writes artifacts/experiments.json

Designs are scored three ways: out-of-template CV on the training data (5 seeded splits x 5 folds),
the hand-written held-out set (src/data/heldout.csv) and the original seed questions (src/data/seeds.csv).
"""
import json
import re
import tempfile
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.stats import binomtest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

import main as M
import router as R

warnings.filterwarnings("ignore")
COLS = M.COLS
TRAIN, HELD, SEEDS = (pd.read_csv(M.DATA / f) for f in ("train.csv", "heldout.csv", "seeds.csv"))
Y, YH, YS = (d[COLS].to_numpy() for d in (TRAIN, HELD, SEEDS))
CHOSEN, ORIGINAL = "Word + char TF-IDF (chosen)", "Original system (130 seeds)"
ORIGINAL_CUES = [  # the original system's seven cue flags, verbatim
    r"\b(color|colour|shirt|wearing|visible|logo|sign|text|object|background)\b",
    r"\b(run|runs|running|jump|jumps|open|opens|fall|falls|enter|enters|move|moving)\b",
    r"\b(before|after|then|next|while|during|until|when|first|last)\b",
    r"\b(entire|whole|overall|throughout|main idea|summary|across the video)\b",
    r"\b(how many|how often|number of times|compare|difference|repeated)\b",
    r"\b(say|says|said|speaking|audio|sound|music|hear)\b", r"\b\d{1,2}:\d{2}(?::\d{2})?\b"]
LEXICON = [  # the 17-group hand-written lexicon, tested and dropped

    r"\b(colou?rs?|red|blue|green|yellow|black|white|orange|purple|pink|gr[ae]y|brown|silver|gold(en)?)\b",
    r"\b(wear(s|ing)?|wore|dressed|outfit|uniform|costume|shirt|jacket|hat|cap|dress|shoes|glasses|hair|"
    r"appearance|looks? like|shape|material|made of|expression|logo|brand)\b",
    r"\b(written|writing|text|title|caption|subtitles?|label|sign|banner|poster|headline|menu|scoreboard|"
    r"screen|slide|whiteboard|printed|displayed|jersey|license plate)\b",
    r"\b(left|right|behind|in front of|next to|beside|above|below|under(neath)?|on top of|between|near|"
    r"background|foreground|located|position(ed)?)\b",
    r"\b(at (this|that|the) moment|in (this|the) (scene|shot|frame|image|picture|photo)|close-up|opening shot|"
    r"last shot|final frame)\b",
    r"\b(do|does|did|doing|happen(s|ed|ing)?|react(s|ion)?|respond(s)?|perform(s|ed)?|action|move(s|d)?|"
    r"moving|try(ing)? to|attempt)\b",
    r"\b(before|after|then|next|while|during|until|as soon as|right after|just before|immediately|when|once)\b",
    r"\b(why|cause[sd]?|reason|because|leads? to|led to|result(s|ed)? in|due to|what makes)\b",
    r"\b(whole|entire|overall|throughout|across|over the course|in total|in general|main(ly)?|primar(y|ily)|"
    r"central|overarching)\b",
    r"\b(topic|theme|genre|purpose|message|summar(y|ize|izes)|takeaway|conclu(de|des|sion)|lesson|moral|"
    r"argument|tone|mood|attitude|opinion|audience|significance|relationship|inferr?ed|suggests?)\b",
    r"\b(how many|number of|count|how often|times|total)\b",
    r"\b(repeated(ly)?|again|each time|every time|recurr(ing|ent|s)|most (often|frequently|common)|every)\b",
    r"\b(first|last|final(ly)?|beginning|end(ing|s)?|start(s|ing)?|opening|closing|order|sequence|later)\b",
    r"\b(chang(e|es|ed|ing)|evolv(e|es|ed|ing)|develop(s|ed)?|progress(es|ed)?|shift(s|ed)?|differ(s|ence)?|"
    r"compare[sd]?|contrast)\b",
    r"\b(says?|said|mention(s|ed)?|talk(s|ed|ing)?|speak(s|ing)?|narrat(or|es|ion)|voice|explain(s|ed)?|"
    r"ask(s|ed)?|tells?|told|interview(er|ee)?|hear(d)?|sound|music|audio)\b",
    r"\b(which of the following|which statement|which of these|true|correct|incorrect|not (mentioned|shown))\b",
    r"\b\d{1,2}:\d{2}(:\d{2})?\b",
]


def flags(texts, patterns, scale):
    rows = [[float(bool(re.search(p, str(t).lower()))) * scale for p in patterns] for t in texts]
    return csr_matrix(np.asarray(rows, dtype=np.float32).reshape(len(texts), len(patterns)))


class Variant:
    def __init__(self, word=True, char=False, cues=None, cue_scale=0.5, powerset=False, original=False, kind="lr",
                 fixed=None):
        self.word, self.char, self.cues, self.cue_scale = word, char, cues, cue_scale
        self.powerset, self.original, self.kind, self.fixed = powerset, original, kind, fixed

    def fit(self, questions, y):
        texts = [R.normalize(q) for q in questions]
        if self.kind == "fixed":  # no learning: the same route set for every question
            return self
        if self.kind == "nb":  # multinomial Naive Bayes on word 1-2-gram counts
            self.vec, self.cvec = CountVectorizer(ngram_range=(1, 2)).fit(texts), None
            self.model = OneVsRestClassifier(MultinomialNB(alpha=0.5)).fit(self._x(texts), y)
            return self
        if self.original:  # the first version exactly: capped word TF-IDF + raw flags, balanced classes
            self.vec = TfidfVectorizer(ngram_range=(1, 2), max_features=4000).fit(texts)
            self.model = OneVsRestClassifier(LogisticRegression(max_iter=500, class_weight="balanced")).fit(self._x(texts), y)
            return self
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32).fit(texts) if self.word else None
        self.cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                                    dtype=np.float32).fit(texts) if self.char else None
        if self.kind == "svm":  # linear SVM, Platt-calibrated so thresholds and the cost rule get probabilities
            self.model = OneVsRestClassifier(CalibratedClassifierCV(estimator=LinearSVC(C=0.5), cv=3, method="sigmoid"))
            self.model.fit(self._x(texts), y)
            return self
        if self.kind == "knn":
            self.model = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=15, metric="cosine", weights="distance"))
            self.model.fit(self._x(texts), y)
            return self
        if self.powerset:
            self.combos = sorted({tuple(r) for r in np.asarray(y)})
            self.model = LogisticRegression(C=8.0, max_iter=5000).fit(self._x(texts), [self.combos.index(tuple(r)) for r in np.asarray(y)])
        else:
            self.model = OneVsRestClassifier(LogisticRegression(C=8.0, max_iter=3000)).fit(self._x(texts), y)
        return self

    def _x(self, texts):
        if self.original:
            return hstack([self.vec.transform(texts), flags(texts, ORIGINAL_CUES, 1.0)], format="csr")
        parts = [v.transform(texts) for v in (self.vec, self.cvec) if v is not None]
        return hstack(parts + ([flags(texts, self.cues, self.cue_scale)] if self.cues else []), format="csr")

    def scores(self, questions):
        if self.kind == "fixed":
            return np.tile(np.asarray(self.fixed, dtype=float), (len(questions), 1))
        x = self._x([R.normalize(q) for q in questions])
        if not self.powerset:
            return self.model.predict_proba(x)
        p = np.zeros((x.shape[0], len(self.combos)))
        p[:, self.model.classes_] = self.model.predict_proba(x)
        return p @ np.array(self.combos)


MAJORITY = tuple(int(v) for v in pd.Series([tuple(r) for r in Y]).mode()[0])  # most common label set in training
BASELINES = ["No router: all three indexes", "No router: whole video only", "No router: clips only",
             "Majority label set", "Naive Bayes (word n-grams)", "k-NN (word TF-IDF)", "Linear SVM (word + char TF-IDF)"]
VARIANTS = {
    "No router: all three indexes": (dict(kind="fixed", fixed=(1, 1, 1)), "fixed"),
    "No router: whole video only": (dict(kind="fixed", fixed=(0, 0, 1)), "fixed"),
    "No router: clips only": (dict(kind="fixed", fixed=(0, 1, 0)), "fixed"),
    "Majority label set": (dict(kind="fixed", fixed=MAJORITY), "fixed"),
    "Naive Bayes (word n-grams)": (dict(kind="nb"), "tuned"),
    "k-NN (word TF-IDF)": (dict(kind="knn"), "tuned"),
    "Linear SVM (word + char TF-IDF)": (dict(char=True, kind="svm"), "tuned"),
    ORIGINAL: (dict(original=True), "fixed"),
    "Original features, new data": (dict(original=True), "fixed"),
    "B0: keyword flags only": (dict(word=False, cues=ORIGINAL_CUES, cue_scale=1.0), "tuned"),
    "B2: word TF-IDF": (dict(), "tuned"),
    "Word TF-IDF + lexicon": (dict(cues=LEXICON), "tuned"),
    CHOSEN: (dict(char=True), "tuned"),
    "Word + char + lexicon": (dict(char=True, cues=LEXICON), "tuned"),
    "Label powerset, word + char": (dict(char=True, powerset=True), "tuned"),
}


def out_of_template(kwargs):
    oof = np.zeros((len(TRAIN), 3))
    for seed in range(M.CV_REPEATS):
        for fit_idx, val_idx in M.template_folds(TRAIN.template_id.tolist(), seed):
            oof[val_idx] += Variant(**kwargs).fit(TRAIN.question.iloc[fit_idx], Y[fit_idx]).scores(TRAIN.question.iloc[val_idx])
    return oof / M.CV_REPEATS


def speed_and_size(v, thresholds):
    questions = pd.concat([HELD.question, SEEDS.question]).tolist()
    for q in questions[:20]:
        v.scores([q])
    times = []
    for q in questions:  # best of 3 per question: filters out operating-system interruptions
        runs = []
        for _ in range(3):
            start = time.perf_counter(); R.decide(v.scores([q])[0], thresholds); runs.append((time.perf_counter() - start) * 1000)
        times.append(min(runs))
    with tempfile.TemporaryDirectory() as tmp:
        joblib.dump(v, Path(tmp) / "m.joblib", compress=3)
        kb = (Path(tmp) / "m.joblib").stat().st_size / 1024
    return round(float(np.percentile(times, 50)), 2), round(float(np.percentile(times, 95)), 2), round(kb, 1)


def macro_f1_boot(y_true, y_pred, idx):
    t, p = y_true[idx].astype(bool), y_pred[idx].astype(bool)
    tp, fp, fn = (t & p).sum(1), (~t & p).sum(1), (t & ~p).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan), axis=1)


def main():
    designs = {}
    for name, (kwargs, rule) in VARIANTS.items():
        if name == ORIGINAL:
            v, oof, th = Variant(**kwargs).fit(SEEDS.question, YS), None, [0.40] * 3
        else:
            oof = out_of_template(kwargs)
            th = [0.40] * 3 if rule == "fixed" else M.tune_thresholds(oof, Y)
            v = Variant(**kwargs).fit(TRAIN.question, Y)
        hw, sd = v.scores(HELD.question), v.scores(SEEDS.question)
        d = {"thresholds": th, "hw_scores": hw, "oof": oof, "hw": M.predict_labels(hw, th), "sd": M.predict_labels(sd, th)}
        d["p50_ms"], d["p95_ms"], d["model_kb"] = speed_and_size(v, th)
        if kwargs.get("kind") == "fixed":  # a fixed router has nothing to calibrate: its routes never change
            d["hw_min_cost"], d["sd_min_cost"], d["fixed"] = d["hw"], d["sd"], True
        elif oof is not None:  # same decision rule for every learned design: minimum expected routing cost
            cal = R.fit_calibrators(oof, Y)
            d["calibrate"] = lambda p, cal=cal: R.calibrate(cal, p)
            d["hw_min_cost"] = R.min_cost_routes(R.calibrate(cal, hw))
            d["sd_min_cost"] = R.min_cost_routes(R.calibrate(cal, sd))
        designs[name] = d

    seed_cv = []  # the original system can only be scored on its own seeds by re-splitting them
    for rs in range(200):
        tr, te = train_test_split(SEEDS, test_size=0.25, random_state=rs)
        p = M.predict_labels(Variant(original=True).fit(tr.question, tr[COLS].to_numpy()).scores(te.question), [0.40] * 3)
        seed_cv.append(f1_score(te[COLS].to_numpy(), p, average="macro", zero_division=0))

    table = {}
    for name, d in designs.items():
        row = {"cv_macro_f1": M.routing_metrics(Y, M.predict_labels(d["oof"], d["thresholds"]))["macro_f1"] if d["oof"] is not None else None,
               "handwritten": M.routing_metrics(YH, d["hw"]),
               "seeds_macro_f1": round(float(np.mean(seed_cv)), 3) if name == ORIGINAL else M.routing_metrics(YS, d["sd"])["macro_f1"],
               "p50_ms": d["p50_ms"], "p95_ms": d["p95_ms"], "model_kb": d["model_kb"], "fixed": d.get("fixed", False),
               "standard_baseline": name in BASELINES}
        if "hw_min_cost" in d:
            row["min_cost_regret_pct"] = {"handwritten": R.expected_routing_cost(YH, d["hw_min_cost"])["regret_pct"],
                                          "seeds": R.expected_routing_cost(YS, d["sd_min_cost"])["regret_pct"]}
        table[name] = row

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(YH), size=(2000, len(YH)))
    ch, og = designs[CHOSEN]["hw"], designs[ORIGINAL]["hw"]
    ci = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    f1_ch, f1_og = macro_f1_boot(YH, ch, idx), macro_f1_boot(YH, og, idx)
    ok_ch, ok_og = np.all(ch == YH, axis=1), np.all(og == YH, axis=1)
    b, c = int(np.sum(ok_ch & ~ok_og)), int(np.sum(ok_og & ~ok_ch))
    headline = {"macro_f1_ci": {"original": ci(f1_og), "chosen": ci(f1_ch), "gain": ci(f1_ch - f1_og)},
                "exact_match_gain_ci": ci(ok_ch[idx].mean(1) - ok_og[idx].mean(1)),
                "mcnemar": {"fixed": b, "broken": c, "p_value": float(binomtest(b, b + c, 0.5).pvalue)},
                "seed_cv_original": [round(float(np.mean(seed_cv)), 3), round(float(np.percentile(seed_cv, 5)), 3),
                                     round(float(np.percentile(seed_cv, 95)), 3)]}
    gaps = {n: ci(f1_ch - macro_f1_boot(YH, d["hw"], idx)) for n, d in designs.items() if n != CHOSEN}
    best = max(BASELINES, key=lambda n: table[n]["handwritten"]["macro_f1"])  # strongest standard baseline
    bp = designs[best]["hw"]; ok_b = np.all(bp == YH, axis=1)
    b2, c2 = int(np.sum(ok_ch & ~ok_b)), int(np.sum(ok_b & ~ok_ch))
    headline["vs_strongest_standard_baseline"] = {
        "baseline": best, "macro_f1_gain_ci": ci(f1_ch - macro_f1_boot(YH, bp, idx)),
        "exact_match_gain_ci": ci(ok_ch[idx].mean(1) - ok_b[idx].mean(1)),
        "mcnemar": {"fixed": b2, "broken": c2, "p_value": float(binomtest(b2, b2 + c2, 0.5).pvalue) if b2 + c2 else 1.0}}

    d = designs[CHOSEN]
    scale95 = next(s for s in np.arange(1.0, 0.09, -0.05)
                   if M.routing_metrics(Y, M.predict_labels(d["oof"], [t * s for t in d["thresholds"]]))["full_query_coverage"] >= 0.95)
    coverage_point = {"scale": round(float(scale95), 2),
                      "handwritten": M.routing_metrics(YH, M.predict_labels(d["hw_scores"], [t * scale95 for t in d["thresholds"]]))}

    oracle = R.expected_routing_cost(YH, YH)["erc"]
    sensitivity = []
    for miss in (24.0, 32.0, 64.0, 128.0, 256.0):
        mc = R.min_cost_routes(d["calibrate"](d["hw_scores"]), miss=miss)
        row = {"miss_cost": miss, **{k: R.expected_routing_cost(YH, p, miss=miss)["regret_pct"] for k, p in (
            ("f1_default", ch), ("min_cost", mc), ("route_everything", np.ones_like(YH)), ("original", og))}}
        best = min(R.expected_routing_cost(YH, p, miss=miss, router_ms=d["p50_ms"])["erc"] for p in (ch, mc))
        row["break_even_ms"] = round((best - oracle) * R.DEFAULT_MS_PER_UNIT)  # a perfect router must be faster than this
        sensitivity.append(row)

    by_cost = sorted(designs, key=lambda n: designs[n]["model_kb"])  # deterministic; latency jitters between runs
    lower = {n: round(float(np.percentile(macro_f1_boot(YH, designs[n]["hw"], idx) - f1_ch, 5)), 3) for n in by_cost}
    se = float(np.std(f1_ch - macro_f1_boot(YH, designs["B2: word TF-IDF"]["hw"], idx)))
    selection = {"lower_bound_vs_chosen": lower,
                 "order": "smallest model first",
                 "selected": {str(delta): next(n for n in by_cost if n == CHOSEN or lower[n] > -delta) for delta in (0.01, 0.02, 0.03)},
                 "mde_163": round(2.8 * se, 3), "mde_270": round(2.8 * se * np.sqrt(163 / 270), 3),
                 "questions_for_mde_0.01": int(np.ceil(163 * (2.8 * se / 0.01) ** 2))}

    out = {"designs": table, "headline": headline, "macro_f1_gap_chosen_minus_design": gaps,
           "coverage_target_point": coverage_point, "miss_cost_sensitivity": sensitivity, "selection": selection,
           "min_cost_policy_chosen": {"handwritten": M.routing_metrics(YH, d["hw_min_cost"]), "seeds": M.routing_metrics(YS, d["sd_min_cost"])}}
    (M.ROOT / "artifacts").mkdir(exist_ok=True)
    (M.ROOT / "artifacts" / "experiments.json").write_text(json.dumps(out, indent=2))
    print_summary(out)
    figure(designs, table, sensitivity, oracle)


def print_summary(out):
    print(f"{'design':30s} {'CV F1':>6s} {'HW F1':>6s} {'exact':>6s} {'extra':>6s} {'seeds':>6s} {'min-cost regret HW|seeds':>25s} {'p50':>5s} {'KB':>6s}")
    for n, r in out["designs"].items():
        h, mc = r["handwritten"], r.get("min_cost_regret_pct")
        mc_txt = f"{mc['handwritten']}% | {mc['seeds']}%" if mc else "-"
        cv = r["cv_macro_f1"] if r["cv_macro_f1"] is not None else float("nan")
        print(f"{n:30s} {cv:6.3f} {h['macro_f1']:6.3f} {h['exact_match']:6.3f} {h['extra_routes_per_query']:6.3f} "
              f"{r['seeds_macro_f1']:6.3f} {mc_txt:>25s} {r['p50_ms']:5.2f} {r['model_kb']:6.1f}")
    print("headline:", json.dumps(out["headline"]))
    print("macro-F1 gap (chosen - design), 95% CI:", json.dumps(out["macro_f1_gap_chosen_minus_design"]))
    print("coverage >= 0.95 point:", json.dumps(out["coverage_target_point"]))
    print("min-cost policy (chosen):", json.dumps(out["min_cost_policy_chosen"]))
    for r in out["miss_cost_sensitivity"]:
        print("  sensitivity:", r)
    print("selection:", json.dumps(out["selection"]))


def figure(designs, table, sensitivity, oracle):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker
    except ImportError:
        print("matplotlib not installed: skipped docs/tradeoffs.png (pip install matplotlib)")
        return
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(15.5, 6.2))
    pts = {n: (r["p95_ms"], r["handwritten"]["macro_f1"], r["model_kb"]) for n, r in table.items() if not r["fixed"]}
    frontier = sorted([n for n, (x, y, _) in pts.items() if not any(x2 <= x and y2 >= y and (x2, y2) != (x, y)
                                                                     for x2, y2, _ in pts.values())], key=lambda n: pts[n][0])
    a.step([pts[n][0] for n in frontier], [pts[n][1] for n in frontier], where="post", color="#38A169", lw=1.4,
           label="Pareto frontier (faster and more accurate)")
    for n, (x, y, kb) in pts.items():
        color = "#DD6B20" if n == CHOSEN else ("#38A169" if n in frontier else "#A0AEC0")
        a.scatter(x, y, s=30 + 1.6 * kb, color=color, alpha=0.95 if n == CHOSEN else 0.6, edgecolor="#2D3748",
                  zorder=5 if n == CHOSEN else 3)
        offset = {CHOSEN: (-34, 20), "Label powerset, word + char": (16, 10), "B2: word TF-IDF": (-44, 16),
                  "Linear SVM (word + char TF-IDF)": (-70, -26), "Naive Bayes (word n-grams)": (-128, -4),
                  "k-NN (word TF-IDF)": (-40, -24),
                  "Word TF-IDF + lexicon": (-112, -18), "Word + char + lexicon": (18, -16),
                  "Original features, new data": (14, -16), ORIGINAL: (14, 6)}.get(n, (10, 4))
        a.annotate(n.replace(" (130 seeds)", "").replace(" TF-IDF (chosen)", " (chosen)"), (x, y), xytext=offset,
                   textcoords="offset points", fontsize=8.5, fontweight="bold" if n == CHOSEN else "normal",
                   arrowprops=dict(arrowstyle="-", color="#A0AEC0", lw=0.8, shrinkA=0, shrinkB=4))
    be = next(r["break_even_ms"] for r in sensitivity if r["miss_cost"] == R.DEFAULT_MISS)
    for yy, txt, col in ((0.715, "latency budget: 10 ms \u2192", "#4A5568"),
                         (0.68, f"break-even for a perfect router: {be} ms \u2192", "#2B6CB0"),
                         (0.645, "typical LLM API call: 300\u20132,000 ms \u2192", "#9B2C2C")):
        a.text(0.98, yy, txt, transform=a.get_yaxis_transform(), ha="right", fontsize=8.5, color=col)
    a.set_xscale("log"); a.set_xlim(0.25, 8.0); a.set_ylim(0.6, 1.0)
    a.set_xticks([0.3, 0.5, 1, 2, 4, 8]); a.set_xticklabels(["0.3", "0.5", "1", "2", "4", "8"])
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlabel("p95 single-question latency (ms, log scale)"); a.set_ylabel("macro-F1, hand-written set")
    a.set_title("Accuracy vs latency (bubble area = model size)", fontsize=11, fontweight="bold")
    a.legend(fontsize=8.5, loc="upper left"); a.grid(alpha=0.25, which="both")

    d = designs[CHOSEN]
    for mult, ls, y_lab in ((1.1, ":", 0.80), (1.25, "--", 0.78), (1.5, "-.", 0.76)):
        xs = np.linspace(0, 22, 50)
        b.plot(xs, 1 - (mult * oracle - xs) / R.DEFAULT_MISS, color="#A0AEC0", ls=ls, lw=1)
        b.text(mult * oracle - R.DEFAULT_MISS * (1 - y_lab) + 0.3, y_lab - 0.004, f"regret {100 * (mult - 1):.0f}%", fontsize=7.5, color="#4A5568")
    sweep = []
    for s in np.round(np.arange(0.10, 1.51, 0.05), 2):
        p = M.predict_labels(d["hw_scores"], [t * s for t in d["thresholds"]])
        sweep.append(((p @ np.array(R.DEFAULT_COSTS)).mean(), np.all(p >= YH, axis=1).mean()))
    sweep = np.array(sweep)
    b.plot(sweep[:, 0], sweep[:, 1], color="#DD6B20", lw=1.6, label="chosen design: threshold sweep")
    marks = (("default (F1 thresholds)", d["hw"], "o", (10, -14)), ("min-cost rule", d["hw_min_cost"], "D", (-78, 10)),
             ("original system", designs[ORIGINAL]["hw"], "X", (8, 6)), ("route everything", np.ones_like(YH), "X", (-62, 6)),
             ("oracle", YH, "X", (8, 6)))
    for text, p, mk, off in marks:
        x, y = (p @ np.array(R.DEFAULT_COSTS)).mean(), np.all(p >= YH, axis=1).mean()
        b.scatter(x, y, marker=mk, s=60, color="#DD6B20" if mk != "X" else "#718096", edgecolor="#2D3748", zorder=4)
        b.annotate(text, (x, y), xytext=off, textcoords="offset points", fontsize=8.5)
    b.set_xlim(0, 22); b.set_ylim(0.75, 1.02)
    b.set_xlabel("retrieval cost per question (units: frame 1, clip 4, video 16)"); b.set_ylabel("full-query coverage")
    b.set_title("Coverage vs retrieval cost (grey lines: equal routing regret, miss cost 32)", fontsize=11, fontweight="bold")
    b.legend(fontsize=8.5, loc="lower right"); b.grid(alpha=0.25)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.92, wspace=0.16)
    fig.savefig(M.ROOT / "docs" / "tradeoffs.png", dpi=140, facecolor="white")
    print("Saved docs/tradeoffs.png")


if __name__ == "__main__":
    main()
