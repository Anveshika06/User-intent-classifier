# Query Intent Router

A **CPU-only, multi-label query classifier** that determines whether a video question needs **FRAME**, **CLIP**, **VIDEO**, or a combination of retrieval scopes.

The goal is to make Video-RAG retrieval more efficient by selecting relevant evidence indexes without an LLM call at inference time.

> **Scope:** This project implements the query-intent router, not a complete video question-answering system.

## How It Works

| Intent | Evidence scope | Retrieval route |
|---|---|---|
| FRAME | A specific visual moment | `frame_index` |
| CLIP | An event or short sequence | `clip_index` |
| VIDEO | The broader video or timeline | `video_index` |

A question can activate multiple routes.

### Architecture

**Offline training**

1. Generate 1,708 labeled questions from 239 templates.
2. Extract word and character TF-IDF features.
3. Train three independent Logistic Regression classifiers.
4. Tune class-specific thresholds using template-grouped validation.
5. Save the trained model.

**Online inference**

Question → TF-IDF features → Three classifier scores → Threshold decisions → Selected retrieval indexes

The router uses a fallback for uncertain questions and supports an optional cost-aware routing policy.

For the full component diagram and design decisions, see [docs/workflow.md](docs/workflow.md).

## Why This Architecture?

The initial prototype used word TF-IDF and seven hand-written keyword features. Although fast, it could overreact to individual words and select unnecessary retrieval routes.

The updated architecture uses:

- **Word + character TF-IDF** to learn phrasing patterns without relying on hard-coded keywords.
- **Three independent classifiers** to support multi-label predictions.
- **Template-generated training data** to expand coverage beyond a small seed set.
- **Per-class thresholds** to control missed evidence versus unnecessary retrieval.
- **CPU-only inference** to meet strict latency and model-size constraints.

The design was informed by AdaVideoRAG, VideoStir, and Adaptive-RAG.

## Installation

Requires Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
```

## Run

Train and route questions:

```bash
python src/main.py
```

Run model comparisons:

```bash
python src/experiments.py
```

Annotate questions and evaluate routing quality:

```bash
python src/evaluate.py --annotate
python src/evaluate.py
```

The root-level `run.yaml` contains the submission commands.

Generated models, predictions, and evaluation reports are saved in `artifacts/`.

