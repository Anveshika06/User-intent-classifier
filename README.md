# Video-MME query-intent router

The router tells a Video-RAG system which evidence a question needs:
- **FRAME:** one moment.
- **CLIP:** a short event.
- **VIDEO:** the whole video.

A question can need more than one. It runs on a CPU in about 1 ms per question, from a model file of about 190 KB, with
no GPU and no LLM call.

| 163 held-out questions | Original system | This router |
|---|---|---|
| Macro-F1 | 0.762 | **0.941** |
| Exact match | 0.436 | **0.896** |
| Extra routes per question | 0.497 | **0.067** |
| Routing regret (cost metric; 0% = perfect) | 97.6% | **33.5%** (24.7% with `--policy min-cost`) |

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s src
python src/main.py          # train (~10 s) and route the 270 Video-MME questions -> artifacts/
python src/evaluate.py      # scores human-verified labels; reports "pending" until you label
```

- **Try one question:** `python src/main.py --ask "What color is the car when it crashes?"`. Leave out the question
  for an interactive prompt.
- **Label Video-MME:** run `python src/evaluate.py --annotate`. Then score the development videos with `--split dev`
  and the test videos with `--split test`.
- **Reproduce all tables and the trade-off figure:** `python src/experiments.py`. The figure needs
  `pip install matplotlib`.
- **If the dataset download fails with 401 or 403:** run `hf auth login`.

## Layout

| File | Purpose |
|---|---|
| `src/router.py` | Features, scoring, routing policies and the Expected Routing Cost metric |
| `src/main.py` | Training, threshold tuning, validation, batch routing, `--ask` |
| `src/evaluate.py` | Blind labelling; scoring with intervals that resample whole videos |
| `src/experiments.py` | Every comparison in the write-up, plus `docs/tradeoffs.png` |
| `src/templates.py` | Training-data generator (239 templates → `src/data/train.csv`) |
| `src/data/` | Training, held-out and seed questions; `annotations.csv` holds your labels |
| `src/test_router.py` | Tests |
| `docs/slides.pptx` | 3-slide summary |
| `docs/report.md` | Write-up: decisions, evidence, trade-offs, references |
| `docs/architecture.drawio`, `docs/architecture.png` | Architecture diagram; open the `.drawio` file in draw.io to edit |
| `docs/tradeoffs.png` | Accuracy–latency–size and coverage–cost figure |
