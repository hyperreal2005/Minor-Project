# T.R.A.C.E. dashboard

An interactive visualization layer over the completed study: every chart reads the result tables,
nothing is typed in by hand. Built with Streamlit and Plotly.

## Run it

From the repository root, with the project venv:

```
venv\Scripts\python.exe -m pip install -e ".[dashboard]"     # first time only
venv\Scripts\streamlit.exe run dashboard\app.py
```

It opens at http://localhost:8501. The light theme is pinned in `.streamlit/config.toml`, so it
looks the same whatever the laptop's dark-mode setting.

## Pages

| Page | What it shows | What you can change |
|---|---|---|
| The study at a glance | Headline numbers (computed live), design, four take-aways | — |
| Guided tour | Six claims, one interactive chart each — for the presentation | Each step's toggle |
| Model inspector | One unlearned model's report card: every audit against the retrain band | Forget set, method, seed, all metrics |
| Audit validity | Native rules vs calibrated bands; canary scoring; canary memory by degree | Families, guard on/off, metric |
| Agreement explorer | Any two audits, model by model, with Kendall τ | Metrics, forget set, pooled vs within, log axes |
| Methods & difficulty | Method ratios; G by method and forget set; forgetting vs utility; method spread | Audit, statistic, forget set |
| Relearning | Relearning curves (when exported); relearning vs starting accuracy; H3 | Forget set, methods, scaling |
| Forget sets | Memorization distribution and real CIFAR-10 images from each forget set | Forget set |
| Calibration lab | What-ifs: band size, band rule, damage threshold, minimum effect | Sliders for each rule |
| Data & provenance | Every table, downloadable | Table |

## A demo script (about 10 minutes)

1. **The study at a glance** (1 min): read the five numbers left to right.
2. **Guided tour**, steps 1–6 (6 min). Flip the toggle at each step:
   - 1: switch the ranking and the destroyed model jumps from first to last;
   - 2: switch the reference from 0.5 to the retrains;
   - 3: native vs calibrated;
   - 4: guard on;
   - 5: RMIA, then Representation;
   - 6: point at the slope.
3. **Calibration lab → C** (1 min): drag the damage threshold below 4 and above 86 points and
   watch the canary scores collapse; the registered 5 sits in the middle of the plateau.
4. **Model inspector** (1 min): SalUn at mem-high, seed 0. Close to a retrain on RMIA, far on
   representation.
5. **Forget sets** (1 min): mem-high vs canary images.

## Updating the data

The dashboard reads its own committed snapshot in `dashboard/data/` (about 2 MB), because
`results/` is gitignored. After unzipping a new results bundle into the repository root, run:

```
venv\Scripts\python.exe dashboard\prepare_data.py
```

Optional inputs are picked up automatically when present:
- `results/calibration/candidates.parquet` (notebook 06): each retrain's and original's own values.
  This turns on retrain dots in the inspector and the false-positive rates in lab B.
- `results/curves/*.parquet` (notebook 07): the step-by-step relearning curves.
- `data/cifar-10-batches-py`: the gallery thumbnails.

The landing page says which of these are still missing.
