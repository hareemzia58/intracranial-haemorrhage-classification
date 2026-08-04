# Intracranial Hemorrhage Detection — Multi-Label CT Classification

Multi-label classification of brain CT scans for intracranial hemorrhage and its 5 subtypes,
using the [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)
dataset. Each scan is classified across 6 binary labels: `any`, `epidural`, `intraparenchymal`,
`intraventricular`, `subarachnoid`, `subdural` (a scan can have multiple positive labels).

Three experiments were run, varying backbone and loss function. This README documents the
shared pipeline first, then each experiment's distinct setup and results separately, since
they differ in more than just backbone/loss — dataset size, scheduler, threshold strategy, and
checkpoint-selection metric all varied between runs.

## Pipeline overview (across all experiments)

```
DICOM (.dcm)
  → HU conversion (rescale slope/intercept)
  → 3-window extraction (brain / subdural / bone)
  → stack into 3-channel PNG, resize to 384x384
  → patient-grouped, label-stratified train/val/test split
  → EfficientNet backbone (ImageNet-pretrained) + custom classification head
  → early stopping, best checkpoint saved
  → classification report + confusion matrices + PR curves
```


## Data & preprocessing (across all experiments)

- **Source:** RSNA Intracranial Hemorrhage Detection (Kaggle). Raw DICOMs converted to
  Hounsfield Units, windowed three ways (brain: center 40/width 80, subdural: center 80/width
  200, bone: center 600/width 2800), and stacked into 3-channel 384x384 PNGs.
- **Split strategy:** patient-grouped + label-stratified (`StratifiedGroupKFold`, stratified on
  `any`), 80/10/10 train/val/test. Grouping by `PatientID` ensures no patient's scans span more
  than one split, preventing leakage between train and evaluation. This split logic is
  identical across all three experiments — what differs is the *size* of the dataset it was
  run on (see each experiment below).
- **Class imbalance:** hemorrhage-positive scans are ~14% of the dataset, with subtypes ranging from ~6.3%
  (subdural) down to ~0.4% (epidural). Handled in each experiment via `pos_weight` (BCE) or `alpha` (focal loss), computed
  per-label from the training split only.


## Setup

```bash
pip install torch torchvision albumentations opencv-python scikit-learn pandas numpy matplotlib seaborn tqdm
```

Update the paths in `config.py` (`PNG_DIR`, `METADATA_CSV`) to point at your local copy of the
preprocessed dataset.

## Usage

**Train:**
For experiments 1 and 2:
```python
from train import main
main()
```
For experiment 3:
```python
!python /kaggle/working/train.py
```
Trains with early stopping, saves the best checkpoint to `checkpoints/best_model.pth`, and
writes plots/reports/history to `results/`.

**Evaluated on the test set once after tuning**


---

## Experiment 1: EfficientNet-B0 + BCEWithLogitsLoss

**Dataset:** 75,279 images (60,485 train / 7,471 val / 7,323 test — 14,722 / 1,838 / 1,828
patients respectively).

**Config:**
| | |
|---|---|
| Backbone | EfficientNet-B0 |
| Loss | `BCEWithLogitsLoss` with per-label `pos_weight` |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Batch size | 16 |
| Dropout | 0.3 |
| Optimizer | AdamW |
| Scheduler | `ReduceLROnPlateau` |
| Classification threshold | Fixed 0.5 |
| Checkpoint selection | Lowest validation loss |
| Max epochs / patience | 50 / 5 |

**Training:** Early stopping triggered at epoch 10. Best validation loss (0.4804) reached at
epoch 5; validation loss climbed afterward while training loss kept falling, overfitting past
epoch 5.

**Evaluation:** the best checkpoint from this run was not retained after the Kaggle
session ended, so it couldn't be reloaded for a test-set evaluation. The results below
are on the **validation set**.

**Results (validation set):**
- `any`: precision 0.66, recall 0.90, F1 0.76
- Subtype F1 ranged 0.07 (epidural) to 0.58 (intraparenchymal)
- Macro F1: 0.47

---

## Experiment 2: EfficientNet-B3 + Focal Loss

**Dataset:** 75,279 images (60,485 train / 7,471 val / 7,323 test — 14,722 / 1,838 / 1,828
patients) — same dataset as Experiment 1.

**Config:**
| | |
|---|---|
| Backbone | EfficientNet-B3 |
| Loss | Focal Loss, per-label `alpha` (computed from train-split class frequencies), `gamma=2.0` |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Batch size | 16 |
| Dropout | 0.3 |
| Optimizer | AdamW |
| Scheduler | `ReduceLROnPlateau` |
| Classification threshold | Fixed 0.5 |
| Checkpoint selection | Lowest validation loss |
| Max epochs / patience | 50 / 5 |

**Training:** Early stopping triggered at epoch 8. Best validation loss (0.0067) reached at
epoch 3, overfitting started notably earlier than Experiment 1, consistent with B3's larger
parameter count fitting the training set faster.

**Results (test set):**
- `any`: precision 0.60, recall 0.94, F1 0.73, PR-AUC 0.898
- intraparenchymal / intraventricular: F1 0.52 / 0.47, PR-AUC 0.82 / 0.86
- subarachnoid / subdural: F1 0.39 / 0.47, PR-AUC 0.70 / 0.75
- epidural: F1 0.07, PR-AUC 0.140 (precision 0.04, recall 0.95 — over-predicts broadly rather
  than identifying reliably)

---

## Experiment 3: EfficientNet-B0 + Focal Loss

**Dataset:** ~100,002 images (79,845 train / 10,088 val / 10,069 test). **Note:** this is a
larger sample than Experiments 1 and 3, same split logic but different dataset size

**Config:**
| | |
|---|---|
| Backbone | EfficientNet-B0 |
| Loss | Focal Loss, `alpha=0.25` (fixed, not per-label), `gamma=2.0` |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Optimizer | AdamW |
| Scheduler | `CosineAnnealingWarmRestarts` (T_0=5, T_mult=2) |
| Classification threshold | Per-label, learned each epoch (swept 0.05–0.95, picked best F1) |
| Checkpoint selection | Highest macro F1 (not validation loss) |
| Max epochs / patience | 50 / 8 |
| Augmentations | Also includes Gaussian noise/blur and elastic transform, in addition to the shared flip/affine/brightness set |

**Training:** Early stopping triggered at epoch 17. Best macro F1 (0.668) reached at epoch 9.
Validation loss trended upward after ~epoch 5 even as macro F1 kept improving slightly

**Learned thresholds at best checkpoint:** ranged from 0.316 (subdural) to 0.411 (any/epidural)
— all below the default 0.5.

**Results (test set):**
- `any`: precision 0.88, recall 0.76, F1 0.82, PR-AUC 0.898
- intraparenchymal / intraventricular: F1 0.78 / 0.77, PR-AUC 0.84 / 0.84
- subarachnoid / subdural: F1 0.65 / 0.67, PR-AUC 0.72 / 0.74
- epidural: F1 0.04, PR-AUC 0.086 (precision 0.11, recall 0.03 — 1 of 39 positives caught)
- Macro F1: 0.62
