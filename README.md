# ImageMol Glaucoma Neuroprotection

This repository provides an AI-assisted workflow for discovering neuroprotective
compounds for glaucoma using endoplasmic reticulum (ER) stress mitigation as the
screening phenotype. It connects compound-level evidence, molecular-image
prediction, virtual screening and experimental candidate evaluation.

The prediction framework is based on the published
[ImageMol implementation](https://github.com/HongxinXiang/ImageMol).
The project contribution is the disease-oriented discovery workflow, not a new
ImageMol architecture.

## Workflow

```text
Positive and provisional negative SMILES
  -> labelled CSV -> 224 x 224 molecular images
  -> ImageMol fine-tuning -> prediction scores
  -> candidate prioritization -> attribution and experimental evaluation
```

## 1. Installation

```bash
git clone https://github.com/jimmyrate/ImageMol_glaucoma_neuroprotection.git
cd ImageMol_glaucoma_neuroprotection
```

Run the following commands from the repository root. Examples use Bash syntax;
Windows users can use WSL. The training and basic Grad-CAM entry points require
an NVIDIA CUDA-capable GPU.

Example environment for the documented workflow:

```bash
conda create -n imagemol_glaucoma python=3.10 -y
conda activate imagemol_glaucoma
conda install -c conda-forge rdkit -y
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install "numpy<2" "pandas<3" scikit-learn==1.0.2 tqdm pillow "matplotlib<3.8" "opencv-python-headless<4.12"
```

Select a PyTorch CUDA build compatible with your GPU driver using the
[official installation commands](https://pytorch.org/get-started/previous-versions/).
This is an example environment, not the historical configuration of every study
experiment. The original ImageMol README also documents a legacy Python 3.7 /
PyTorch 1.4 environment; do not combine it with the example above.

```bash
python -c "import torch, torchvision, rdkit, pandas, sklearn, cv2; print('CUDA available:', torch.cuda.is_available())"
```

## 2. Download and Prepare the Dataset

The dataset is available in the
[public data directory](https://github.com/jimmyrate/ImageMol_glaucoma_neuroprotection/tree/main/data).
Both files are included in a repository clone:

| File | Input column | Assigned label |
| --- | --- | --- |
| `data/positive_smiles.csv` | `SMILES` | `1`: literature-supported ER-stress mitigation |
| `data/negative_smiles.csv` | `SMILES` | `0`: provisional negative |

Negative labels indicate the absence of identified supporting evidence in the
curated sources, not standardized experimental confirmation of inactivity.
The prediction task is binary classification rather than potency regression.

### 2.1 Combine the SMILES files

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

frames = []
for filename, label in [('positive_smiles.csv', 1), ('negative_smiles.csv', 0)]:
    frame = pd.read_csv(Path('data') / filename)[['SMILES']].copy()
    frame = frame.rename(columns={'SMILES': 'smiles'})
    frame['label'] = label
    frames.append(frame)
data = pd.concat(frames, ignore_index=True)
data.insert(0, 'index', range(1, len(data) + 1))
for part in ['raw', 'processed']:
    Path('datasets/finetuning/glaucoma', part).mkdir(parents=True, exist_ok=True)
data.to_csv('datasets/finetuning/glaucoma/raw/glaucoma_data.csv', index=False)
print('Prepared rows:', len(data))
print(data['label'].value_counts().sort_index())
PY
```

The output has columns `index,smiles,label`. Check molecular identity, duplicate
structures and conflicting labels before creating evaluation partitions. If
resampling is used, apply it only to the training partition after splitting.

### 2.2 Convert SMILES to images

```bash
python data_process/smiles2img_finetune.py \
  --dataroot datasets/finetuning/glaucoma/raw \
  --csvname glaucoma_data.csv \
  --theCsvFirstLineSMILES smiles \
  --outputImageDir datasets/finetuning/glaucoma/processed/224 \
  --outputCsvName datasets/finetuning/glaucoma/processed/glaucoma_processed_ac.csv
```

The RDKit-based converter renders one molecule per 224 x 224 image and excludes
failed conversions from the cleaned output CSV. Image filenames follow original
one-based row positions. Keep the cleaned CSV's `index` values unchanged; do not
renumber the CSV independently of the images.

```text
datasets/finetuning/glaucoma/
  raw/glaucoma_data.csv
  processed/glaucoma_processed_ac.csv
  processed/224/1.png
  processed/224/2.png
  ...
```

Train on the cleaned CSV. Use consistent molecular rendering settings during
training, screening and interpretation.

## 3. Download the Initial ImageMol Checkpoint

Download the **original pretrained ImageMol weights** using the download link
under **Finetuning -> Download pre-trained ImageMol** in the
[original ImageMol GitHub README](https://github.com/HongxinXiang/ImageMol#finetuning).
Create `ckpts/` and place or rename the downloaded file to:

```text
ckpts/ImageMol.pth.tar
```

Use the released pretrained checkpoint, not a checkpoint generated by the
upstream toy-data example. Only load weights obtained from trusted sources.

| Checkpoint | Role |
| --- | --- |
| `ckpts/ImageMol.pth.tar` | Initial pretrained weights for ER-stress fine-tuning |
| `logs/glaucoma/valid_best.pth` | Task-specific model produced by fine-tuning, used for scoring and Grad-CAM |

The fine-tuned checkpoint is not included in the two SMILES files. Generate it
with the training command below or obtain the corresponding released task model
when available. Initial pretraining weights are not an ER-stress predictor.

## 4. Train the ER-Stress Classifier

```bash
python finetune.py \
  --dataroot datasets/finetuning \
  --dataset glaucoma \
  --task_type classification \
  --image_model ResNet18 \
  --resume ckpts/ImageMol.pth.tar \
  --log_dir logs/glaucoma \
  --save_finetune_ckpt 1 \
  --split random \
  --seed 42 \
  --runseed 2021 \
  --gpu 0 \
  --ngpu 1 \
  --workers 2 \
  --imageSize 224 \
  --lr 0.01 \
  --weight_decay -5 \
  --batch 128 \
  --epochs 100
```

This command is a workflow example. Use the recorded experiment-specific
configuration when reproducing manuscript results.

- `--dataroot` is the parent of `glaucoma/`, not the `glaucoma/` directory itself.
- `--seed` controls partitioning; `--runseed` controls model-training randomness.
- `--weight_decay -5` means `1e-5`, because this argument is a base-10 exponent.
- Supported splits are `random`, `stratified`, `scaffold`, `random_scaffold`
  and `scaffold_balanced`.
- `--image_aug` and `--weighted_CE` are optional flags, omitted in this example.

The trainer creates approximately 80%/10%/10% training/validation/test partitions.
For classification it selects the best checkpoint by validation AUROC and saves
`logs/glaucoma/valid_best.pth`. Training output includes the corresponding test
metrics. Retain console logs, seeds, input versions and package versions.

`evaluate.py` constructs a scaffold-based test split. It should not be used to
reproduce a random-split test result without matching the original partition.
Scoring all rows, as below, is inference rather than held-out evaluation.

## 5. Predict Scores

### 5.1 Score the prepared dataset

```bash
mkdir -p outputs
python predict.py \
  --dataroot datasets/finetuning \
  --dataset glaucoma \
  --resume logs/glaucoma/valid_best.pth \
  --image_model ResNet18 \
  --imageSize 224 \
  --gpu 0 \
  --ngpu 1 \
  --workers 2 \
  --batch 128 \
  --output_csv outputs/glaucoma_predictions.csv
```

The output preserves input columns and adds `pred_score`, the sigmoid score for
the ER-stress-mitigation class. Higher scores indicate stronger model support for
this class, not stronger experimental inhibitory potency.

### 5.2 Score a new screening library

Supply your own `data/screening_compounds.csv` with a `SMILES` column and optional
compound names or metadata. This is a user-supplied input, not one of the released
labelled dataset files. Labels are not required for prediction.

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

data = pd.read_csv('data/screening_compounds.csv').rename(columns={'SMILES': 'smiles'})
data = data.drop(columns=['index'], errors='ignore')
data.insert(0, 'index', range(1, len(data) + 1))
for part in ['raw', 'processed']:
    Path('datasets/finetuning/screening', part).mkdir(parents=True, exist_ok=True)
data.to_csv('datasets/finetuning/screening/raw/screening_data.csv', index=False)
PY

python data_process/smiles2img_finetune.py \
  --dataroot datasets/finetuning/screening/raw \
  --csvname screening_data.csv \
  --theCsvFirstLineSMILES smiles \
  --outputImageDir datasets/finetuning/screening/processed/224 \
  --outputCsvName datasets/finetuning/screening/processed/screening_processed_ac.csv

python predict.py \
  --dataroot datasets/finetuning \
  --dataset screening \
  --resume logs/glaucoma/valid_best.pth \
  --gpu 0 \
  --ngpu 1 \
  --batch 128 \
  --output_csv outputs/screening_predictions.csv
```

Rank the screened molecules and export a high-score subset:

```bash
python - <<'PY'
import pandas as pd
data = pd.read_csv('outputs/screening_predictions.csv')
data = data.sort_values('pred_score', ascending=False)
data.to_csv('outputs/screening_ranked.csv', index=False)
data.loc[data['pred_score'] > 0.99].to_csv('outputs/screening_score_above_099.csv', index=False)
PY
```

A score of at least 0.5 defines the predicted positive class. The stricter cutoff
above 0.99 was used for high-confidence prioritization in this study. Molecular
weight, synthetic accessibility and toxicity are separate assessments, not
outputs of the ER-stress classification score. The study's toxicity predictor
is a separately trained ImageMol-based Tox21 model.

## 6. Grad-CAM Interpretation

Use the **fine-tuned ER-stress model**, not the initial pretrained weights:

```bash
mkdir -p outputs/gradcam
python -m cam_visualization.main \
  --image_model ResNet18 \
  --resume logs/glaucoma/valid_best.pth \
  --img_path datasets/finetuning/glaucoma/processed/224/1.png \
  --gradcam_save_path outputs/gradcam/compound_1.png \
  --thresh 0.0
```

Choose an existing image; indices can contain gaps after failed conversions.
Warmer colors indicate higher contribution to the model output. `--thresh` is a
normalized heatmap-display threshold, not an activity-score cutoff. The supplied
utility exports a PNG overlay and an SVG containing the raster heatmap; it does
not create a fully vectorized, atom-editable heatmap.

### Optional candidate masking and cross-modal analysis

These extended analyses require the supplementary `reviewer_response_glaucoma/`
scripts to be present in your checkout. A user-supplied `data/candidates.csv`
must contain `SMILES` (or `smiles`) and `Drug` (or `drug`/`name`) columns.

```bash
python reviewer_response_glaucoma/candidate_interpretability.py \
  --project_root "$PWD" \
  --checkpoint logs/glaucoma/valid_best.pth \
  --candidate_csv data/candidates.csv \
  --output_dir outputs/candidate_attribution

python reviewer_response_glaucoma/candidate_gradcam_occlusion.py \
  --project_root "$PWD" \
  --checkpoint logs/glaucoma/valid_best.pth \
  --candidate_csv data/candidates.csv \
  --output_dir outputs/candidate_occlusion \
  --top_selection score \
  --random_trials 100
```

The attribution script exports prediction summaries, Grad-CAM maps and atom
scores. The masking script compares high-attribution masking with 100 random
masks matched by selected atom count. Score decrease is the original score minus
the masked-image score; random-mask decreases are averaged within each candidate.

Chemprop comparisons additionally require the graph-analysis scripts and a
separately trained Chemprop checkpoint. They are not generated by the basic
Grad-CAM command or the ImageMol checkpoint. For atom sets A (ImageMol) and B
(Chemprop), coverage is `100 * |A intersection B| / |A|`, and Jaccard overlap is
`|A intersection B| / |A union B|`.

## 7. Outputs and Troubleshooting

| Output | Contents |
| --- | --- |
| `glaucoma_processed_ac.csv` | Successfully rendered rows with image indices and labels |
| `processed/224/*.png` | Molecular images |
| `logs/glaucoma/valid_best.pth` | Validation-selected ER-stress model |
| `outputs/*predictions.csv` | Input records with `pred_score` |
| `outputs/gradcam/` | Grad-CAM overlays |

- **Dataset not found:** point `--dataroot` to `datasets/finetuning`, the parent
  directory of `glaucoma/` or `screening/`.
- **Missing image:** match the cleaned CSV's `index` to `processed/224/<index>.png`.
- **Missing checkpoint:** obtain the initial model from the original ImageMol
  GitHub repository, then fine-tune to generate the ER-stress model.
- **CUDA error:** check the PyTorch build and GPU driver. The original training
  and basic Grad-CAM scripts call CUDA explicitly.
- **Prediction input format:** use comma-separated CSV files and the processed
  image index mapping.
- **Different metrics:** match the partition, seeds, preprocessing and checkpoint
  selection. Whole-dataset scores are not held-out performance estimates.

## Acknowledgements and Citation

This project builds on [ImageMol](https://github.com/HongxinXiang/ImageMol).
Retain the original software notices and cite the framework paper:

Zeng, X. et al. Accurate prediction of molecular properties and drug targets using
a self-supervised image representation learning framework. *Nature Machine
Intelligence* **4**, 1004-1016 (2022).
[https://doi.org/10.1038/s42256-022-00557-6](https://doi.org/10.1038/s42256-022-00557-6)
