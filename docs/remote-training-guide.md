# Remote Training Guide — RTX PRO 6000 (96GB VRAM)

This guide covers how to label and train a custom YOLO model on the remote GPU host.

## What to Copy to the Remote Machine

From your local `SeMaBot2000/` directory, copy these:

```
SeMaBot2000/
  config/ontologies/steal_a_brainrot.json   # ontology (class -> prompt)
  datasets/steal_a_brainrot/images/          # recorded frames (104 PNGs)
  scripts/autodistill_label.py               # labeling script
  scripts/train.py                           # training script
  src/semabot/training/dataset.py            # dataset split utilities
```

Easiest: zip and copy the whole project:

```bash
# On local (PowerShell)
cd C:\Users\ovatc\Desktop
Compress-Archive -Path SeMaBot2000 -DestinationPath SeMaBot2000.zip

# Copy to remote
scp SeMaBot2000.zip remote-host:~/
```

## Remote Setup (One-Time)

```bash
ssh remote-host

# Unzip
cd ~
unzip SeMaBot2000.zip
cd SeMaBot2000

# Create Python environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install ultralytics autodistill autodistill-grounding-dino
pip install opencv-python-headless numpy
```

## Step 1: Auto-Label with Grounding DINO

This uses the ontology you reviewed to generate bounding box labels for every frame:

```bash
cd ~/SeMaBot2000

python scripts/autodistill_label.py \
    --ontology config/ontologies/steal_a_brainrot.json \
    --images datasets/steal_a_brainrot/images \
    --output datasets/steal_a_brainrot/labels \
    --threshold 0.3
```

**Expected output**: ~104 `.txt` label files in `datasets/steal_a_brainrot/labels/`, one per image, in YOLO format.

**Review the labels**: Spot-check a few label files to make sure the detections look reasonable. Each line is: `class_id cx cy w h` (normalized 0-1).

## Step 2: Split into Train/Val

```bash
python -c "
from semabot.training.dataset import split_dataset, generate_data_yaml
import json

# Load class names from ontology
with open('config/ontologies/steal_a_brainrot.json') as f:
    ontology = json.load(f)

class_names = list(ontology.keys())

# Split 80/20
split_dataset(
    'datasets/steal_a_brainrot/images',
    'datasets/steal_a_brainrot/labels',
    'datasets/steal_a_brainrot/split',
    train_ratio=0.8,
)

# Generate data.yaml
generate_data_yaml(
    'datasets/steal_a_brainrot/split',
    class_names,
)
print('Done. Check datasets/steal_a_brainrot/split/data.yaml')
"
```

## Step 3: Train YOLO11n

```bash
python scripts/train.py \
    --data datasets/steal_a_brainrot/split/data.yaml \
    --epochs 50 \
    --batch 32 \
    --output models/yolo11n_custom.onnx
```

With 96GB VRAM you can use `--batch 64` or higher for faster training. For 104 images, 50 epochs should be sufficient — watch for overfitting in the training logs (val loss increasing).

**Expected output**:
- `runs/train/weights/best.pt` — best PyTorch weights
- `models/yolo11n_custom.onnx` — exported ONNX model for the bot

## Step 4: Copy Model Back to Local Machine

```bash
# On local (PowerShell)
scp remote-host:~/SeMaBot2000/models/yolo11n_custom.onnx C:\Users\ovatc\Desktop\SeMaBot2000\models\
```

## Step 5: Run the Bot with Custom Model

```bash
python -m semabot run --game steal_a_brainrot --model models/yolo11n_custom.onnx --duration 30
```

## Tips

- **More data = better model**: Record additional gameplay sessions with `semabot record` and re-run the pipeline
- **Adjust threshold**: If Grounding DINO produces too many false positives, increase `--threshold` to 0.4 or 0.5
- **Iterate on ontology**: If a class isn't detected well, refine its prompt in the ontology JSON and re-label
- **Monitor training**: Check `runs/train/` for loss plots and mAP metrics
