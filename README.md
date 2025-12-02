# ML Training Demo

A simple Python ML training project for testing TrainForge workflows.

## Usage

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python train.py --epochs 10 --lr 0.001 --batch-size 32

# With custom output directory
python train.py --epochs 20 --output ./my-results
```

### With TrainForge

1. Install the TrainForge GitHub App on this repository
2. Go to TrainForge dashboard and click "Set up workflow"
3. Add your AWS credentials as GitHub Secrets:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION`
4. Trigger a training job from the TrainForge dashboard

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--lr` | 0.001 | Learning rate |
| `--batch-size` | 32 | Batch size |
| `--output` | outputs | Output directory for results |

## Output

Training produces the following files in the output directory:

- `metrics.json` - Training metrics per epoch
- `model.pt` - Trained model weights
- `summary.json` - Training summary with final metrics
