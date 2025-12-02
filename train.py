#!/usr/bin/env python3
"""Simple ML training script for testing TrainForge."""

import argparse
import json
import os
import time
import random


def train_model(epochs: int, learning_rate: float, batch_size: int):
    """Simulate model training."""
    print(f"Starting training with:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"  - Batch Size: {batch_size}")
    print("-" * 50)

    metrics = {"train_loss": [], "val_loss": [], "accuracy": []}

    for epoch in range(1, epochs + 1):
        # Simulate training time
        time.sleep(0.5)

        # Generate fake metrics (improving over time)
        train_loss = 2.0 / epoch + random.uniform(-0.1, 0.1)
        val_loss = 2.2 / epoch + random.uniform(-0.1, 0.1)
        accuracy = min(0.99, 0.5 + (epoch / epochs) * 0.4 + random.uniform(-0.02, 0.02))

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["accuracy"].append(accuracy)

        print(
            f"Epoch {epoch}/{epochs} - "
            f"loss: {train_loss:.4f} - "
            f"val_loss: {val_loss:.4f} - "
            f"accuracy: {accuracy:.4f}"
        )

    return metrics


def save_results(metrics: dict, output_dir: str):
    """Save training results."""
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Save a fake model file
    model_path = os.path.join(output_dir, "model.pt")
    with open(model_path, "w") as f:
        f.write("FAKE_MODEL_WEIGHTS_v1.0")
    print(f"Model saved to {model_path}")

    # Save summary
    summary = {
        "final_accuracy": metrics["accuracy"][-1],
        "final_loss": metrics["train_loss"][-1],
        "total_epochs": len(metrics["accuracy"]),
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a simple ML model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    print("=" * 50)
    print("TrainForge Demo - ML Training")
    print("=" * 50)

    # Run training
    metrics = train_model(args.epochs, args.lr, args.batch_size)

    # Save results
    save_results(metrics, args.output)

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Final accuracy: {metrics['accuracy'][-1]:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
