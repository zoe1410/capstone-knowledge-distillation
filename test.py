import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from dataset import DatasetHandler
from Student_extraction import StudentWithHead

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def load_student(checkpoint_path, num_classes):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("student_state_dict", checkpoint.get("model_state_dict", checkpoint))
    checkpoint_classes = state["classifier.6.weight"].shape[0]
    if checkpoint_classes != num_classes:
        raise ValueError(
            f"Dataset has {num_classes} classes, but the checkpoint expects {checkpoint_classes}."
        )

    # Older checkpoints can use the narrower classifier from the original evaluator.
    hidden_sizes = (state["classifier.0.weight"].shape[0], state["classifier.3.weight"].shape[0])
    model = StudentWithHead(num_classes, pretrained=False, hidden_sizes=hidden_sizes)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def test_model(model, test_loader, device):
    model.eval()
    
    correct = 0
    total = 0
    
    pbar = tqdm(test_loader, desc="Testing")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        _, logits = model(images)
        _, predicted = logits.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})
    
    accuracy = 100. * correct / total
    print(f"Correct predictions: {correct}/{total}")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MobileNetV2 student.")
    parser.add_argument("--data-dir", type=Path, default=Path("TomatoDataset"))
    parser.add_argument("--checkpoint", type=Path, default=Path("Capstone_KD/checkpoints/best.pth"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    data_handler = DatasetHandler(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        random_seed=42,
    )
    class_names = data_handler.get_class_names()
    model = load_student(args.checkpoint, len(class_names)).to(device)
    _, _, test_loader = data_handler.get_dataloaders()

    print(f"\nTest samples: {len(test_loader.dataset)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Classes: {class_names}")

    accuracy = test_model(model, test_loader, device)
    print(f"Test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
