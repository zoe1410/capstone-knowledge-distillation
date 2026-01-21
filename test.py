import torch
import torch.nn as nn
from tqdm import tqdm
import os

# Import modules
from dataset import DatasetHandler, set_seed
from Student_extraction import StudentExtractor

# ===== SEED CONSISTENCY =====
SEED = 42
set_seed(SEED)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(SEED)

class StudentWithHead(nn.Module):
    """Student model với classification head"""
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.backbone = StudentExtractor(pretrained=pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(96, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        feat_map = self.backbone(x)
        pooled = self.gap(feat_map)
        pooled = pooled.flatten(1)
        logits = self.classifier(pooled)
        return feat_map, logits


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
    return accuracy


def main():
    # Config - PHẢI GIỐNG VỚI main.py
    config = {
        "data_dir": r"./TomatoDataset",
        "num_classes": 10,
        "batch_size": 32,
        "num_workers": 16,
        "checkpoint_path": "./Capstone_KD/checkpoints/best.pth",  # Đường dẫn checkpoint
        "device": "cuda:0"
    }
    
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ===== QUAN TRỌNG: Dùng cùng seed và DatasetHandler =====
    data_handler = DatasetHandler(
        root_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        random_seed=SEED  # Phải giống với training
    )
    
    # Verify consistency
    print("\n===== Verifying split consistency =====")
    data_handler.verify_split_consistency()
    
    _, _, test_loader = data_handler.get_dataloaders()
    
    print(f"\nTest samples: {len(test_loader.dataset)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Classes: {data_handler.get_class_names()}")
    
    # Load model
    model = StudentWithHead(num_classes=config["num_classes"], pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    if 'student_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['student_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print(f"\n✅ Loaded checkpoint from: {config['checkpoint_path']}")
    
    # Test
    accuracy = test_model(model, test_loader, device)
    print(f"\n🏆 Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()