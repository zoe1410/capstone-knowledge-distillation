import os
import torch
from main import StudentWithHead
from dataset import DatasetHandler, set_seed
import numpy as np

# ===== SET SEED TO MATCH TRAINING (use same function from dataset.py) =====
set_seed(42)

# ===== CONFIG =====
ckpt_path = "./Capstone_KD/checkpoints/best.pth"  # Đường dẫn checkpoint student
data_dir = "./TomatoDataset"  # Đường dẫn dữ liệu
num_classes = 10
batch_size = 32
num_workers = 0  # Để inference reproducible

# ===== LOAD STUDENT MODEL =====
student = StudentWithHead(num_classes=num_classes, pretrained=False)
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
student.load_state_dict(checkpoint["student_state_dict"])
student.eval()

# CRITICAL: Disable gradient computation entirely
torch.set_grad_enabled(False)

# ===== LOAD TEST LOADER (ĐÚNG SPLIT, ĐÚNG SEED, ĐÚNG TRANSFORM) =====
# DatasetHandler từ dataset.py đảm bảo:
# - val_test_transform: Resize(224) -> ToTensor() -> Normalize(ImageNet mean/std)
# - Stratified split với random_seed=42
# - worker_init_fn để reproducibility
handler = DatasetHandler(
    root_dir=data_dir,
    image_size=224,
    batch_size=batch_size,
    num_workers=num_workers,
    random_seed=42  # Đảm bảo giống training
)
_, _, test_loader = handler.get_dataloaders()

print(f"Test samples: {len(test_loader.dataset)}")
print(f"Test batches: {len(test_loader)}")
print(f"Classes: {handler.get_class_names()}")
print(f"Test loader shuffle: {test_loader.sampler.__class__.__name__}")
print(f"Test loader drop_last: {test_loader.drop_last}")

# ===== INFERENCE =====
correct = 0
total = 0
all_preds = []
all_labels = []

for images, labels in test_loader:
    outputs = student(images)[1]  # Lấy logits
    _, predicted = outputs.max(1)
    correct += predicted.eq(labels).sum().item()
    total += labels.size(0)
    all_preds.extend(predicted.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

test_acc = 100. * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")

# Re-enable gradients
torch.set_grad_enabled(True)

# ===== LƯU KẾT QUẢ DỰ ĐOÁN =====
np.save("test_preds.npy", np.array(all_preds))
np.save("test_labels.npy", np.array(all_labels))
