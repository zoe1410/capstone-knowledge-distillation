import os
import torch
from main import StudentWithHead
from dataset import DatasetHandler
import numpy as np

# ===== SET SEED TO MATCH TRAINING =====
def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ===== CONFIG =====
ckpt_path = "./ViTBase-Resnet50Code_AddLogits_Completed_Ver1_MobileNet85,13/checkpoints/best.pth"  # Đường dẫn checkpoint student
data_dir = "./kaggle/working/ProcessedOriginal"  # Đường dẫn dữ liệu
num_classes = 5
batch_size = 32
num_workers = 0  # Để inference reproducible

# ===== LOAD STUDENT MODEL =====
student = StudentWithHead(num_classes=num_classes, pretrained=False)
checkpoint = torch.load(ckpt_path, map_location='cpu')
student.load_state_dict(checkpoint["student_state_dict"])
student.eval()

# ===== LOAD TEST LOADER (ĐÚNG SPLIT, ĐÚNG SEED) =====
handler = DatasetHandler(
    root_dir=data_dir,
    image_size=224,
    batch_size=batch_size,
    num_workers=num_workers,
    random_seed=42  # Đảm bảo giống training
)
_, _, test_loader = handler.get_dataloaders()

# ===== INFERENCE =====
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = student(images)[1]  # Lấy logits
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100. * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")

# ===== LƯU KẾT QUẢ DỰ ĐOÁN =====
np.save("test_preds.npy", np.array(all_preds))
np.save("test_labels.npy", np.array(all_labels))