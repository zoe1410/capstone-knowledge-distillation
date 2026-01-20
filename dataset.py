import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import numpy as np
import random

# =========================
# GLOBAL SEED FUNCTION
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# =========================
# WORKER SEED FUNCTION
# =========================
def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
class DatasetHandler:
    """
    Handle dataset loading, augmentation, and splitting (70/15/15)
    
    Data structure:
        root/
            class1/
                img1.jpg
                img2.jpg
                ...
            class2/
                img1.jpg
                ...
    """
    
    def __init__(
        self,
        root_dir,
        image_size=224,
        batch_size=32,
        num_workers=4,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        # ===== SET GLOBAL SEED =====
        set_seed(self.random_seed)
        # Augmentation transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.val_test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _split_indices(self, dataset):
        """
        Split dataset indices into train/val/test (70/15/15)
        Stratified split to maintain class balance
        """
        targets = np.array(dataset.targets)
        indices = np.arange(len(dataset))
        
        # First split: train (70%) vs temp (30%)
        train_indices, temp_indices = train_test_split(
            indices,
            test_size=(self.val_ratio + self.test_ratio),
            stratify=targets,
            random_state=self.random_seed
        )
        
        # Second split: val (15%) vs test (15%) from temp (30%)
        temp_targets = targets[temp_indices]
        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
        
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=(1 - val_ratio_adjusted),
            stratify=temp_targets,
            random_state=self.random_seed
        )
        
        return train_indices, val_indices, test_indices
    
    def get_datasets(self):
        """
        Returns train, val, test datasets
        """
        # Load full dataset without transform first (for splitting)
        full_dataset = ImageFolder(root=self.root_dir)
        
        # Get split indices
        train_indices, val_indices, test_indices = self._split_indices(full_dataset)
        
        # Create datasets with appropriate transforms
        train_dataset = ImageFolder(root=self.root_dir, transform=self.train_transform)
        val_dataset = ImageFolder(root=self.root_dir, transform=self.val_test_transform)
        test_dataset = ImageFolder(root=self.root_dir, transform=self.val_test_transform)
        
        # Create subsets
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)
        test_subset = Subset(test_dataset, test_indices)
        
        return train_subset, val_subset, test_subset
    
    def get_dataloaders(self):
        """
        Returns train, val, test dataloaders
        """
        train_subset, val_subset, test_subset = self.get_datasets()
        
        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
        
        test_loader = DataLoader(
            test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
        
        return train_loader, val_loader, test_loader
    
    def get_class_names(self):
        """
        Returns list of class names
        """
        dataset = ImageFolder(root=self.root_dir)
        return dataset.classes
    
    def get_num_classes(self):
        """
        Returns number of classes
        """
        return len(self.get_class_names())


# ===== Test =====
if __name__ == "__main__":
    # Example usage
    root_dir = r"./kaggle/working/ProcessedOriginal"  # Change to your data path
    
    handler = DatasetHandler(
        root_dir=root_dir,
        image_size=224,
        batch_size=32,
        num_workers=0  # Set to 0 for Windows debugging
    )
    
    # Get dataloaders
    train_loader, val_loader, test_loader = handler.get_dataloaders()
    
    print(f"Number of classes: {handler.get_num_classes()}")
    print(f"Class names: {handler.get_class_names()}")
    print(f"\nTrain samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch
    for images, labels in train_loader:
        print(f"\nBatch shape: {images.shape}")  # [B, 3, 224, 224]
        print(f"Labels shape: {labels.shape}")
        break

# for nw in [4, 8, 4]:
#     handler = DatasetHandler(root_dir, num_workers=nw)
#     train_loader, _, _ = handler.get_dataloaders()
#     images, labels = next(iter(train_loader))
#     print(nw, images.mean().item())