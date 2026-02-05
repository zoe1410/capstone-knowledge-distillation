import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import os
from tqdm import tqdm

# Import các module đã tạo
from Teacher_extraction import TeacherExtractor
from Student_extraction import StudentExtractor
from PCA_projector import PCAttentionProjector
from GWLinear_projector import GWLinearProjector
from loss_functions import TotalKDLoss, LogitsKDLoss, DIST
from dataset import DatasetHandler
torch.use_deterministic_algorithms(True, warn_only=True)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class StudentWithHead(nn.Module):
    """
    Student model với classification head
    """
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.backbone = StudentExtractor(pretrained=pretrained)
        
        # Classification head: Global Average Pooling + MLP (1024 -> 256 -> 128 -> num_classes)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(96, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Returns:
            feat_map: [B, 1024, 14, 14] - for distillation
            logits: [B, num_classes] - for classification
        """
        feat_map = self.backbone(x)  # [B, 1024, 14, 14]
        
        # Classification
        pooled = self.gap(feat_map)  # [B, 1024, 1, 1]
        pooled = pooled.flatten(1)   # [B, 1024]
        logits = self.classifier(pooled)  # [B, num_classes]
        
        return feat_map, logits
class DistillationPipeline:
    def __init__(
        self,
        data_dir,
        num_classes,
        batch_size=32,
        num_workers=16,
        lr_student=1e-4,
        # lr_teacher=1e-4,
        epochs=120,
        warmup_epochs_student=5,
        # warmup_epochs_teacher=5,
        device="cuda",
        save_dir="checkpoints",
        lambda1=1.0,  # weight for L_proj1 (PCA loss)
        lambda2=1.0,  # weight for L_proj2 (GL loss)
        lambda3=1.0,  # weight for L_logits (Hinton loss)
        patience=15,  # early stopping patience
        start_factor_student=1e-8,
        # start_factor_teacher=1e-8,  # warmup start factor
        eta_min_student=1e-7,
        block_ids=[11,10,9,8,7],
        block_qkv_id=11,
        temperature=4.0,
        lambda_dist=1.0, # weight for DIST loss
        dist_beta=2.0,
        dist_gamma=2.0
        # eta_min_teacher=1e-7,  # cosine annealing min lr  
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.warmup_epochs_student = warmup_epochs_student
        # self.warmup_epochs_teacher = warmup_epochs_teacher
        self.save_dir = save_dir
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.temperature = temperature
        self.patience = patience
        self.start_factor_student = start_factor_student
        # self.start_factor_teacher = start_factor_teacher
        self.eta_min_student = eta_min_student
        self.lambda_dist = lambda_dist
        self.dist_beta = dist_beta
        self.dist_gamma = dist_gamma
        # self.eta_min_teacher = eta_min_teacher,
        os.makedirs(save_dir, exist_ok=True)
        
        # ===== Dataset =====
        print("Loading dataset...")
        self.data_handler = DatasetHandler(
            root_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers
        )
        self.train_loader, self.val_loader, self.test_loader = self.data_handler.get_dataloaders()
        
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        print(f"Num classes: {num_classes}")
        
        # ===== Models =====
        print("\nInitializing models...")
        
        # Teacher (frozen, inference only)
        self.teacher = TeacherExtractor(pretrained=False,
                                        checkpoint_path=r"./Capstone_KD-Burmese/results/ViT-88.46/vit_base_patch16_224/saved_checkpoints/strategy2_top_3_averaged.pth",
                                        block_ids=block_ids,
                                        block_qkv_id=block_qkv_id)
        self.teacher.to(self.device)
        print("✅ Teacher (ViT-B/16) loaded and frozen")
        
        # Student with classification head
        self.student = StudentWithHead(num_classes=num_classes, pretrained=True)
        self.student = self.student.to(self.device)
        print("✅ Student (ResNet-50) loaded")
        
        # # Teacher Head (trainable)
        # self.teacher_head = TeacherHead(num_classes=num_classes, embed_dim=768)
        # self.teacher_head = self.teacher_head.to(self.device)
        # print("✅ Teacher Head (trainable) loaded")
        
        # Projectors
        self.pca_projector = PCAttentionProjector(in_channels=96, embed_dim=768)
        self.pca_projector = self.pca_projector.to(self.device)
        print("✅ PCA Projector loaded")
        
        self.gl_projector = GWLinearProjector(in_dim=96, out_dim=768)
        self.gl_projector = self.gl_projector.to(self.device)
        print("✅ GL Projector loaded")
        
        # ===== Loss functions =====
        self.kd_loss_fn = TotalKDLoss()
        self.ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.logits_loss = LogitsKDLoss(temperature=temperature)
        self.dist_loss_fn = DIST(beta=dist_beta, gamma=dist_gamma)
        # ===== Optimizer (chỉ train student + projectors) =====
        trainable_params = list(self.student.parameters())    + \
                          list(self.pca_projector.parameters()) + \
                          list(self.gl_projector.parameters())
        # self.optimizer_teacher = optim.Adam(self.teacher_head.parameters(), lr=lr_teacher)
        self.optimizer_student = optim.Adam(trainable_params, lr=lr_student)
        
        # ===== Scheduler: Linear warmup + Cosine Annealing (epoch-level) =====
        self.scheduler_student = self._get_scheduler()
        
        print(f"\n✅ Pipeline initialized on {self.device}")
    
    def _get_scheduler(self):
        """
        Linear warmup + Cosine annealing scheduler using SequentialLR (epoch-level)
        Tạo scheduler riêng cho teacher và student
        """
        warmup_epochs_student = self.warmup_epochs_student
        cosine_epochs_student = self.epochs - self.warmup_epochs_student
        # warmup_epochs_teacher = self.warmup_epochs_teacher
        # cosine_epochs_teacher = self.epochs - self.warmup_epochs_teacher 
    
        # ===== SCHEDULER CHO STUDENT =====
        warmup_scheduler_student = LinearLR(
            self.optimizer_student,
            start_factor=self.start_factor_student,
            end_factor=1.0,
            total_iters=self.warmup_epochs_student
        )
        
        cosine_scheduler_student = CosineAnnealingLR(
            self.optimizer_student,
            T_max=cosine_epochs_student,
            eta_min=self.eta_min_student
        )
        
        scheduler_student = SequentialLR(
            self.optimizer_student,
            schedulers=[warmup_scheduler_student, cosine_scheduler_student],
            milestones=[warmup_epochs_student]
        )
        
        # # ===== SCHEDULER CHO TEACHER HEAD =====
        # warmup_scheduler_teacher = LinearLR(
        #     self.optimizer_teacher,
        #     start_factor=self.start_factor_teacher,
        #     end_factor=1.0,
        #     total_iters=self.warmup_epochs_teacher
        # )
        
        # cosine_scheduler_teacher = CosineAnnealingLR(
        #     self.optimizer_teacher,
        #     T_max=cosine_epochs_teacher,
        #     eta_min=self.eta_min_teacher
        # )
        
        # scheduler_teacher = SequentialLR(
        #     self.optimizer_teacher,
        #     schedulers=[warmup_scheduler_teacher, cosine_scheduler_teacher],
        #     milestones=[warmup_epochs_teacher]
        # )
        
        return scheduler_student
    
    def train_one_epoch(self, epoch):
        """Train for one epoch"""
        self.student.train()
        self.pca_projector.train()
        self.gl_projector.train()
        # self.teacher_head.train()  # ← Teacher head cũng train!
        total_loss = 0.0
        total_kd_loss = 0.0
        total_logits_loss = 0.0
        total_ce_loss_s = 0.0
        total_l1 = 0.0
        total_l2 = 0.0
        # total_ce_loss_t = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # ===== Teacher forward (no grad) =====
            with torch.no_grad():
                teacher_out = self.teacher.extract(images)
            Q_t = teacher_out["Q_t"]
            K_t = teacher_out["K_t"]
            V_t = teacher_out["V_t"]
            Attn_t = teacher_out["Attn_t"]
            h_t = teacher_out["block_mean"]  # [B, 196, 768]
            logit_t = teacher_out["logits"] 
            
            # ===== Student forward =====
            feat_map, logit_s = self.student(images) 
            # ===== Teacher Head forward (để lấy teacher logits) =====
            
            # # ===== PCA Projector =====
            pca_out = self.pca_projector(feat_map, Q_t, K_t, V_t)
            PCAttn_s = pca_out["PCAttnS"]
            V_s = pca_out["VS"]
            
            # # ===== GL Projector =====
            h_s_proj = self.gl_projector(feat_map)  # [B, 196, 768]
            
            # ===== Calculate losses =====
            l_proj1, l_proj2 = self.kd_loss_fn(Attn_t, PCAttn_s, V_t, V_s, h_t, h_s_proj)
            ce_loss_s = self.ce_loss_fn(logit_s, labels)
            # ce_loss_t = self.ce_loss_fn(logit_t, labels) 
            # ===== NEW: Logits Loss (Hinton) ====

            logits_kd_loss = self.logits_loss(logit_s, logit_t.detach())
            dist_loss = self.dist_loss_fn(logit_s, logit_t.detach())
            
            # ===== TÍNH LOSS RIÊNG =====
            # Loss cho STUDENT (KHÔNG có ce_loss_teacher!)(Offline learning)
            loss_student = ce_loss_s + self.lambda1 * l_proj1 + self.lambda2 * l_proj2 + self.lambda3 * logits_kd_loss + self.lambda_dist * dist_loss
            # Loss cho TEACHER HEAD (chỉ CE)
            # loss_teacher = ce_loss_t

            # ===== BACKWARD RIÊNG CHO STUDENT TRƯỚC =====
            self.optimizer_student.zero_grad()
            loss_student.backward()  # ← STUDENT TRƯỚC (thêm retain_graph=True)
            self.optimizer_student.step()

            # # ===== BACKWARD RIÊNG CHO TEACHER SAU =====
            # self.optimizer_teacher.zero_grad()
            # loss_teacher.backward()  # ← TEACHER SAU (bỏ retain_graph=True)
            # self.optimizer_teacher.step()

            # ===== Metrics =====
            total_loss += loss_student.item()
            total_kd_loss += (l_proj1.item() + l_proj2.item())
            total_ce_loss_s += ce_loss_s.item()
            total_logits_loss += logits_kd_loss.item()
            total_l1 += l_proj1.item()
            total_l2 += l_proj2.item()
            _, predicted = logit_s.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
            "Loss_S": f"{loss_student.item():.3f}",
            "KD": f"{(self.lambda1*l_proj1.item()+self.lambda2*l_proj2.item()+self.lambda3*logits_kd_loss.item()):.3f}",
            "DIST": f"{(self.lambda_dist*dist_loss.item()):.3f}",
            "CE": f"{ce_loss_s.item():.3f}",
            "Acc": f"{100.*correct/total:.1f}%",
            "LR": f"{self.scheduler_student.get_last_lr()[0]:.4e}",
        })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_kd_loss = total_kd_loss / len(self.train_loader)
        avg_ce_loss_s = total_ce_loss_s / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {
            "loss": avg_loss,
            "kd_loss": avg_kd_loss,
            "ce_loss_s": avg_ce_loss_s,
            "l1_weighted": (total_l1 / len(self.train_loader)) * self.lambda1,
            "l2_weighted": (total_l2 / len(self.train_loader)) * self.lambda2,
            "l1_weighted": (total_l1 / len(self.train_loader)) * self.lambda1,
            "l2_weighted": (total_l2 / len(self.train_loader)) * self.lambda2,
            "l3_weighted": (total_logits_loss / len(self.train_loader)) * self.lambda3,
            "dist_weighted": (dist_loss.item()) * self.lambda_dist, # Note: this is last batch loss, ideally average it too but acceptable for now or add accumulator
            "accuracy": accuracy
        }
    
    @torch.no_grad()
    def validate(self, loader, desc="Val"):
        """Validate on given loader"""
        self.student.eval()
        # self.pca_projector.eval()
        # self.gl_projector.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"[{desc}]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Student forward
            feat_map, logit_s = self.student(images)
            
            # CE Loss
            ce_loss = self.ce_loss_fn(logit_s, labels)
            total_loss += ce_loss.item()
            
            # Accuracy
            _, predicted = logit_s.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                "Loss": f"{ce_loss.item():.4f}",
                "Acc": f"{100.*correct/total:.2f}%"
            })
        
        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
    
    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            "epoch": epoch+1,
            "student_state_dict": self.student.state_dict(),
            # "pca_projector_state_dict": self.pca_projector.state_dict(),
            # "gl_projector_state_dict": self.gl_projector.state_dict(),
            "optimizer_student_state_dict": self.optimizer_student.state_dict(),
            "scheduler_student_state_dict": self.scheduler_student.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(self.save_dir, "latest.pth"))
        
        # Save best (based on lowest val_loss)
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, "best.pth"))
            print(f"💾 Best model saved with val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%")
    
    def load_checkpoint(self, path):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.student.load_state_dict(checkpoint["student_state_dict"])
        # self.pca_projector.load_state_dict(checkpoint["pca_projector_state_dict"])
        # self.gl_projector.load_state_dict(checkpoint["gl_projector_state_dict"])
        self.optimizer_student.load_state_dict(checkpoint["optimizer_student_state_dict"])
        self.scheduler_student.load_state_dict(checkpoint["scheduler_student_state_dict"])
        
        val_loss = checkpoint.get('val_loss', float('inf'))
        val_acc = checkpoint.get('val_acc', 0.0)
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']} with val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%")
        
        return checkpoint["epoch"], val_loss
    
    def train(self, resume_path=None):
        """Full training loop"""
        start_epoch = 0
        best_val_loss = float('inf')  # Lower is better
        epochs_no_improve = 0  # Early stopping counter
        
        if resume_path and os.path.exists(resume_path):
            start_epoch, best_val_loss = self.load_checkpoint(resume_path)
            start_epoch += 1
        
        print("\n" + "="*60)
        print("🚀 Starting Training")
        print(f"   Early Stopping: patience = {self.patience}")
        print("="*60)
        
        for epoch in range(start_epoch, self.epochs):
            # Train
            train_metrics = self.train_one_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(self.val_loader, desc="Val")
            
            # Get current LR (before step)
            current_lr_student = self.scheduler_student.get_last_lr()[0]
            # current_lr_teacher = self.scheduler_teacher.get_last_lr()[0]
            
            # Step scheduler (epoch-level)
            self.scheduler_student.step()
            # self.scheduler_teacher.step()
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1}/{self.epochs} Summary (LR_S: {current_lr_student:.6f}")
            print(f"   Train - Loss: {train_metrics['loss']:.4f}, "
                  f"KD: {train_metrics['l1_weighted']+train_metrics['l2_weighted']+train_metrics['l3_weighted']:.4f}, "
                  f"DIST: {train_metrics['dist_weighted']:.4f}, " 
                  f"CE: {train_metrics['ce_loss_s']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.2f}%")
            print(f"   Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.2f}%")
            print(f" L1 projection: {train_metrics['l1_weighted']:.4f}")
            print(f" L2 projection: {train_metrics['l2_weighted']:.4f}")
            print(f" Logits projection: {train_metrics['l3_weighted']:.4f}")
            # Save checkpoint (based on lowest val_loss)
            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            self.save_checkpoint(epoch, val_metrics["loss"], val_metrics["accuracy"], is_best)
            
            # Early stopping check
            if epochs_no_improve >= self.patience:
                print(f"\n⚠️ Early stopping triggered! No improvement for {self.patience} epochs.")
                print(f"   Best val_loss: {best_val_loss:.4f}")
                break
            
            print(f"   Early stopping: {epochs_no_improve}/{self.patience}")
            print()
        
        # Final test
        print("\n" + "="*60)
        print("🧪 Final Test")
        print("="*60)
        
        # Load best model
        self.load_checkpoint(os.path.join(self.save_dir, "best.pth"))
        test_metrics = self.validate(self.test_loader, desc="Test")
        
        print(f"\n🏆 Final Test Accuracy: {test_metrics['accuracy']:.2f}%")
        
        return test_metrics
    
    def get_student_model(self):
        """
        Trả về student model (không có projectors) để inference
        """
        return self.student


# ===== Main =====
if __name__ == "__main__":
    # Configuration
    config = {
        "data_dir": r"./Burmese-Version1",
        "num_workers": 16,
        "num_classes": 5,
        "batch_size": 16,
        "epochs": 100 ,
        # ===== STUDENT CONFIG =====
        "lr_student": 2e-4,
        "warmup_epochs_student": int(0.1 * 100),  # 10 epochs
        "start_factor_student": 0.1,
        "eta_min_student": 1e-6,
        
        # # ===== TEACHER HEAD CONFIG (riêng biệt) =====
        # "lr_teacher": 8e-5,  # ← LR cao hơn cho teacher head
        # "warmup_epochs_teacher": int(0.01 * 100),  # 5 epochs (ngắn hơn)
        # "start_factor_teacher": 0.01,  # ← Start cao hơn
        # "eta_min_teacher": 1e-8,  # ← Min LR cao hơn
        ##### Chỉnh vị trí lấy ma trận Q,K,V và output transformer block #####
        "block_ids":[0],
        "block_qkv_id":[11],
        # ===== COMMON =====
        "device": "cuda",
        "save_dir": "checkpoints",
        "lambda1": 0.05,
        "lambda2": 0.05,
        "lambda3": 0.5,
        "temperature": 4,
        "patience": 20,
        "lambda_dist": 1.0,
        "dist_beta": 2.0,
        "dist_gamma": 2.0,
    }
    print(f"[KD] block_ids = {config['block_ids']}")
    print(f"[KD] block_qkv_id = {config['block_qkv_id']}")
    pipeline = DistillationPipeline(**config)
    pipeline.train()
