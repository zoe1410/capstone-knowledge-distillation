import torch
import timm
import torch.nn.functional as F
import math


class TeacherExtractor:
    def __init__(self, model_name="vit_base_patch16_224", pretrained=False,checkpoint_path=None):
        self.model = timm.create_model(model_name, pretrained=pretrained)
        # Load checkpoint nếu có
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Lọc bỏ các key liên quan đến head/classifier
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head') and not k.startswith('classifier')}
            missing, unexpected = self.model.load_state_dict(filtered_state_dict, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
        self.model.eval()
        self.cache = {
            "block_outs": []
        }
        # ===== ĐÓNG BĂNG ViT =====
        for param in self.model.parameters():
            param.requires_grad = False
        

        self.hooks = []
    def _hook_block_multi(self, module, input, output):
        self.cache["block_outs"].append(output.detach())
    # ---- hook QKV ----
    def _hook_qkv(self, module, input, output):
        # output: [B, 197, 2304]
        self.cache["qkv_out"] = output.detach()
    def _register_hooks(self):
        h_qkv = self.model.blocks[7].attn.qkv.register_forward_hook(self._hook_qkv)
    
        block_ids = [7]
        block_hooks = []
        for i in block_ids:
            h = self.model.blocks[i].register_forward_hook(self._hook_block_multi)
            block_hooks.append(h)
    
        self.hooks = [h_qkv] + block_hooks
    # ---- hook Transformer Block output ---- ( đang lấy block 11)
    def _remove_hooks(self):
        # cleanup hooks (rất quan trọng)
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def extract(self, x):
        """
        Forward pass và extract Q, K, V, Attntion teacher, block11_out
        
        Args:
            x: input tensor [B, 3, 224, 224]
        
        Returns:
            dict với các key: Q_t, K_t, V_t, Attn_t, block11_out
        """
        self.cache["block_outs"].clear()
        self._register_hooks()
        
        # ===== forward teacher =====
        with torch.no_grad():
            _ = self.model(x)
        
        # ===== retrieve outputs =====
        qkv_out = self.cache["qkv_out"]          # [B, 197, 2304]
        block_outs = self.cache["block_outs"]   # list length = 5
        # [5, B, 197, 768]
        # block_stack = torch.stack(block_outs, dim=0)
        # # mean theo block
        # block_mean = block_stack.mean(dim=0)    # [B,197,768]
        # ===== split Q K V =====
        Q = qkv_out[:, :, 0:768]
        K = qkv_out[:, :, 768:1536]
        V = qkv_out[:, :, 1536:2304]

        Q_t = Q[:, 1:, :] 
        K_t = K[:, 1:, :] 
        V_t = V[:, 1:, :] 
        # add CLS token
        block_out = block_outs[0]  # Lấy output của block 11
        cls_token = block_out[:, 0, :]
        block_mean = block_out[:, 1:, :]
        
        # Tính attention map để bỏ vào hàm loss
        d = Q_t.shape[-1]  # 768
        attn_logits = torch.matmul(Q_t, K_t.transpose(-2, -1)) / math.sqrt(d)
        attn_weights = F.softmax(attn_logits, dim=-1)
        Attn_t = torch.matmul(attn_weights, V_t)
        
        self._remove_hooks()
        
        return {
            "Q_t": Q_t,
            "K_t": K_t,
            "V_t": V_t,
            "Attn_t": Attn_t,
            "block_mean": block_mean,
            "cls_token": cls_token
        }
    
    def to(self, device):
        """Move model to device"""
        self.model = self.model.to(device)
        return self


# ===== Test =====
if __name__ == "__main__":
    teacher = TeacherExtractor(checkpoint_path="path/to/your/checkpoint.pth")
    
    x = torch.randn(2, 3, 224, 224)
    outputs = teacher.extract(x)
    
    print("Q teacher:", outputs["Q_t"].shape)
    print("K teacher:", outputs["K_t"].shape)
    print("V teacher:", outputs["V_t"].shape)
    print("Block11:", outputs["block11_out"].shape)
    print("Attention teacher:", outputs["Attn_t"].shape)  # [B, 196, 768]
