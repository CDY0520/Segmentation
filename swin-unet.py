import os
import numpy as np
import random
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from tensorflow.keras.callbacks import EarlyStopping
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import roc_curve, auc


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3584)]
#         )
#         print("Success")
#     except RuntimeError as e:
#         print("Fail", e)

os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

#### 1) 데이터 로드
X_train = np.load('X_train_1.npy')
y_train = np.load('y_train_1.npy')
X_val = np.load('X_val_1.npy')
y_val = np.load('y_val_1.npy')
X_test = np.load('X_test_1.npy')
y_test = np.load('y_test_1.npy')

print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")



# swin_modules.py

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_windows = window_partition(x, self.window_size)
        x_windows = self.attn(self.norm1(x_windows))
        x = window_reverse(x_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        return x + self.mlp(self.norm2(x))

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x))

class SwinUnet(nn.Module):
    def __init__(self, img_size=224, in_channels=1, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.H = img_size // 4
        self.W = img_size // 4
        self.patch_embed = PatchEmbedding(in_channels, embed_dim)
        self.stage1_block1 = SwinBlock(embed_dim, num_heads=3, window_size=7, shift_size=0)
        self.stage1_block2 = SwinBlock(embed_dim, num_heads=3, window_size=7, shift_size=3)
        self.merge1 = PatchMerging(embed_dim)
        self.stage2_block1 = SwinBlock(embed_dim * 2, num_heads=6, window_size=7, shift_size=0)
        self.stage2_block2 = SwinBlock(embed_dim * 2, num_heads=6, window_size=7, shift_size=3)
        self.merge2 = PatchMerging(embed_dim * 2)
        self.stage3_block1 = SwinBlock(embed_dim * 4, num_heads=12, window_size=7, shift_size=0)
        self.stage3_block2 = SwinBlock(embed_dim * 4, num_heads=12, window_size=7, shift_size=3)
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(embed_dim * 4, embed_dim * 2, kernel_size=3, padding=1)
        )
        self.decode2 = nn.Sequential(nn.Conv2d(embed_dim * 4, embed_dim * 2, 3, padding=1), nn.ReLU(inplace=True))
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1)
        )
        self.decode1 = nn.Sequential(nn.Conv2d(embed_dim * 2, embed_dim, 3, padding=1), nn.ReLU(inplace=True))
        self.output = nn.Conv2d(embed_dim, 1, 1)
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        H, W = self.H, self.W

        x1 = self.stage1_block1(x, H, W)
        x1 = self.stage1_block2(x1, H, W)
        x2 = self.merge1(x1, H, W)
        H, W = H // 2, W // 2

        x2 = self.stage2_block1(x2, H, W)
        x2 = self.stage2_block2(x2, H, W)
        x3 = self.merge2(x2, H, W)
        H, W = H // 2, W // 2

        x3 = self.stage3_block1(x3, H, W)
        x3 = self.stage3_block2(x3, H, W)

        x3 = x3.transpose(1, 2).view(B, -1, H, W)
        d2 = self.upsample2(x3)
        x2_spatial = x2.transpose(1, 2).view(B, -1, H * 2, W * 2)
        d2 = self.decode2(torch.cat([d2, x2_spatial], dim=1))
        d1 = self.upsample1(d2)
        x1_spatial = x1.transpose(1, 2).view(B, -1, H * 4, W * 4)
        d1 = self.decode1(torch.cat([d1, x1_spatial], dim=1))
        d1 = self.final_upsample(d1)  # (B, embed_dim, 224, 224)
        return self.output(d1)



def load_tensor_dataset(x_path, y_path):
    x = np.load(x_path)
    y = np.load(y_path)
    if len(x.shape) == 3:
        x = x[:, np.newaxis, :, :]
    if len(y.shape) == 3:
        y = y[:, np.newaxis, :, :]
    x_tensor = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    return TensorDataset(x_tensor, y_tensor)

train_ds = load_tensor_dataset("X_train_1.npy", "y_train_1.npy")
val_ds = load_tensor_dataset("X_val_1.npy", "y_val_1.npy")
test_ds = load_tensor_dataset("X_test_1.npy", "y_test_1.npy")

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# 3. 모델 정의 및 학습 함수
model = SwinUnet(img_size=224).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

history = {
    "train_losses": [], "val_losses": [],
    "train_dices": [], "val_dices": []
}

# EarlyStopping 클래스 정의
class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"\u23f3 EarlyStopping counter {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model():
    early_stopper = EarlyStopping(patience=7, verbose=True)

    for epoch in range(20):
        model.train()
        train_loss, train_dice = 0, 0
        for img, mask in train_loader:
            img, mask = img.cuda(), mask.cuda()
            pred = model(img)

            if mask.ndim == 5:
                mask = mask.squeeze(-1)
            mask = F.interpolate(mask, size=pred.shape[2:], mode='bilinear', align_corners=False)

            loss = criterion(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred_bin = (torch.sigmoid(pred) > 0.5).float()
            intersection = (pred_bin * mask).sum()
            dice = (2. * intersection) / (pred_bin.sum() + mask.sum() + 1e-8)
            train_dice += dice.item()

        history["train_losses"].append(train_loss / len(train_loader))
        history["train_dices"].append(train_dice / len(train_loader))

        model.eval()
        val_loss, val_dice = 0, 0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.cuda(), mask.cuda()
                pred = model(img)

                if mask.ndim == 5:
                    mask = mask.squeeze(-1)
                mask = F.interpolate(mask, size=pred.shape[2:], mode='bilinear', align_corners=False)

                loss = criterion(pred, mask)
                val_loss += loss.item()
                pred_bin = (torch.sigmoid(pred) > 0.5).float()
                intersection = (pred_bin * mask).sum()
                dice = (2. * intersection) / (pred_bin.sum() + mask.sum() + 1e-8)
                val_dice += dice.item()

        val_loss_mean = val_loss / len(val_loader)
        history["val_losses"].append(val_loss_mean)
        history["val_dices"].append(val_dice / len(val_loader))

        print(f"[Epoch {epoch+1}] Train Loss: {history['train_losses'][-1]:.4f}, Dice: {history['train_dices'][-1]:.4f} | Val Loss: {val_loss_mean:.4f}, Dice: {history['val_dices'][-1]:.4f}")

        early_stopper(val_loss_mean)
        if early_stopper.early_stop:
            print("\ud83d\uded1 Early stopping triggered. Training stopped.")
            break

print("🚀 모델 훈련 시작")
train_model()

# 4. 학습 결과 시각화 함수
def plot_training_curves():
    if not history["train_losses"] or not history["val_losses"]:
        raise ValueError("📛 먼저 train_model()을 실행해서 학습을 진행하세요!")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_losses"], label='Train Loss')
    plt.plot(history["val_losses"], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Crossentropy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_dices"], label='Train Dice')
    plt.plot(history["val_dices"], label='Val Dice')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

print("📈 Training 곡선 시각화")
plot_training_curves()

# 5. 평가 함수 정의
def evaluate_and_display(y_true, y_prob, split_name="Test"):
    if y_true.ndim == 4:
        y_true = np.transpose(y_true, (0, 3, 1, 2))


    if y_true.shape[2:] != y_prob.shape[2:]:
        y_true_resized = torch.tensor(y_true).float()
        y_true_resized = F.interpolate(y_true_resized, size=y_prob.shape[2:], mode='bilinear', align_corners=False).numpy()
    else:
        y_true_resized = y_true

    y_true_flat = y_true_resized.squeeze(1).flatten().astype(int)
    y_pred_flat = (y_prob > 0.5).astype(int).squeeze(1).flatten()

    dice = f1_score(y_true_flat, y_pred_flat)
    iou = jaccard_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat)
    recall = recall_score(y_true_flat, y_pred_flat)
    acc = accuracy_score(y_true_flat, y_pred_flat)
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    print(f"\n📊 {split_name} 평가 지표")
    print(f"Dice Coefficient (F1 Score): {dice:.4f}")
    print(f"IoU (Jaccard Index):        {iou:.4f}")
    print(f"Precision:                  {precision:.4f}")
    print(f"Recall:                     {recall:.4f}")
    print(f"Specificity:                {specificity:.4f}")
    print(f"Accuracy:                   {acc:.4f}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Background', 'Lesion'])
    plt.figure(figsize=(5, 4))
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

# 6. Validation 예측 및 평가
def evaluate_validation():
    print("\n================= 🟡 VALIDATION EVALUATION =================")
    model.eval()
    val_pred_prob = []

    with torch.no_grad():
        for img_batch, _ in val_loader:
            img_batch = img_batch.cuda()
            pred_batch = torch.sigmoid(model(img_batch)).cpu().numpy()
            val_pred_prob.append(pred_batch)

    val_pred_prob = np.concatenate(val_pred_prob, axis=0)

    print(f"[DEBUG] val_pred_prob shape: {val_pred_prob.shape}")
    print(f"[DEBUG] val_pred_prob sample values (0~1):\n{val_pred_prob[0, 0, :5, :5]}")

    evaluate_and_display(y_val, val_pred_prob, split_name="Validation")
evaluate_validation()

def evaluate_test():
    print("\n================= 🔵 TEST EVALUATION =================")
    model.eval()
    test_pred_prob = []

    with torch.no_grad():
        for img_batch, _ in test_loader:
            img_batch = img_batch.cuda()
            pred_batch = torch.sigmoid(model(img_batch)).cpu().numpy()
            test_pred_prob.append(pred_batch)

    test_pred_prob = np.concatenate(test_pred_prob, axis=0)

    print(f"[DEBUG] test_pred_prob shape: {test_pred_prob.shape}")
    print(f"[DEBUG] test_pred_prob sample values (0~1):\n{test_pred_prob[0, 0, :5, :5]}")

    evaluate_and_display(y_test, test_pred_prob, split_name="Test")
evaluate_test()

# 8. 예측 시각화 함수
def visualize_predictions():
    print("\n================= 🖼️ VISUALIZATION =================")
    num_samples = 5
    indices = random.sample(range(len(X_test)), num_samples)
    test_sample_tensor = torch.tensor(X_test[indices]).permute(0,3,1,2).float().cuda()
    with torch.no_grad():
        preds_prob = torch.sigmoid(model(test_sample_tensor)).cpu().numpy()
        preds_bin = (preds_prob > 0.5).astype(np.uint8)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
    for i, idx in enumerate(indices):
        axes[i, 0].imshow(X_test[idx].squeeze(), cmap='gray')
        axes[i, 0].set_title('Input')
        axes[i, 1].imshow(y_test[idx].squeeze(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 2].imshow(preds_bin[i].squeeze(), cmap='gray')
        axes[i, 2].set_title('Prediction')
        for j in range(3):
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()
    print("✅ Visualization Finished")

print("🖼️ 시각화")
visualize_predictions()



a=1