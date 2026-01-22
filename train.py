import os
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
import importlib
import dataset
import subsampling
import conformer
import model_layers

importlib.reload(dataset)
importlib.reload(subsampling)
importlib.reload(conformer)
importlib.reload(model_layers)
from dataset import AudioDataset, collate_fn
from conformer import ConformerEncoder

CSV_PATH = "r/"
DATA_ROOT = "r/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("正在构建字典")
df = pd.read_csv(CSV_PATH, encoding='gbk', sep=None, engine='python')

all_text = "".join(df['Text:LABEL'].astype(str).tolist()).replace(" ", "")
unique_chars = sorted(list(set(all_text)))
VOCAB = {"<pad>": 0, "<unk>": 1}
for i, char in enumerate(unique_chars):
    VOCAB[char] = i + 2

VOCAB_SIZE = len(VOCAB)
print(f"字典构建完成，大小: {VOCAB_SIZE}")

audio_paths = df['Audio:FILE'].apply(lambda x: os.path.join(DATA_ROOT, x)).tolist()
texts = df['Text:LABEL'].apply(lambda x: x.replace(" ", "")).tolist()

train_dataset = AudioDataset(audio_paths, texts, VOCAB)
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)


class AISHELLConformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = ConformerEncoder(
            input_dim=80,
            hidden_dim=256,
            num_layers=6
        )
        self.fc = nn.Linear(256, vocab_size)

    def forward(self, x):
        # x: [Batch, Time, 80]
        x = self.encoder(x)
        logits = self.fc(x)
        return logits


model = AISHELLConformer(VOCAB_SIZE).to(DEVICE)

criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

print(f"开始在 {DEVICE} 上训练...")
EPOCHS = 10

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for batch in pbar:
        fbanks, targets, input_lengths, target_lengths = [x.to(DEVICE) for x in batch]
        logits = model(fbanks)
        log_probs = logits.transpose(0, 1).log_softmax(2)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss / (pbar.n + 1):.4f}"})

    scheduler.step()

    torch.save(model.state_dict(), f"conformer_aishell_epoch{epoch}.pth")
    print(f"Epoch {epoch} 完成! 模型已保存。")

print("训练任务全部完成。")