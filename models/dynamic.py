import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim

# ==== Load Degradation Map ====
with open("../Integrated_Dataset/randomized_data/modality_config.json", "r") as f:
    degradation_info = json.load(f)

# ==== Supervised Dynamic Model ====
class SupervisedDynamicFusionModel(nn.Module):
    def __init__(self, audio_dim=40, lyrics_dim=768, midi_dim=3, num_classes=5):
        super(SupervisedDynamicFusionModel, self).__init__()
        self.fc1 = nn.Linear(audio_dim + lyrics_dim + midi_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, num_classes)

    def forward(self, audio_feat, lyrics_feat, midi_feat, degradation_mask):
        audio_weighted = audio_feat * degradation_mask[:, 0].unsqueeze(1)
        lyrics_weighted = lyrics_feat * degradation_mask[:, 1].unsqueeze(1)
        midi_weighted = midi_feat * degradation_mask[:, 2].unsqueeze(1)

        fused = torch.cat((audio_weighted, lyrics_weighted, midi_weighted), dim=1)
        x = F.relu(self.fc1(fused))
        x = F.relu(self.fc2(x))
        return self.output(x)

# ==== Dataset Loader ====
class MusicDatasetWithMask(torch.utils.data.Dataset):
    def __init__(self, audio, lyrics, midi, labels, file_ids):
        self.audio = torch.tensor(audio, dtype=torch.float32)
        self.lyrics = torch.tensor(lyrics, dtype=torch.float32)
        self.midi = torch.tensor(midi, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.file_ids = file_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_feat = self.audio[idx]
        lyrics_feat = self.lyrics[idx]
        midi_feat = self.midi[idx]
        label = self.labels[idx]

        file_id = self.file_ids[idx]
        degrade_status = degradation_info[file_id]

        mask = torch.tensor([
            1.0 if degrade_status['audio'] == 'clean' else 0.3,
            1.0 if degrade_status['lyrics'] == 'clean' else 0.3,
            1.0 if degrade_status['midi'] == 'clean' else 0.3
        ], dtype=torch.float32)

        return audio_feat, lyrics_feat, midi_feat, mask, label

# ==== Main ====
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load Data ===
    audio = np.load("../degraded_data/audio_features.npy")
    lyrics = np.load("../degraded_data/lyrics_features.npy")
    midi = np.load("../degraded_data/midi_features.npy")
    labels = np.load("../degraded_data/labels.npy")

    file_ids = sorted(degradation_info.keys())[:len(labels)]

    dataset = MusicDatasetWithMask(audio, lyrics, midi, labels, file_ids)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    num_classes = len(np.load("../degraded_data/label_mapping.npy"))
    model = SupervisedDynamicFusionModel(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # === Training ===
    print("Training Supervised Dynamic Model...")
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for audio_feat, lyrics_feat, midi_feat, mask, batch_labels in dataloader:
            audio_feat, lyrics_feat, midi_feat, mask, batch_labels = audio_feat.to(device), lyrics_feat.to(device), midi_feat.to(device), mask.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(audio_feat, lyrics_feat, midi_feat, mask)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

    print("Training Complete!")

    # === Evaluation ===
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for audio_feat, lyrics_feat, midi_feat, mask, batch_labels in dataloader:
            audio_feat, lyrics_feat, midi_feat, mask = audio_feat.to(device), lyrics_feat.to(device), midi_feat.to(device), mask.to(device)
            outputs = model(audio_feat, lyrics_feat, midi_feat, mask)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    # === Metrics ===
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"\nEvaluation Metrics:")
    print(f"Accuracy :  {acc:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall   :  {recall:.4f}")
    print(f"F1-Score :  {f1:.4f}")

if __name__ == "__main__":
    main()
