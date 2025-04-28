import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ==== Baseline Model ====
class BaselineFusionModel(nn.Module):
    def __init__(self, audio_dim=40, lyrics_dim=768, midi_dim=3, num_classes=5):
        super(BaselineFusionModel, self).__init__()
        input_dim = audio_dim + lyrics_dim + midi_dim
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, num_classes)

    def forward(self, audio_feat, lyrics_feat, midi_feat):
        fused = torch.cat((audio_feat, lyrics_feat, midi_feat), dim=1)
        x = torch.relu(self.fc1(fused))
        x = torch.relu(self.fc2(x))
        return self.output(x)

# ==== Dataset Loader ====
class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, audio, lyrics, midi, labels):
        self.audio = torch.tensor(audio, dtype=torch.float32)
        self.lyrics = torch.tensor(lyrics, dtype=torch.float32)
        self.midi = torch.tensor(midi, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.audio[idx], self.lyrics[idx], self.midi[idx], self.labels[idx]

# ==== Main ====
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load Full Degraded Dataset ===
    audio = np.load("../degraded_data/audio_features.npy")
    lyrics = np.load("../degraded_data/lyrics_features.npy")
    midi = np.load("../degraded_data/midi_features.npy")
    labels = np.load("../degraded_data/labels.npy")

    dataset = MusicDataset(audio, lyrics, midi, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    num_classes = len(np.load("../degraded_data/label_mapping.npy"))
    model = BaselineFusionModel(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # === Training on Full Dataset ===
    print("Training Baseline Model on Full Degraded Dataset...")
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for audio_feat, lyrics_feat, midi_feat, batch_labels in dataloader:
            audio_feat, lyrics_feat, midi_feat, batch_labels = audio_feat.to(device), lyrics_feat.to(device), midi_feat.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(audio_feat, lyrics_feat, midi_feat)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

    print("Training Complete!")

    # === Evaluation on Same Dataset ===
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for audio_feat, lyrics_feat, midi_feat, batch_labels in dataloader:
            audio_feat, lyrics_feat, midi_feat = audio_feat.to(device), lyrics_feat.to(device), midi_feat.to(device)
            outputs = model(audio_feat, lyrics_feat, midi_feat)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    # === Metrics ===
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

    print("\nBaseline Model Evaluation (Train & Test on Full Dataset):")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

if __name__ == "__main__":
    main()
