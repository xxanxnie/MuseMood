import torch

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, audio_features, lyrics_features, midi_features, labels):
        self.audio = torch.tensor(audio_features, dtype=torch.float32)
        self.lyrics = torch.tensor(lyrics_features, dtype=torch.float32)
        self.midi = torch.tensor(midi_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.audio[idx],
            self.lyrics[idx],
            self.midi[idx],
            self.labels[idx]
        )
