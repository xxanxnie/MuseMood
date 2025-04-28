import os
import numpy as np
import librosa
import pretty_midi
from transformers import BertTokenizer, BertModel
import torch
import sys

# === Load BERT for Lyrics ===
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def extract_audio_features(path):
    y, sr = librosa.load(path, sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def extract_lyrics_features(path):
    if not os.path.exists(path):
        return np.zeros(768)
    with open(path, 'r') as f:
        text = f.read()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:,0,:].detach().numpy().flatten()

def extract_midi_features(path):
    if not os.path.exists(path):
        return np.zeros(3)
    try:
        midi = pretty_midi.PrettyMIDI(path)
        notes = [note.pitch for inst in midi.instruments for note in inst.notes]
        return np.array([len(notes), np.mean(notes) if notes else 0, np.mean(midi.get_tempo_changes()[1])])
    except Exception as e:
        print(f"Warning: Failed to process MIDI file {path}. Error: {e}")
        return np.zeros(3)

def load_labels_for_files(audio_files, label_file):
    with open(label_file) as f:
        raw_labels = [line.strip() for line in f.readlines()]
    selected_labels = raw_labels[:len(audio_files)]
    unique_labels = sorted(set(raw_labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = [label_to_index[label] for label in selected_labels]
    return np.array(encoded_labels), unique_labels

def main(mode):
    if mode == "clean":
        AUDIO_DIR = "Integrated_Dataset/Audio_cleaned"
        LYRICS_DIR = "Integrated_Dataset/Lyrics_cleaned"
        MIDI_DIR = "Integrated_Dataset/MIDIs_cleaned"
        OUTPUT_DIR = "cleaned_data"
    elif mode == "degraded":
        AUDIO_DIR = "Integrated_Dataset/Audio_degraded"
        LYRICS_DIR = "Integrated_Dataset/Lyrics_degraded"
        MIDI_DIR = "Integrated_Dataset/MIDIs_degraded"
        OUTPUT_DIR = "degraded_data"
    else:
        print("❌ Invalid mode! Use 'clean' or 'degraded'")
        return

    LABEL_FILE = "Integrated_Dataset/clusters.txt"

    audio_features, lyrics_features, midi_features = [], [], []
    audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(".mp3")])

    for file in audio_files:
        idx = file.split(".")[0]
        print(f"Processing {idx}...")

        audio_feat = extract_audio_features(os.path.join(AUDIO_DIR, f"{idx}.mp3"))
        lyrics_feat = extract_lyrics_features(os.path.join(LYRICS_DIR, f"{idx}.txt"))
        midi_feat = extract_midi_features(os.path.join(MIDI_DIR, f"{idx}.mid"))

        audio_features.append(audio_feat)
        lyrics_features.append(lyrics_feat)
        midi_features.append(midi_feat)

    labels, label_mapping = load_labels_for_files(audio_files, LABEL_FILE)

    # Save to respective folder
    np.save(f"{OUTPUT_DIR}/audio_features.npy", np.array(audio_features))
    np.save(f"{OUTPUT_DIR}/lyrics_features.npy", np.array(lyrics_features))
    np.save(f"{OUTPUT_DIR}/midi_features.npy", np.array(midi_features))
    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    np.save(f"{OUTPUT_DIR}/label_mapping.npy", label_mapping)

    print(f"✅ Preprocessing for '{mode}' dataset complete! Saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess_data.py [clean|degraded]")
    else:
        main(sys.argv[1])
