# MuseMood

MuseMood addresses the challenges faced by traditional multimodal systems, which typically assume all inputs — audio, text (lyrics), and MIDI — are equally reliable. In real-world settings, these inputs are often noisy, corrupted, or incomplete due to factors like background noise, sensor failures, or transmission errors. Static fusion models that treat all modalities the same struggle in such conditions, leading to degraded performance. MuseMood introduces a Dynamic Modality Weighting Module that evaluates the real-time quality of each input and adaptively adjusts the fusion weights, enabling the system to prioritize cleaner modalities and minimize the influence of degraded ones.

# Installation
brew install ffmpeg

python3 -m venv venv
source venv/bin/activate
python3 -m venv venv
source venv/bin/activate

1. Prepare Data
Run preprocessing to extract features from the datasets.
# For cleaned dataset
python preprocess_audioData.py clean



cleaned_data/ ➔ Processed clean features

audio_degraded/ ➔ Processed features where only audio is degraded

# Running Model
**  Baseline Model
cd models
python baseline.py
Treats all modalities equally without considering degradation.

** Dynamic Model
python dynamic.py

# Output
The evaluation metrix output is in model.txt




