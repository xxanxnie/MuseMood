# 🎶 MuseMood: A Dynamic Multimodal System for Robust Input Fusion 🎶

MuseMood addresses the limitations of traditional multimodal systems by introducing dynamic weighting to handle degraded inputs across audio, lyrics, and MIDI modalities. In real-world settings, inputs can be noisy or incomplete — MuseMood dynamically adjusts fusion weights based on input quality, resulting in improved robustness and emotion recognition performance.

## Dataset Source
MuseMood is built upon the **MIREX Multimodal Emotion Dataset**, which we obtained from Kaggle: https://www.kaggle.com/datasets/imsparsh/multimodal-mirex-emotion-dataset  
R. Panda et al., "Multi-modal music emotion recognition: A new dataset, methodology and comparative analysis," CMMR 2013.

We use a filtered subset of **196 songs** from the dataset — each of which contains all three modalities (audio, lyrics, MIDI). This ensures that our system has complete input during training and evaluation.

## Installation

This project was developed and tested primarily on macOS, but should work on any system with Python 3.7+.

We use ffmpeg for audio processing tasks such as converting .mp3 files to .wav and extracting audio features. It must be installed separately before running any audio-related preprocessing.

```bash
# 1. Install ffmpeg (required for audio preprocessing)
brew install ffmpeg

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install required Python packages
pip install -r requirements.txt
```

The requirements.txt file contains all additional Python libraries we used, including:
- numpy, pandas
- scikit-learn, matplotlib, seaborn
- torch, torchaudio
- pretty_midi, librosa

## Data Cleaning & Preprocessing
MuseMood uses raw **.mp3, .txt, and .mid** files stored in integrated_dataset/ to extract features. The dataset in the repo has already been preprocessed, but the below contain the preprocessing step from raw.

**Step 1: Preprocessing Raw Data**
```bash
# For clean dataset
python preprocess_data.py clean
# For  degraded data
python preprocess_data.py degraded
```
This creates .npy files in cleaned_data/ and degraded_data/ including audio_features.npy, label_mapping.npy, labels.npy, lyrics_features.npy, midi_features.npy.

**Step 2: Modality Degradation**
You can manually degrade specific modalities using:
```bash
python integrated_dataset/randomized_data/degrade_modalities.py audio
python integrated_dataset/randomized_data/degrade_modalities.py lyrics
python integrated_dataset/randomized_data/degrade_modalities.py midi
python integrated_dataset/randomized_data/degrade_modalities.py all
```
process_audio(), process_lyrics(), process_midi() generate degraded files in corresponding folders like Audio_degraded, etc.

**Step 3: Mixing Clean and Degraded Modalities**
```bash
python integrated_dataset/randomized_data/mix_modalities.py
```
This script creates mixed-input cases — e.g., clean audio with degraded lyrics — to simulate realistic inconsistent input conditions.

***Key Metadata Files***
- filename_to_emotion.txt: Maps each file (e.g., song001.wav) to its corresponding emotion label.
- cluster_summary.txt: Lists each cluster and the emotion it represents (e.g., Cluster 1 → Boisterous, Cluster 2 → Cheerful).
- clusters.txt: Cluster IDs used during pre-grouping.
- categories.txt: Full list of target emotion categories (e.g., Confident, Poignant, Silly, Intense).

## Running the Models

**Baseline (Static Fusion)**
```bash
python models/baseline.py
```
- All modalities treated equally.
- Uses features from cleaned_data/ or degraded_data/.
- Feature vectors are concatenated and fed into a 3-layer feedforward neural network (FNN). Assumes perfect inputs.

**Dynamic (Adaptive Fusion)**
```bash
python models/dynamic.py
```
- Uses reliability-aware fusion.
- Better robustness when inputs are partially missing or noisy.
- Each vector is scaled (0–1 score) and then fused before prediction.

## Model Specs
**Built on:**
- Architecture: 3-layer FNN (simple and lightweight)
- Framework: PyTorch
- Batch size: 16
- Optimizer: Adam (lr = 0.001)
- Loss: Cross-entropy
- Epochs: 50
- Evaluation: torch.no_grad() for consistent inference

**Evaluation metrics:** Accuracy, Precision, Recall, F1-Scor

Degradation is applied randomly to one modality at test time, and results are averaged across 3 random seeds.

## Output
The performance summary including accuracy, F1-score, and confusion matrix is written to:  
***model.txt***

## Reports and Documentation
All written deliverables related to the project are located in the reports/ folder. This includes:
- Project Proposal – Initial design, goals, and methodology
- Progress Report – Mid-project status and adjustments
- Final Report – Full description of the system, results, and takeaways
- Demo Slides – Visual summary used during the final presentation

## Authors
Developed by **Annie Xu** and **Beijia Zhang**
For COMS E6156 – Topics in Software Engineering
Columbia University, Spring 2025