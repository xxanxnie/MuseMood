import os
import random
import shutil
import json

# --------- CONFIG --------- #
input_folders = {
    "audio": {"clean": "Audio_cleaned", "degraded": "Audio_degraded"},
    "lyrics": {"clean": "Lyrics_cleaned", "degraded": "Lyrics_degraded"},
    "midi": {"clean": "MIDIs_cleaned", "degraded": "MIDIs_degraded"}
}

output_root = "randomized_data"
output_paths = {
    "audio": os.path.join(output_root, "Audio"),
    "lyrics": os.path.join(output_root, "Lyrics"),
    "midi": os.path.join(output_root, "MIDIs")
}

# Create output folders
for path in output_paths.values():
    os.makedirs(path, exist_ok=True)

# --------- MAIN --------- #
modality_config = {}
degraded_counter = {"audio": 0, "lyrics": 0, "midi": 0}

# Assume Audio_cleaned/ has all base filenames (004.mp3, 008.mp3, etc.)
base_filenames = sorted(f for f in os.listdir(input_folders["audio"]["clean"]) if f.endswith('.mp3'))

for audio_filename in base_filenames:
    file_id = audio_filename.split('.')[0]  # e.g., 004

    # Randomly pick one modality to degrade
    modalities = ["audio", "lyrics", "midi"]
    degraded_modality = random.choice(modalities)
    degraded_counter[degraded_modality] += 1

    modality_config[file_id] = {}

    # Process each modality
    for modality in modalities:
        extension = {"audio": "mp3", "lyrics": "txt", "midi": "mid"}[modality]
        clean_file = os.path.join(input_folders[modality]["clean"], f"{file_id}.{extension}")
        degraded_file = os.path.join(input_folders[modality]["degraded"], f"{file_id}.{extension}")

        if modality == degraded_modality:
            selected_file = degraded_file
            modality_config[file_id][modality] = "degraded"
        else:
            selected_file = clean_file
            modality_config[file_id][modality] = "clean"

        # Copy selected file into output
        output_file = os.path.join(output_paths[modality], f"{file_id}.{extension}")
        shutil.copy(selected_file, output_file)

    print(f"Mixed {file_id}: {modality_config[file_id]}")

# Save JSON modality config
with open(os.path.join(output_root, "modality_config.json"), "w") as f:
    json.dump(modality_config, f, indent=4)

# Save TXT summary
summary_txt_path = os.path.join(output_root, "modality_summary.txt")
total_files = len(base_filenames)

with open(summary_txt_path, "w") as f:
    f.write("Degradation Choices per File:\n")
    f.write("-----------------------------------\n")
    for file_id, modalities in modality_config.items():
        degraded = [k for k, v in modalities.items() if v == "degraded"]
        f.write(f"{file_id}: {degraded[0]} degraded\n")
    
    f.write("\nSummary:\n")
    f.write("-----------------------------------\n")
    for modality in ["audio", "lyrics", "midi"]:
        percent = (degraded_counter[modality] / total_files) * 100
        f.write(f"{modality.capitalize()} degraded: {degraded_counter[modality]} files ({percent:.2f}%)\n")

print("\nRandomized mixing (only one degraded) completed!")
print(f"Modality choices saved to {os.path.join(output_root, 'modality_config.json')}")
print(f"Degradation summary saved to {summary_txt_path}")
