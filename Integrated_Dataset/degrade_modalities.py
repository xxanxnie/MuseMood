import os
import random
import sys
from pydub import AudioSegment
from pydub.generators import WhiteNoise
import pretty_midi
import shutil

# --------- CONFIG --------- #
AUDIO_IN = "Audio_cleaned"
AUDIO_OUT = "Audio_degraded"

LYRICS_IN = "Lyrics_cleaned"
LYRICS_OUT = "Lyrics_degraded"

MIDI_IN = "MIDIs_cleaned"
MIDI_OUT = "MIDIs_degraded"

os.makedirs(AUDIO_OUT, exist_ok=True)
os.makedirs(LYRICS_OUT, exist_ok=True)
os.makedirs(MIDI_OUT, exist_ok=True)

# --------- AUDIO --------- #
def degrade_audio(file_path):
    audio = AudioSegment.from_mp3(file_path)
    noise = WhiteNoise().to_audio_segment(duration=len(audio)).apply_gain(-3)
    audio = audio.overlay(noise)
    audio = audio.low_pass_filter(1500)
    audio = audio.set_frame_rate(8000)
    return audio

def process_audio():
    count = 0
    for filename in sorted(os.listdir(AUDIO_IN)):
        if filename.endswith(".mp3"):
            input_path = os.path.join(AUDIO_IN, filename)
            output_path = os.path.join(AUDIO_OUT, filename)
            degraded = degrade_audio(input_path)
            degraded.export(output_path, format='mp3')
            print(f"[AUDIO] Degraded {filename}")
            count += 1
    print(f"\nTotal degraded audio files: {count}\n")

# --------- LYRICS --------- #
def random_typo(word):
    if len(word) < 3:
        return word
    i = random.randint(1, len(word) - 2)
    return word[:i] + random.choice("!@#$%123") + word[i+1:]

def degrade_text(text):
    words = text.strip().split()
    new_words = []
    for word in words:
        r = random.random()
        if r < 0.6:
            continue
        elif r < 0.7:
            new_words.append("[UNK]")
        elif r < 0.8:
            new_words.append(random_typo(word))
        else:
            new_words.append(word)
    return " ".join(new_words)

def process_lyrics():
    count = 0
    for filename in sorted(os.listdir(LYRICS_IN)):
        if filename.endswith(".txt"):
            with open(os.path.join(LYRICS_IN, filename), 'r') as f:
                text = f.read()
            degraded = degrade_text(text)
            with open(os.path.join(LYRICS_OUT, filename), 'w') as f:
                f.write(degraded)
            print(f"[LYRICS] Degraded {filename}")
            count += 1
    print(f"\nTotal degraded lyrics files: {count}\n")

# --------- MIDI --------- #
def degrade_midi(file_path, output_path):
    midi = pretty_midi.PrettyMIDI(file_path)

    if len(midi.instruments) > 1:
        del midi.instruments[random.randint(0, len(midi.instruments) - 1)]

    for instrument in midi.instruments:
        new_notes = []
        for note in instrument.notes:
            if random.random() < 0.6:
                continue
            if random.random() < 0.5:
                pitch_shift = random.choice([-3, -2, -1, 1, 2, 3])
                note.pitch = max(0, min(127, note.pitch + pitch_shift))
            new_notes.append(note)
        instrument.notes = new_notes

    midi.write(output_path)
    print(f"[MIDI] Degraded {os.path.basename(file_path)}")


def process_midi():
    count = 0
    for filename in sorted(os.listdir(MIDI_IN)):
        if filename.endswith(".mid"):
            input_path = os.path.join(MIDI_IN, filename)
            output_path = os.path.join(MIDI_OUT, filename)

            try:
                degrade_midi(input_path, output_path)
            except Exception as e:
                print(f"[Warning] Failed to degrade {filename}: {e}")
                print(f"[Fallback] Copying clean file instead for {filename}")
                shutil.copy(input_path, output_path)

            count += 1

    print(f"\nTotal MIDI files processed (degraded or fallback): {count}\n")

# --------- MAIN --------- #
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python degrade_modalities.py [audio|lyrics|midi|all]")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "audio":
        process_audio()
    elif mode == "lyrics":
        process_lyrics()
    elif mode == "midi":
        process_midi()
    elif mode == "all":
        process_audio()
        process_lyrics()
        process_midi()
    else:
        print("Invalid mode. Use 'audio', 'lyrics', 'midi', or 'all'.")
        sys.exit(1)

    print(f"\nDegradation for {mode.upper()} completed!")
