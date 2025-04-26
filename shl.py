import os
import sys
import subprocess
import pandas as pd
import numpy as np
import whisper
import language_tool_python

# Check if ffmpeg is installed
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        print("❌ FFmpeg not found. Please install it and add it to your system PATH.")
        sys.exit(1)

check_ffmpeg()

# Load Whisper model
print("Loading Whisper model...")
model = whisper.load_model("base")  # You can change 'base' to 'small', 'medium', etc.

# Load LanguageTool
print("Loading Grammar Checker...")
tool = language_tool_python.LanguageTool('en-US')

def transcribe_audio(audio_path):
    print(f"Transcribing: {audio_path}")
    try:
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        print(f"❌ Error transcribing {audio_path}: {e}")
        return ""

def grammar_score(text):
    if not text.strip():
        return 0.0, 0  # Handle empty transcription safely

    matches = tool.check(text)
    num_errors = len(matches)
    num_words = len(text.split())

    if num_words == 0:
        return 0.0, num_errors  # Avoid division by zero

    errors_per_100_words = (num_errors / num_words) * 100

    # Scoring logic
    if errors_per_100_words <= 2:
        score = 5
    elif errors_per_100_words <= 5:
        score = 4
    elif errors_per_100_words <= 10:
        score = 3
    elif errors_per_100_words <= 20:
        score = 2
    elif errors_per_100_words <= 30:
        score = 1
    else:
        score = 0

    return score, num_errors

def process_audio_files(audio_files):
    results = []
    for audio_path in audio_files:
        if not os.path.isfile(audio_path):
            print(f"❌ File not found: {audio_path}")
            continue

        text = transcribe_audio(audio_path)
        score, errors = grammar_score(text)
        results.append({
            "file": os.path.basename(audio_path),
            "transcription": text,
            "grammar_errors": errors,
            "grammar_score": score
        })

    return pd.DataFrame(results)

def collect_audio_files(folder_path):
    if not os.path.isdir(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return []

    supported_formats = ('.mp3', '.wav', '.m4a')
    return [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith(supported_formats)
    ]

if __name__ == "__main__":
    # Define folders
    train_folder = os.path.join("audios", "train")
    test_folder = os.path.join("audios", "test")

    # Collect audio files
    train_files = collect_audio_files(train_folder)
    test_files = collect_audio_files(test_folder)

    if not train_files:
        print("⚠️ No audio files found in 'train' folder!")
    if not test_files:
        print("⚠️ No audio files found in 'test' folder!")

    # Process and save results
    if train_files:
        print("\nProcessing training files...")
        train_results = process_audio_files(train_files)
        train_results.to_csv("train_results.csv", index=False)
        print("✅ Saved training results to 'train_results.csv'.")

    if test_files:
        print("\nProcessing testing files...")
        test_results = process_audio_files(test_files)
        test_results.to_csv("test_results.csv", index=False)
        print("✅ Saved testing results to 'test_results.csv'.")

