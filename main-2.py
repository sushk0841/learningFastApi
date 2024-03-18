import os
import json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pandas as pd

app = FastAPI()

# Check for GPU availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model initialization
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

# Processor initialization
processor = AutoProcessor.from_pretrained(model_id)

# Pipeline initialization
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=10,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Summarizer pipeline
summarizer = pipeline("summarization")

# Hate speech detection pipeline
hate_speech_detector = pipeline(
    "text-classification",
    model="badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification",
    tokenizer="badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification",
)

# Function to read red flags data from a JSON file
def read_red_flags_from_file():
    try:
        with open("results/red_flags.json", 'r') as file:
            red_flags = json.load(file)
        return red_flags
    except FileNotFoundError as e:
        return str(e)

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

uploaded_files = {}


@app.post("/upload/")
async def upload_audio(audio_file: UploadFile = File(...)):
    try:
        # Load the audio file using librosa
        input_audio, _ = librosa.load(audio_file.file, sr=16000)

        # Perform speech recognition with timestamps
        result = pipe(input_audio)

        # Recognized text
        recognized_text = result["text"]

        # Extract timestamps
        result = pipe(input_audio, return_timestamps="word")
        timestamps = result["chunks"]

        # Save transcription and timestamps to separate text files
        filename = audio_file.filename
        uploaded_files[filename] = input_audio

        with open(results_dir / "transcription.txt", "w") as f:
            f.write(recognized_text)

        with open(results_dir / "timestamps.txt", "w") as f:
            json.dump(timestamps, f)

        return {"filename": filename}

    except Exception as e:
        return {"error": str(e)}

@app.get("/transcribe/")
async def get_transcription():
    transcription_file = results_dir / "transcription.txt"

    if not transcription_file.exists():
        raise HTTPException(status_code=404, detail="Transcription not found")

    with open(transcription_file, "r") as f:
        recognized_text = f.read()

    return {"transcription": recognized_text}

@app.get("/timestamps/")
async def get_timestamps():
    timestamps_file = results_dir / "timestamps.txt"

    if not timestamps_file.exists():
        raise HTTPException(status_code=404, detail="Timestamps not found")

    with open(timestamps_file, "r") as f:
        timestamps = json.load(f)

    return {"timestamps": timestamps}

@app.get("/hate/")
async def get_hate():
    transcription_file = results_dir / "transcription.txt"
    with open(transcription_file, "r") as f:
        recognized_text = f.read()
    
    # Read red flags from the file
    red_flags = read_red_flags_from_file()
    
    # Initialize red flag matches list
    red_flag_matches = []
    for red_flag in red_flags:
        if red_flag['Phrase'] in recognized_text:
            red_flag_matches.append([red_flag['Phrase'], red_flag['Category']])
    
    # calling hate_speech_detector function 
    hate_speech_result = hate_speech_detector(recognized_text)
    hate_speech_score = hate_speech_result[0]['score']
    
    # Determine hate speech label
    hate_speech_label = "Hate speech detected" if hate_speech_score > 0.8 else "No hate speech detected"
    hate_speech_data = hate_speech_result[0]
    
    return {
        "red_flag_matches": pd.DataFrame(red_flag_matches, columns=["Phrase", "Category"]),
        "hate_speech_label": hate_speech_label,
        "hate_speech_data": hate_speech_data
    }


@app.get("/summary/")
async def get_summary():
    transcription_file = results_dir / "transcription.txt"
    with open(transcription_file, "r") as f:
        recognized_text = f.read()
    summary=summarizer(recognized_text)
    return {"summary": summary}
    
