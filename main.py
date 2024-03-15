import os
import json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

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
    filename = list(uploaded_files.keys())[0]
    transcription_file = results_dir / "transcription.txt"

    if not transcription_file.exists():
        raise HTTPException(status_code=404, detail="Transcription not found")

    with open(transcription_file, "r") as f:
        recognized_text = f.read()

    return {"transcription": recognized_text}

@app.get("/timestamps/")
async def get_timestamps():
    filename = list(uploaded_files.keys())[0]
    timestamps_file = results_dir / "timestamps.txt"

    if not timestamps_file.exists():
        raise HTTPException(status_code=404, detail="Timestamps not found")

    with open(timestamps_file, "r") as f:
        timestamps = json.load(f)

    return {"timestamps": timestamps}