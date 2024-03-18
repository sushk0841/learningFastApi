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

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

uploaded_files = {}

red_flags=[
    {
        "Phrase": "Totally unqualified",
        "Category": "professionalism"
    },
    {
        "Phrase": "She's a fraud",
        "Category": "professionalism"
    },
    {
        "Phrase": "They're useless",
        "Category": "professionalism"
    },
    {
        "Phrase": "Completely inept",
        "Category": "professionalism"
    },
    {
        "Phrase": "Total sham",
        "Category": "professionalism"
    },
    {
        "Phrase": "They're unethical",
        "Category": "professionalism"
    },
    {
        "Phrase": "What a disaster",
        "Category": "professionalism"
    },
    {
        "Phrase": "Unbelievably bad",
        "Category": "professionalism"
    },
    {
        "Phrase": "Totally shady",
        "Category": "professionalism"
    },
    {
        "Phrase": "They're a disgrace",
        "Category": "professionalism"
    },
    {
        "Phrase": "Absolute incompetence",
        "Category": "professionalism"
    },
    {
        "Phrase": "He's a cheat",
        "Category": "professionalism"
    },
    {
        "Phrase": "Such a scam",
        "Category": "professionalism"
    },
    {
        "Phrase": "Lacks integrity",
        "Category": "professionalism"
    },
    {
        "Phrase": "Totally unreliable",
        "Category": "professionalism"
    },
    {
        "Phrase": "Complete joke",
        "Category": "professionalism"
    },
    {
        "Phrase": "Unethical behavior",
        "Category": "professionalism"
    },
    {
        "Phrase": "Serious misconduct",
        "Category": "professionalism"
    },
    {
        "Phrase": "He's a charlatan",
        "Category": "professionalism"
    },
    {
        "Phrase": "They're corrupt",
        "Category": "professionalism"
    },
    {
        "Phrase": "Absolutely dishonest",
        "Category": "professionalism"
    },
    {
        "Phrase": "Lack of professionalism",
        "Category": "professionalism"
    },
    {
        "Phrase": "He's a swindler",
        "Category": "professionalism"
    },
    {
        "Phrase": "Totally unscrupulous",
        "Category": "professionalism"
    },
    {
        "Phrase": "She's deceitful",
        "Category": "professionalism"
    },
    {
        "Phrase": "They're fraudulent",
        "Category": "professionalism"
    },
    {
        "Phrase": "Completely unethical",
        "Category": "professionalism"
    },
    {
        "Phrase": "Utterly dishonest",
        "Category": "professionalism"
    },
    {
        "Phrase": "Blatantly corrupt",
        "Category": "professionalism"
    },
    {
        "Phrase": "Grossly incompetent",
        "Category": "professionalism"
    },
    {
        "Phrase": "Instant wealth creation",
        "Category": "financial"
    },
    {
        "Phrase": "Surefire investment success",
        "Category": "financial"
    },
    {
        "Phrase": "Quick financial gain",
        "Category": "financial"
    },
    {
        "Phrase": "Rapid wealth accumulation",
        "Category": "financial"
    },
    {
        "Phrase": "Easy pro\ufb01t-making",
        "Category": "financial"
    },
    {
        "Phrase": "Guaranteed return on investment",
        "Category": "financial"
    },
    {
        "Phrase": "Effortless financial growth",
        "Category": "financial"
    },
    {
        "Phrase": "Speedy income generation",
        "Category": "financial"
    },
    {
        "Phrase": "Rapid revenue increase",
        "Category": "financial"
    },
    {
        "Phrase": "Quick financial turnaround",
        "Category": "financial"
    },
    {
        "Phrase": "Instant money-making",
        "Category": "financial"
    },
    {
        "Phrase": "Guaranteed financial results",
        "Category": "financial"
    },
    {
        "Phrase": "Easy wealth accumulation",
        "Category": "financial"
    },
    {
        "Phrase": "Rapid pro\ufb01t-making",
        "Category": "financial"
    },
    {
        "Phrase": "Effortless income boost",
        "Category": "financial"
    },
    {
        "Phrase": "Speedy pro\ufb01t increase",
        "Category": "financial"
    },
    {
        "Phrase": "Quick revenue growth",
        "Category": "financial"
    },
    {
        "Phrase": "Instant financial improvement",
        "Category": "financial"
    },
    {
        "Phrase": "Guaranteed wealth boost",
        "Category": "financial"
    },
    {
        "Phrase": "Easy financial success",
        "Category": "financial"
    },
    {
        "Phrase": "Rapid wealth gain",
        "Category": "financial"
    },
    {
        "Phrase": "Quick pro\ufb01t generation",
        "Category": "financial"
    },
    {
        "Phrase": "Instant revenue increase",
        "Category": "financial"
    },
    {
        "Phrase": "Guaranteed income boost",
        "Category": "financial"
    },
    {
        "Phrase": "Effortless wealth creation",
        "Category": "financial"
    },
    {
        "Phrase": "Speedy financial gains",
        "Category": "financial"
    },
    {
        "Phrase": "Quick wealth improvement",
        "Category": "financial"
    },
    {
        "Phrase": "Instant pro\ufb01t increase",
        "Category": "financial"
    },
    {
        "Phrase": "Guaranteed revenue boost",
        "Category": "financial"
    },
    {
        "Phrase": "Blasting calls out",
        "Category": "telemarketing"
    },
    {
        "Phrase": "Spamming their inbox",
        "Category": "telemarketing"
    },
    {
        "Phrase": "Bombarding with texts",
        "Category": "telemarketing"
    },
    {
        "Phrase": "Nonstop robocalls",
        "Category": "telemarketing"
    },
    {
        "Phrase": "Endless phone spam",
        "Category": "telemarketing"
    },
    {
        "Phrase": "Persistent marketing tactics",
        "Category": "telemarketing"
    },
    {
        "Phrase": "Secret pro\ufb01t sharing",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Covert referral fee",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Hidden kickback deal",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Undercovered commission",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Backhanded payment",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Disguised referral fee",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Cloaked compensation",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Sneaky pro\ufb01t split",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Concealed kickback",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Underhanded commission",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Shrouded payment",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Masked referral agreement",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Veiled kickback arrangement",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Hidden \financial incentive",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Covert pro\ufb01t exchange",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Undercover referral agreement",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Sly commission deal",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Disguised pro\ufb01t sharing",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Cloaked \ufb01nancial reward",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Sneaky commission split",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Concealed referral incentive",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Underhanded pro\ufb01t arrangement",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Shrouded commission incentive",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Masked kickback deal",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Veiled \ufb01nancial agreement",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Hidden commission scheme",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Covert referral incentive",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Undercover pro\ufb01t sharing",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Sly kickback arrangement",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Disguised commission incentive",
        "Category": "RESPA compliance"
    },
    {
        "Phrase": "Avoid renting to them",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Prefer not to sell to those people",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Don't deal with that community",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Exclude certain types",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Only rent to our kind",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Steer clear of that neighborhood",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Not suitable for certain families",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Don't want those kinds of tenants",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Avoid selling to that group",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Prefer not to work with them",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Exclude these types of buyers",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Only deal with certain demographics",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Steer away from those areas",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Not for those kinds of people",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Avoid these types of clients",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Prefer not to associate with them",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Exclude certain income levels",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Only cater to speci\ufb01c groups",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Steer clients to certain neighborhoods",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Not open to all families",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Avoid certain types of families",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Not interested in certain demographics",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Prefer homogenous neighborhood",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Selective about client race",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Gender-speci\ufb01c housing",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Excluding disabled tenants",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Age-restricted sales",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Ethnicity-based client choices",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Sexual orientation discrimination",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Family status limitations",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Biased housing advertisements",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Racially targeted marketing",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Exclusionary rental practices",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Segregation in housing offers",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Discriminatory lending practices",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Steering towards certain areas",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Blockbusting tactics",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Redlining practices",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Discriminatory zoning policies",
        "Category": "fair housing laws"
    },
    {
        "Phrase": "Personal investment involved",
        "Category": "conflict of interest"
    },
    {
        "Phrase": "Hidden stake in the property",
        "Category": "conflict of interest"
    },
    {
        "Phrase": "Benetting from this deal",
        "Category": "conflict of interest"
    },
    {
        "Phrase": "My partner's listing",
        "Category": "conflict of interest"
    },
    {
        "Phrase": "Friend's property on sale",
        "Category": "conflict of interest"
    },
    {
        "Phrase": "Selling my own investment",
        "Category": "conflict of interest"
    },
    {
        "Phrase": "Family member's business",
        "Category": "conflict of interest"
    },
    {
        "Phrase": "Personal connection to the buyer",
        "Category": "conflict of interest"
    },
    {
        "Phrase": "bloody",
        "Category": "offensive"
    },
    {
        "Phrase": "bastard",
        "Category": "hate language"
    },
    {
        "Phrase": "lingo",
        "Category": "frustration"
    },
    {
        "Phrase": "frustrations",
        "Category": "not suitable"
    },
    {
        "Phrase": "plonker",
        "Category": "bad words"
    },
    {
        "Phrase": "plonker",
        "Category": "bad words"
    },
    {
        "Phrase": "piss",
        "Category": "bad words"
    },
    {
        "Phrase": "hell",
        "Category": "wrong choice of words"
    }
]

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

@app.get("/hate/")
async def get_hate():
    filename = list(uploaded_files.keys())[0]
    transcription_file = results_dir / "transcription.txt"
    with open(transcription_file, "r") as f:
        recognized_text = f.read()
    hate_speech_result = hate_speech_detector(recognized_text)
    hate_speech_score = hate_speech_result[0]['score']
    red_flag_matches=[]
    for red_flag in red_flags:
        if red_flag['Phrase'] in recognized_text:
            red_flag_matches.append([red_flag['Phrase'],red_flag['Category']])
    hate_speech_label="Hate speech detected" if hate_speech_score>0.8 else "No hate speech detected"
    hate_speech_data=hate_speech_result[0]
    return pd.DataFrame(red_flag_matches,columns=["Phrase","Category"]),hate_speech_label,hate_speech_data

@app.get("/summary/")
async def get_summary():
    filename = list(uploaded_files.keys())[0]
    transcription_file = results_dir / "transcription.txt"
    with open(transcription_file, "r") as f:
        recognized_text = f.read()
    summary=summarizer(recognized_text)
    return {"summary": summary}
    
