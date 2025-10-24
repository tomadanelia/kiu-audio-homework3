import os
import shutil
import uuid
import whisper
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from .audio_pipeline import process_audio_file

# --- Model Loading ---
# This is done once when the server starts.
print("Loading AI models, this may take a moment...")

# 1. Whisper for Transcription (using 'base.en' for a good balance)
# Models: tiny.en, base.en, small.en, medium.en, large
print("Loading Whisper model...")
WHISPER_MODEL = whisper.load_model("base.en")

# 2. Presidio for PII Redaction
print("Loading Presidio analyzer and anonymizer...")
ANALYZER = AnalyzerEngine()
ANONYMIZER = AnonymizerEngine()

# 3. Transformers for Summarization (using a popular BART model)
print("Loading Summarization model...")
SUMMARIZER = pipeline("summarization", model="facebook/bart-large-cnn")

print("All models loaded successfully!")
# --- End Model Loading ---


# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

app = FastAPI()

# Configure CORS
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.post("/process-audio/")
async def create_upload_file(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    # Use original file extension
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    upload_path = os.path.join("uploads", unique_filename)
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the audio file using the pre-loaded models
        results = process_audio_file(
            audio_path=upload_path,
            output_dir="outputs",
            whisper_model=WHISPER_MODEL,
            summarizer=SUMMARIZER,
            analyzer=ANALYZER,
            anonymizer=ANONYMIZER
        )
        
        results["summary_audio_url"] = f"/outputs/{results['summary_audio_filename']}"

    except Exception as e:
        # Clean up in case of error
        if os.path.exists(upload_path):
            os.remove(upload_path)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Clean up the uploaded file after processing is complete
        if os.path.exists(upload_path):
            os.remove(upload_path)
            
    return results

@app.get("/")
def read_root():
    return {"message": "AI Audio Pipeline API (Open Source Edition) is running."}