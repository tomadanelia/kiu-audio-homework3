import os
import shutil
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .audio_pipeline import process_audio_file

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

app = FastAPI()

# Configure CORS to allow requests from our React frontend
origins = [
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the generated audio files statically
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.post("/process-audio/")
async def create_upload_file(file: UploadFile = File(...)):
    """
    Accepts an audio file, processes it through the AI pipeline,
    and returns the results.
    """
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

    # Save the uploaded file temporarily
    file_extension = ".mp3" # Assuming mp3 for simplicity
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    upload_path = os.path.join("uploads", unique_filename)
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the audio file using our pipeline
        results = process_audio_file(upload_path, "outputs")
        
        # Add the full URL for the audio file to the response
        results["summary_audio_url"] = f"/outputs/{results['summary_audio_filename']}"

    except Exception as e:
        # Clean up the uploaded file in case of an error
        if os.path.exists(upload_path):
            os.remove(upload_path)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Clean up the uploaded file after processing
        if os.path.exists(upload_path):
            os.remove(upload_path)
            
    return results

@app.get("/")
def read_root():
    return {"message": "AI Audio Pipeline API is running."}