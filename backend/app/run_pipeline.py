import sys
import os
import whisper
import datetime
from transformers import pipeline
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Ensure the app module can be found
# This allows us to import from app.audio_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
from audio_pipeline import process_audio_file

# --- File Definitions ---
OUTPUT_DIR = "pipeline_outputs"
TRANSCRIPT_FILENAME = "output_transcript.txt"
SUMMARY_FILENAME = "output_summary.mp3"
LOG_FILENAME = "audit.log"

def main():
    """
    Main function to run the entire pipeline from the command line.
    """
    # 1. Check for input audio file
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <path_to_audio_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"Error: File not found at {audio_path}")
        sys.exit(1)

    # --- Load Models ---
    # This is done here so we only load them when running this script.
    print("Loading AI models, this may take a moment...")
    whisper_model = whisper.load_model("base.en")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    print("All models loaded successfully!")

    # --- Run Processing ---
    print(f"\nProcessing audio file: {audio_path}")
    try:
        results = process_audio_file(
            audio_path=audio_path,
            output_dir=OUTPUT_DIR, # We will handle file naming explicitly
            whisper_model=whisper_model,
            summarizer=summarizer,
            analyzer=analyzer,
            anonymizer=anonymizer
        )
    except Exception as e:
        print(f"An error occurred during pipeline processing: {e}")
        sys.exit(1)

    # --- Generate Output Files ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nGenerating output files in '{OUTPUT_DIR}/' directory...")

    # 1. Save Redacted Transcript (output_transcript.txt)
    transcript_path = os.path.join(OUTPUT_DIR, TRANSCRIPT_FILENAME)
    with open(transcript_path, 'w') as f:
        f.write(results['redacted_transcript'])
    print(f"✅ Saved redacted transcript to: {transcript_path}")

    # 2. Rename Summary Audio (output_summary.mp3)
    # The pipeline already created a summary file with a unique ID. We rename it.
    generated_summary_path = os.path.join(OUTPUT_DIR, results['summary_audio_filename'])
    final_summary_path = os.path.join(OUTPUT_DIR, SUMMARY_FILENAME)
    if os.path.exists(final_summary_path):
        os.remove(final_summary_path) # Remove old one if it exists
    os.rename(generated_summary_path, final_summary_path)
    print(f"✅ Saved audio summary to: {final_summary_path}")

    # 3. Create Audit Log (audit.log)
    log_path = os.path.join(OUTPUT_DIR, LOG_FILENAME)
    with open(log_path, 'w') as f:
        f.write("--- AI AUDIO PIPELINE AUDIT LOG ---\n")
        f.write(f"Processing Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write("="*40 + "\n\n")
        
        f.write(f"[INPUT]\n")
        f.write(f"Source Audio File: {os.path.basename(audio_path)}\n\n")
        
        f.write(f"[TRANSCRIPTION & CONFIDENCE]\n")
        f.write(f"Confidence Score: {results['confidence_score']} ({results['confidence_level']})\n")
        f.write(f"Full Transcript (Original):\n---\n{results['transcript']}\n---\n\n")

        f.write(f"[PII REDACTION]\n")
        f.write(f"Items Redacted: {len(results['pii_results'])}\n")
        if results['pii_results']:
            for pii in results['pii_results']:
                f.write(f"  - Type: {pii.entity_type}, Text: '{results['transcript'][pii.start:pii.end]}'\n")
        f.write(f"Final Redacted Transcript saved to {TRANSCRIPT_FILENAME}\n\n")

        f.write(f"[SUMMARIZATION]\n")
        f.write(f"Summary Text:\n---\n{results['summary']}\n---\n")
        f.write(f"Audio Summary saved to {SUMMARY_FILENAME}\n\n")

        f.write("--- END OF LOG ---")
    print(f"✅ Saved audit log to: {log_path}")


if __name__ == "__main__":
    main()