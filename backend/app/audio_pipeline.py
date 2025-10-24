import os
import uuid
import numpy as np
from gtts import gTTS
from transformers import pipeline
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Note: The models themselves will be loaded once in main.py for efficiency.

def process_audio_file(
    audio_path: str,
    output_dir: str,
    whisper_model,
    summarizer,
    analyzer,
    anonymizer
) -> dict:
    """
    Runs the full audio processing pipeline using local open-source models.
    """
    results = {}
    print("Step 1: Transcribing audio with Whisper...")
    
    # 1. Transcription with Whisper
    transcription_result = whisper_model.transcribe(audio_path, fp16=False) # Set fp16=False if not using a GPU
    results['transcript'] = transcription_result['text']

    # 2. Confidence Scoring (simplified for Whisper)
    # Whisper provides average log probability for segments. We can use this as a confidence proxy.
    # Lower logprob is better (less negative). We'll convert it to a 0-1 scale.
    if transcription_result['segments']:
        logprobs = [segment['avg_logprob'] for segment in transcription_result['segments']]
        avg_logprob = np.mean(logprobs)
        # np.exp brings it to a 0-1 probability scale. Closer to 1 is more confident.
        confidence = np.exp(avg_logprob) 
    else:
        confidence = 0.0

    results['confidence_score'] = f"{confidence:.2f}"
    if confidence > 0.85:
        results['confidence_level'] = "HIGH"
    elif confidence > 0.65:
        results['confidence_level'] = "MEDIUM"
    else:
        results['confidence_level'] = "LOW"
    
    print(f"Step 2: Transcription complete. Confidence: {results['confidence_level']}")

    # 3. PII Redaction with Presidio
    print("Step 3: Redacting PII with Presidio...")
    analyzer_results = analyzer.analyze(
        text=results['transcript'],
        entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD_NUMBER", "LOCATION"],
        language='en'
    )
    anonymized_result = anonymizer.anonymize(
        text=results['transcript'],
        analyzer_results=analyzer_results
    )
    results['redacted_transcript'] = anonymized_result.text
    print("Step 3: PII Redaction complete.")

    # 4. Summarization with Transformers (BART)
    print("Step 4: Summarizing text...")
    # Summarization is best for longer texts.
    if len(results['transcript'].split()) > 50:
        summary = summarizer(
            results['transcript'],
            max_length=150,
            min_length=30,
            do_sample=False
        )
        results['summary'] = summary[0]['summary_text']
    else:
        results['summary'] = "Text is too short to summarize."
    print("Step 4: Summarization complete.")
    
    # 5. Text-to-Speech with gTTS
    print("Step 5: Generating audio summary with gTTS...")
    tts = gTTS(results['summary'], lang='en')
    
    output_filename = f"summary_{uuid.uuid4()}.mp3"
    output_path = os.path.join(output_dir, output_filename)
    tts.save(output_path)
    
    results['summary_audio_filename'] = output_filename
    print("Step 5: Audio summary generated.")

    return {
        **results,  # Unpack existing results dict
        "pii_results": analyzer_results # Add the list of found PII entities
    }