import os
import re
import uuid
from typing import Dict, Any

import librosa
import numpy as np
import spacy
from google.cloud import speech, texttospeech

# Load spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- Helper Functions from your scripts ---

def _calculate_snr(audio_path: str) -> float:
    y, sr = librosa.load(audio_path, sr=16000)
    signal_power = np.mean(y ** 2)
    noise_power = np.var(y)
    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

def _calculate_word_perplexity(words) -> float:
    confidences = [word.confidence for word in words]
    avg_confidence = np.mean(confidences)
    return 1.0 / avg_confidence if avg_confidence > 0 else float('inf')

def _redact_pii_regex(text: str) -> str:
    patterns = {
        'CREDIT_CARD': r'\b(?:\d[ -]*?){13,16}\b', # More robust CC regex
        'SSN': r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        'PHONE': r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
    }
    redacted_text = text
    for pii_type, pattern in patterns.items():
        redacted_text = re.sub(pattern, f'[REDACTED_{pii_type}]', redacted_text)
    return redacted_text

def _redact_pii_ner(text: str) -> str:
    doc = nlp(text)
    redacted_text = text
    entities = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)
    for ent in entities:
        if ent.label_ in ['PERSON', 'DATE', 'GPE', 'ORG']: # Redact names, dates, locations, organizations
            redacted_text = redacted_text[:ent.start_char] + f'[REDACTED_{ent.label_}]' + redacted_text[ent.end_char:]
    return redacted_text

def _summarize_text(text: str, max_sentences=3) -> str:
    sentences = text.split('. ')
    summary = '. '.join(sentences[:max_sentences])
    if not summary.endswith('.'):
        summary += '.'
    return summary

def _text_to_speech(text: str, output_dir: str) -> str:
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Neural2-A")
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    
    output_filename = f"summary_{uuid.uuid4()}.mp3"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'wb') as out:
        out.write(response.audio_content)
    
    return output_filename

# --- Main Pipeline Function ---

def process_audio_file(audio_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Runs the full audio processing pipeline.
    """
    results = {}
    
    # 1. Basic Transcription
    speech_client = speech.SpeechClient()
    with open(audio_path, "rb") as audio_file:
        content = audio_file.read()
    
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3, # Assuming MP3
        language_code="en-US",
        enable_automatic_punctuation=True,
        enable_word_confidence=True,
    )
    
    response = speech_client.recognize(config=config, audio=audio)
    
    if not response.results or not response.results[0].alternatives:
        raise ValueError("Could not transcribe audio. The file might be empty or corrupt.")

    alternative = response.results[0].alternatives[0]
    results['transcript'] = alternative.transcript
    
    # 2. Confidence Scoring
    api_confidence = alternative.confidence
    words = alternative.words
    snr = _calculate_snr(audio_path)
    perplexity = _calculate_word_perplexity(words)
    
    snr_normalized = min(max((snr - 10) / 20, 0), 1)
    perplexity_normalized = max(1 - (perplexity - 1), 0)
    
    combined_score = (0.5 * api_confidence + 0.3 * snr_normalized + 0.2 * perplexity_normalized)
    
    if combined_score > 0.85: confidence_level = "HIGH"
    elif combined_score > 0.70: confidence_level = "MEDIUM"
    else: confidence_level = "LOW"
    
    results['confidence_score'] = f"{combined_score:.2f}"
    results['confidence_level'] = confidence_level

    # 3. PII Redaction
    redacted_regex = _redact_pii_regex(results['transcript'])
    redacted_final = _redact_pii_ner(redacted_regex)
    results['redacted_transcript'] = redacted_final
    
    # 4. Summarization
    summary = _summarize_text(results['transcript'])
    results['summary'] = summary
    
    # 5. Text-to-Speech for Summary
    summary_audio_filename = _text_to_speech(summary, output_dir)
    results['summary_audio_filename'] = summary_audio_filename

    return results