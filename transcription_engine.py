# transcription_engine.py - Speech-to-Text Processing Module
import threading
import queue
import time
import numpy as np
import whisper
from typing import Optional, Dict, Any, Callable, List
import logging
from dataclasses import dataclass
from collections import namedtuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Transcription result structure
TranscriptionResult = namedtuple('TranscriptionResult', [
    'text', 'confidence', 'segments', 'language', 'timestamp', 'processing_time'
])

@dataclass
class AudioChunk:
    """Audio data container for processing"""
    data: np.ndarray
    timestamp: float
    chunk_id: str
    sample_rate: int = 16000

class SpeechTranscriber:
    """Whisper-based transcription with medical terminology optimization"""
    
    def __init__(self, model_name: str = "base", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.model_lock = threading.Lock()
        self.load_model()
        
        # Medical terminology enhancement
        self.medical_terms = self._load_medical_vocabulary()
        
    def load_model(self):
        """Load Whisper model with error handling"""
        try:
            with self.model_lock:
                logger.info(f"Loading Whisper model: {self.model_name}")
                self.model = whisper.load_model(self.model_name, device=self.device)
                logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise
    
    def _load_medical_vocabulary(self) -> List[str]:
        """Load medical terminology for enhanced recognition"""
        # Common medical terms that might be misrecognized
        medical_terms = [
            "hypertension", "diabetes", "cardiovascular", "myocardial", "infarction",
            "pneumonia", "bronchitis", "asthma", "allergies", "medication",
            "prescription", "dosage", "milligrams", "symptoms", "diagnosis",
            "treatment", "therapy", "surgery", "procedure", "examination",
            "blood pressure", "heart rate", "temperature", "weight", "height",
            "patient", "clinic", "hospital", "doctor", "nurse", "physician",
            "stethoscope", "thermometer", "sphygmomanometer", "otoscope"
        ]
        return medical_terms
    
    def transcribe_speech(self, audio_chunk: AudioChunk) -> TranscriptionResult:
        """Transcribe audio with confidence scoring and medical optimization"""
        start_time = time.time()
        
        try:
            with self.model_lock:
                # Whisper expects float32 audio normalized to [-1, 1]
                if audio_chunk.data.dtype != np.float32:
                    audio_data = audio_chunk.data.astype(np.float32)
                else:
                    audio_data = audio_chunk.data
                
                # Normalize audio if needed
                if np.max(np.abs(audio_data)) > 1.0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                # Perform transcription
                result = self.model.transcribe(
                    audio_data,
                    language="en",  # Focus on English for medical contexts
                    task="transcribe",
                    verbose=False,
                    word_timestamps=True
                )
                
                # Extract segments with confidence
                segments = []
                overall_confidence = 0.0
                
                if "segments" in result:
                    for segment in result["segments"]:
                        segment_info = {
                            "start": segment.get("start", 0),
                            "end": segment.get("end", 0),
                            "text": segment.get("text", ""),
                            "confidence": self._estimate_confidence(segment)
                        }
                        segments.append(segment_info)
                        overall_confidence += segment_info["confidence"]
                    
                    if segments:
                        overall_confidence /= len(segments)
                else:
                    overall_confidence = 0.5  # Default confidence
                
                # Post-process text for medical terminology
                processed_text = self._enhance_medical_terminology(result["text"])
                
                processing_time = time.time() - start_time
                
                return TranscriptionResult(
                    text=processed_text,
                    confidence=overall_confidence,
                    segments=segments,
                    language=result.get("language", "en"),
                    timestamp=audio_chunk.timestamp,
                    processing_time=processing_time
                )
                
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            processing_time = time.time() - start_time
            return TranscriptionResult(
                text="[Transcription Error]",
                confidence=0.0,
                segments=[],
                language="en",
                timestamp=audio_chunk.timestamp,
                processing_time=processing_time
            )
    
    def _estimate_confidence(self, segment: Dict[str, Any]) -> float:
        """Estimate confidence score for a segment"""
        # Whisper doesn't provide direct confidence scores
        # We estimate based on various factors
        
        text = segment.get("text", "").strip()
        if not text:
            return 0.0
        
        # Base confidence
        confidence = 0.7
        
        # Adjust based on text characteristics
        if len(text) < 3:
            confidence -= 0.2  # Very short text often unreliable
        
        if any(term in text.lower() for term in self.medical_terms):
            confidence += 0.1  # Medical terms often recognized well
        
        # Check for common transcription artifacts
        if "[" in text or "]" in text or "♪" in text:
            confidence -= 0.3
        
        if text.count(" ") == 0 and len(text) > 10:
            confidence -= 0.2  # Likely merged words
        
        return max(0.0, min(1.0, confidence))
    
    def _enhance_medical_terminology(self, text: str) -> str:
        """Post-process text to correct common medical term misrecognitions"""
        if not text:
            return text
        
        # Common medical term corrections
        corrections = {
            "high per tension": "hypertension",
            "die a beetus": "diabetes", 
            "die beetus": "diabetes",
            "cardio vascular": "cardiovascular",
            "my cardial": "myocardial",
            "in farction": "infarction",
            "new monia": "pneumonia",
            "bron kitis": "bronchitis",
            "prescription": "prescription",
            "mill a grams": "milligrams",
            "milli grams": "milligrams",
            "die ag no sis": "diagnosis",
            "exam in nation": "examination",
            "blood pressure": "blood pressure",
            "heart rate": "heart rate",
            "steth o scope": "stethoscope"
        }
        
        corrected_text = text
        for wrong, correct in corrections.items():
            corrected_text = corrected_text.replace(wrong, correct)
        
        return corrected_text
    
    def handle_audio_quality_issues(self, audio_data: np.ndarray) -> np.ndarray:
        """Pre-process audio to handle quality issues"""
        # Apply noise reduction and normalization
        processed_audio = audio_data.copy()
        
        # Simple noise gate
        noise_threshold = 0.01
        processed_audio[np.abs(processed_audio) < noise_threshold] *= 0.1
        
        # Normalize
        if np.max(np.abs(processed_audio)) > 0:
            processed_audio = processed_audio / np.max(np.abs(processed_audio)) * 0.95
        
        return processed_audio

class TranscriptionQueue:
    """Robust processing pipeline for handling transcription requests"""
    
    def __init__(self, transcriber: SpeechTranscriber, max_queue_size: int = 100):
        self.transcriber = transcriber
        self.max_queue_size = max_queue_size
        self.processing_queue = queue.Queue(maxsize=max_queue_size)
        self.result_callbacks = {}
        self.processing_thread = None
        self.is_running = False
        self.stats = {
            "processed": 0,
            "failed": 0,
            "queue_overflows": 0,
            "avg_processing_time": 0.0
        }
        self.stats_lock = threading.Lock()
    
    def start(self):
        """Start the processing thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
        logger.info("Transcription queue processing started")
    
    def stop(self):
        """Stop the processing thread"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        logger.info("Transcription queue processing stopped")
    
    def queue_audio_for_processing(self, audio_chunk: AudioChunk, 
                                  callback: Optional[Callable[[TranscriptionResult], None]] = None) -> bool:
        """Add audio chunk to processing queue"""
        try:
            # Check queue capacity
            if self.processing_queue.qsize() >= self.max_queue_size:
                with self.stats_lock:
                    self.stats["queue_overflows"] += 1
                logger.warning("Transcription queue full, dropping audio chunk")
                return False
            
            # Store callback if provided
            if callback:
                self.result_callbacks[audio_chunk.chunk_id] = callback
            
            self.processing_queue.put(audio_chunk, block=False)
            return True
            
        except queue.Full:
            with self.stats_lock:
                self.stats["queue_overflows"] += 1
            logger.warning("Failed to queue audio chunk - queue full")
            return False
    
    def _process_queue(self):
        """Process audio chunks from queue"""
        while self.is_running:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.processing_queue.get(timeout=1.0)
                
                # Process transcription
                result = self.transcriber.transcribe_speech(audio_chunk)
                
                # Update stats
                with self.stats_lock:
                    self.stats["processed"] += 1
                    if self.stats["avg_processing_time"] == 0:
                        self.stats["avg_processing_time"] = result.processing_time
                    else:
                        # Running average
                        self.stats["avg_processing_time"] = (
                            self.stats["avg_processing_time"] * 0.9 + 
                            result.processing_time * 0.1
                        )
                
                # Call result callback if exists
                if audio_chunk.chunk_id in self.result_callbacks:
                    callback = self.result_callbacks.pop(audio_chunk.chunk_id)
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Error in transcription callback: {e}")
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue  # Timeout, check if still running
            except Exception as e:
                logger.error(f"Error processing transcription queue: {e}")
                with self.stats_lock:
                    self.stats["failed"] += 1
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        with self.stats_lock:
            return {
                "queue_size": self.processing_queue.qsize(),
                "max_queue_size": self.max_queue_size,
                "processed_total": self.stats["processed"],
                "failed_total": self.stats["failed"],
                "queue_overflows": self.stats["queue_overflows"],
                "avg_processing_time": self.stats["avg_processing_time"],
                "is_running": self.is_running
            }
    
    def handle_processing_backlog(self):
        """Handle processing backlog by prioritizing recent audio"""
        if self.processing_queue.qsize() < self.max_queue_size * 0.8:
            return  # No backlog
        
        logger.warning(f"Processing backlog detected: {self.processing_queue.qsize()} items")
        # For now, just log the backlog. In a production system,
        # we might implement priority queuing or selective dropping

class TextProcessor:
    """Post-process transcribed text for clinical documentation"""
    
    def __init__(self):
        self.sentence_endings = [".", "!", "?"]
        self.common_filler_words = ["um", "uh", "er", "ah", "like", "you know"]
    
    def clean_transcription_text(self, text: str) -> str:
        """Clean and normalize transcribed text"""
        if not text:
            return ""
        
        # Remove filler words
        cleaned_text = self._remove_filler_words(text)
        
        # Fix capitalization
        cleaned_text = self._fix_capitalization(cleaned_text)
        
        # Add proper punctuation
        cleaned_text = self._improve_punctuation(cleaned_text)
        
        # Remove extra whitespace
        cleaned_text = " ".join(cleaned_text.split())
        
        return cleaned_text
    
    def _remove_filler_words(self, text: str) -> str:
        """Remove common filler words"""
        words = text.split()
        filtered_words = []
        
        for word in words:
            cleaned_word = word.lower().strip(".,!?")
            if cleaned_word not in self.common_filler_words:
                filtered_words.append(word)
        
        return " ".join(filtered_words)
    
    def _fix_capitalization(self, text: str) -> str:
        """Fix capitalization for sentences"""
        if not text:
            return text
        
        # Capitalize first letter
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Capitalize after sentence endings
        for ending in self.sentence_endings:
            parts = text.split(ending + " ")
            if len(parts) > 1:
                capitalized_parts = [parts[0]]
                for part in parts[1:]:
                    if part:
                        capitalized_parts.append(part[0].upper() + part[1:] if len(part) > 1 else part.upper())
                text = (ending + " ").join(capitalized_parts)
        
        return text
    
    def _improve_punctuation(self, text: str) -> str:
        """Add appropriate punctuation"""
        if not text:
            return text
        
        # Ensure text ends with punctuation
        if not any(text.endswith(ending) for ending in self.sentence_endings):
            text += "."
        
        return text
    
    def handle_common_errors(self, text: str) -> str:
        """Handle common transcription errors"""
        if not text:
            return text
        
        # Common medical transcription corrections
        corrections = {
            " ,": ",",
            " .": ".",
            " !": "!",
            " ?": "?",
            "  ": " ",  # Multiple spaces
            "patient's patient": "patient",
            "the the": "the",
            "and and": "and"
        }
        
        corrected_text = text
        for wrong, correct in corrections.items():
            corrected_text = corrected_text.replace(wrong, correct)
        
        return corrected_text
    
    def format_output_text(self, text: str, format_type: str = "clinical") -> str:
        """Format text for specific output types"""
        if not text:
            return text
        
        cleaned_text = self.clean_transcription_text(text)
        
        if format_type == "clinical":
            # Format for clinical documentation
            return self._format_clinical_text(cleaned_text)
        elif format_type == "conversation":
            # Format for conversation display
            return self._format_conversation_text(cleaned_text)
        else:
            return cleaned_text
    
    def _format_clinical_text(self, text: str) -> str:
        """Format text for clinical documentation"""
        # Ensure proper medical formatting
        formatted_text = text
        
        # Capitalize medical abbreviations
        medical_abbrevs = ["BP", "HR", "RR", "O2", "CBC", "EKG", "ECG", "MRI", "CT", "X-ray"]
        for abbrev in medical_abbrevs:
            formatted_text = formatted_text.replace(abbrev.lower(), abbrev)
        
        return formatted_text
    
    def _format_conversation_text(self, text: str) -> str:
        """Format text for conversation display"""
        # Add timestamps or speaker identification if needed
        return text