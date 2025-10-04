# audio_processor.py - Audio Processing Module
import threading
import queue
import numpy as np
import sounddevice as sd
import webrtcvad
import time
from collections import namedtuple

# AudioSegment data structure
AudioSegment = namedtuple('AudioSegment', ['id', 'data', 'timestamp'])

class AudioStream:
    """Handles sounddevice audio input and buffering"""
    
    def __init__(self, sample_rate=16000, chunk_duration=3.0, buffer_duration=30.0):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        
        # Continuous buffer settings
        self.buffer_duration = buffer_duration
        self.buffer_size = int(sample_rate * buffer_duration)
        self.continuous_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_index = 0
        self.buffer_lock = threading.Lock()
        
        # Audio processing
        self.raw_audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None
    
    def audio_callback(self, indata, frames, time_info, status):
        """Real-time audio input callback"""
        if status:
            print(f"Audio status: {status}")
        
        if self.is_recording:
            # Convert to float32 and add to processing queue
            audio_data = indata.flatten().astype(np.float32)
            
            # Add to audio processing queue
            self.raw_audio_queue.put(audio_data.copy())
            
            # Update continuous circular buffer for recovery
            with self.buffer_lock:
                data_len = len(audio_data)
                if self.buffer_index + data_len <= self.buffer_size:
                    # Fits in current position
                    self.continuous_buffer[self.buffer_index:self.buffer_index + data_len] = audio_data
                    self.buffer_index += data_len
                else:
                    # Wrap around
                    remaining = self.buffer_size - self.buffer_index
                    self.continuous_buffer[self.buffer_index:] = audio_data[:remaining]
                    overflow = data_len - remaining
                    if overflow > 0:
                        self.continuous_buffer[:overflow] = audio_data[remaining:]
                    self.buffer_index = overflow
                
                # Reset index if we've filled the buffer
                if self.buffer_index >= self.buffer_size:
                    self.buffer_index = 0
    
    def start_audio_stream(self):
        """Start real-time audio capture"""
        print("🎤 Starting audio stream...")
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self.audio_callback,
            blocksize=1024,
        )
        self.stream.start()
        self.is_recording = True
        print("✅ Audio stream started")
    
    def stop_audio_stream(self):
        """Stop audio capture"""
        if hasattr(self, 'stream') and self.stream:
            self.is_recording = False
            self.stream.stop()
            self.stream.close()
            print("🛑 Audio stream stopped")
    
    def get_recent_audio_buffer(self, duration_seconds=10.0):
        """Get recent audio from the continuous buffer for recovery purposes"""
        with self.buffer_lock:
            samples_needed = int(duration_seconds * self.sample_rate)
            samples_needed = min(samples_needed, self.buffer_size)
            
            if self.buffer_index >= samples_needed:
                # Simple case - no wrap around
                return self.continuous_buffer[self.buffer_index - samples_needed:self.buffer_index].copy()
            else:
                # Handle wrap around
                start_index = self.buffer_size - (samples_needed - self.buffer_index)
                part1 = self.continuous_buffer[start_index:].copy()
                part2 = self.continuous_buffer[:self.buffer_index].copy()
                return np.concatenate([part1, part2])

class VoiceActivityDetector:
    """Robust VAD with noise rejection and adaptive thresholds"""
    
    def __init__(self, sample_rate=16000, aggressiveness=2):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(aggressiveness)
        
        # Energy settings
        self.background_noise_level = 0.01  # Will be calibrated
        self.energy_threshold_multiplier = 3.0  # Energy must be 3x background noise
        self.min_energy_threshold = 0.005  # Absolute minimum energy threshold
        self.energy_history = []  # For background noise estimation
        self.calibration_frames = 0
        self.max_calibration_frames = 100  # ~6 seconds of calibration
        
        # Speech confidence tracking
        self.speech_confidence_history = []
        self.confidence_window_size = 10  # Smooth over 10 frames
    
    def detect_speech(self, audio_chunk):
        """Multi-factor speech detection with noise rejection"""
        try:
            # First check: Energy gate - fast rejection of low-energy noise
            if not self.passes_energy_gate(audio_chunk):
                return False
            
            # Second check: VAD analysis
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            
            # Split into 30ms frames for VAD (480 samples at 16kHz)
            frame_size = 480
            frames = [audio_int16[i:i+frame_size] for i in range(0, len(audio_int16), frame_size)]
            
            # Count speech frames detected by VAD
            speech_frames = 0
            total_frames = 0
            for frame in frames:
                if len(frame) == frame_size:
                    total_frames += 1
                    try:
                        if self.vad.is_speech(frame.tobytes(), self.sample_rate):
                            speech_frames += 1
                    except:
                        continue
            
            if total_frames == 0:
                return False
            
            # Higher threshold - require at least 30% of frames to be speech
            vad_confidence = speech_frames / total_frames
            vad_passes = vad_confidence >= 0.3
            
            # Third check: Spectral analysis for speech-like characteristics
            spectral_centroid = self.calculate_spectral_centroid(audio_chunk)
            # Human speech typically has spectral centroid between 500-3000 Hz
            spectral_passes = 500 <= spectral_centroid <= 3000
            
            # Calculate overall confidence score
            energy = self.calculate_rms_energy(audio_chunk)
            energy_ratio = energy / max(self.background_noise_level, 0.001)
            
            # Combine factors into confidence score
            confidence = 0.0
            if vad_passes:
                confidence += 0.4 * vad_confidence
            if spectral_passes:
                confidence += 0.3
            if energy_ratio > 2.0:
                confidence += 0.3 * min(energy_ratio / 10.0, 1.0)
            
            # Add to confidence history for smoothing
            self.speech_confidence_history.append(confidence)
            if len(self.speech_confidence_history) > self.confidence_window_size:
                self.speech_confidence_history.pop(0)
            
            # Smooth confidence over recent frames
            smoothed_confidence = np.mean(self.speech_confidence_history)
            
            # Final decision: require significant confidence
            is_speech = smoothed_confidence >= 0.5
            
            # Debug output during calibration
            if self.calibration_frames < self.max_calibration_frames:
                if self.calibration_frames % 20 == 0:  # Every ~1.3 seconds
                    print(f"🔧 [CALIBRATION] Energy: {energy:.6f}, VAD: {vad_confidence:.2f}, Spectral: {spectral_centroid:.0f}Hz, Confidence: {smoothed_confidence:.2f}")
            
            return is_speech
            
        except Exception as e:
            print(f"⚠️ [VAD] Error in voice activity detection: {e}")
            return False  # Fail safe - reject on error
    
    def calculate_rms_energy(self, audio_chunk):
        """Calculate RMS energy of audio chunk"""
        try:
            return np.sqrt(np.mean(np.square(audio_chunk.astype(np.float32))))
        except:
            return 0.0
    
    def calculate_spectral_centroid(self, audio_chunk):
        """Calculate spectral centroid to help distinguish speech from noise"""
        try:
            # Simple spectral centroid calculation
            fft = np.fft.rfft(audio_chunk)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_chunk), 1.0/self.sample_rate)
            
            if np.sum(magnitude) == 0:
                return 0
            
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            return centroid
        except:
            return 0.0
    
    def passes_energy_gate(self, audio_chunk):
        """Check if audio chunk passes energy-based filtering"""
        energy = self.calculate_rms_energy(audio_chunk)
        self.update_background_noise(energy)
        
        # Dynamic threshold based on background noise
        energy_threshold = max(
            self.background_noise_level * self.energy_threshold_multiplier,
            self.min_energy_threshold
        )
        
        return energy > energy_threshold
    
    def update_background_noise(self, energy_level):
        """Update background noise estimation during calibration and quiet periods"""
        if self.calibration_frames < self.max_calibration_frames:
            # Still calibrating
            self.energy_history.append(energy_level)
            self.calibration_frames += 1
            
            if self.calibration_frames == self.max_calibration_frames:
                # Calibration complete
                if self.energy_history:
                    self.background_noise_level = np.percentile(self.energy_history, 75)  # Use 75th percentile
                    print(f"🔧 [CALIBRATION] Background noise level: {self.background_noise_level:.6f}")
                else:
                    self.background_noise_level = self.min_energy_threshold
        else:
            # Ongoing noise level updates during silence - placeholder for future implementation
            pass

class UtteranceSegmenter:
    """Smart conversation segmentation with boundary detection"""
    
    def __init__(self, min_utterance_length=1.0, pause_threshold=1.5, min_consecutive_speech=3):
        # Utterance boundary detection
        self.min_utterance_length = min_utterance_length  # Minimum seconds for valid utterance
        self.pause_threshold = pause_threshold  # Seconds of silence to end utterance
        self.min_consecutive_speech = min_consecutive_speech  # Frames needed to start utterance
        
        # State tracking
        self.is_in_utterance = False
        self.speech_frames_count = 0
        self.silence_frames_count = 0
        self.current_utterance_start = 0
        
        # Thread synchronization
        self.utterance_lock = threading.Lock()
    
    def detect_utterance_boundary(self, audio_chunk, has_speech):
        """Detect utterance boundaries with thread-safe state management"""
        # Each chunk is typically 1024 samples at 16kHz = ~64ms
        frame_duration = len(audio_chunk) / 16000  # Assuming 16kHz sample rate
        frames_per_second = 1.0 / frame_duration
        
        with self.utterance_lock:  # Thread-safe state management
            if has_speech:
                self.speech_frames_count += 1
                self.silence_frames_count = 0
                
                # Require minimum consecutive speech frames before starting utterance
                if not self.is_in_utterance and self.speech_frames_count >= self.min_consecutive_speech:
                    # Start of new utterance - confirmed by consecutive speech frames
                    self.is_in_utterance = True
                    self.current_utterance_start = time.time()
                    print("🎙️ [VOICE] Speech detected - starting utterance...")
                    return "utterance_start"
                    
            else:
                self.silence_frames_count += 1
                
                # Reset speech count if we don't have enough consecutive frames yet
                if not self.is_in_utterance:
                    self.speech_frames_count = 0
                
                # Check if we've had enough silence to end utterance
                silence_duration = self.silence_frames_count / frames_per_second
                
                if self.is_in_utterance and silence_duration >= self.pause_threshold:
                    # End of utterance
                    utterance_duration = time.time() - self.current_utterance_start
                    self.is_in_utterance = False
                    self.speech_frames_count = 0  # Reset for next utterance
                    
                    # Only consider it a valid utterance if it was long enough
                    if utterance_duration >= self.min_utterance_length:
                        print(f"🎙️ [VOICE] Utterance complete ({utterance_duration:.1f}s)")
                        return "utterance_end"
                    else:
                        print(f"🎙️ [VOICE] Utterance too short ({utterance_duration:.1f}s), ignoring")
                        return "utterance_too_short"
            
            return "continuing" if self.is_in_utterance else "silence"

class AudioProcessor:
    """Main audio processing coordinator that combines all audio components"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
        # Initialize components
        self.audio_stream = AudioStream(sample_rate=sample_rate)
        self.vad = VoiceActivityDetector(sample_rate=sample_rate)
        self.utterance_segmenter = UtteranceSegmenter()
        
        # Audio processing state
        self.current_utterance_buffer = []
        self.processing_callbacks = {}  # Callbacks for different events
    
    def register_callback(self, event_type, callback):
        """Register callbacks for audio events"""
        if event_type not in self.processing_callbacks:
            self.processing_callbacks[event_type] = []
        self.processing_callbacks[event_type].append(callback)
    
    def emit_event(self, event_type, data=None):
        """Emit events to registered callbacks"""
        if event_type in self.processing_callbacks:
            for callback in self.processing_callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"⚠️ [AUDIO] Callback error: {e}")
    
    def start_processing(self):
        """Start audio processing pipeline"""
        self.audio_stream.start_audio_stream()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio_loop, daemon=True)
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop audio processing"""
        self.audio_stream.stop_audio_stream()
    
    def _process_audio_loop(self):
        """Main audio processing loop"""
        while self.audio_stream.is_recording:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_stream.raw_audio_queue.get(timeout=0.1)
                
                # Detect speech
                has_speech = self.vad.detect_speech(audio_chunk)
                
                # Detect utterance boundaries
                boundary_event = self.utterance_segmenter.detect_utterance_boundary(audio_chunk, has_speech)
                
                # Handle boundary events
                if boundary_event == "utterance_start":
                    self.current_utterance_buffer = [audio_chunk]
                    self.emit_event("utterance_start", audio_chunk)
                    
                elif boundary_event == "continuing" and self.utterance_segmenter.is_in_utterance:
                    self.current_utterance_buffer.append(audio_chunk)
                    self.emit_event("audio_chunk", audio_chunk)
                    
                elif boundary_event == "utterance_end":
                    if self.current_utterance_buffer:
                        # Concatenate all audio chunks from this utterance
                        complete_utterance = np.concatenate(self.current_utterance_buffer)
                        self.emit_event("utterance_complete", complete_utterance)
                        self.current_utterance_buffer = []
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ [AUDIO] Processing error: {e}")
                continue