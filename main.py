# main.py - Clinical Documentation System Orchestrator
import threading
import time
import queue
import signal
import sys
from typing import Dict, Any, Optional
import logging

# Import all modules from the refactored system
from config import init_config
from audio_processor import AudioStream
from storage_manager import SessionManager, FormStateStorage, SchemaStorage, AudioArchiver
from transcription_engine import SpeechTranscriber, TranscriptionQueue, TextProcessor, AudioChunk
from llm_service import LLMClient
from mobile_interface import MobileAudioServer
from form_processor import StructuredFormProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClinicalDocumentationSystem:
    """Main orchestrator for the clinical documentation system"""
    
    def __init__(self, config_file: Optional[str] = None):
        # Initialize configuration
        self.config_manager = init_config(config_file)
        self.config = self.config_manager.get_config()
        
        # System state
        self.is_running = False
        self.current_session_id = None
        self.shutdown_event = threading.Event()
        
        # Component instances
        self.session_manager = None
        self.audio_stream = None
        self.transcription_queue = None
        self.llm_client = None
        self.mobile_server = None
        self.form_processor = None
        
        # Processing queues and threads
        self.audio_processing_thread = None
        self.transcription_callbacks = queue.Queue()
        self.system_stats = {
            "start_time": None,
            "sessions_created": 0,
            "audio_segments_processed": 0,
            "transcriptions_completed": 0,
            "forms_created": 0,
            "clinical_analyses_performed": 0
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def initialize_components(self) -> bool:
        """Initialize all system components"""
        try:
            logger.info("Initializing Clinical Documentation System...")
            
            # Initialize storage components
            self.session_manager = SessionManager(self.config.storage.base_dir)
            
            self.form_state_storage = FormStateStorage(self.session_manager)
            self.schema_storage = SchemaStorage(self.config.storage.forms_dir)
            self.audio_archiver = AudioArchiver(self.session_manager)
            
            # Initialize audio processing components
            if self.config.enable_mobile_interface:
                logger.info("Initializing mobile interface...")
                self.mobile_server = MobileAudioServer(
                    host=self.config.mobile.host,
                    port=self.config.mobile.port
                )
                # Set up audio callback
                self.mobile_server.on_audio_data = self._handle_mobile_audio
            else:
                logger.info("Initializing local audio stream...")
                self.audio_stream = AudioStream(
                    sample_rate=self.config.audio.sample_rate,
                    chunk_duration=self.config.audio.chunk_duration,
                    buffer_duration=self.config.audio.buffer_duration
                )
            
            # Initialize transcription components
            logger.info("Initializing transcription engine...")
            self.speech_transcriber = SpeechTranscriber(
                model_name=self.config.transcription.model_size,
                device=self.config.transcription.device
            )
            
            self.transcription_queue = TranscriptionQueue(
                transcriber=self.speech_transcriber,
                max_queue_size=self.config.transcription.queue_size
            )
            
            self.text_processor = TextProcessor()
            
            # Initialize LLM components
            if self.config.enable_clinical_ai:
                logger.info("Initializing LLM service...")
                self.llm_client = LLMClient(
                    provider=self.config.llm.provider,
                    base_url=self.config.llm.base_url,
                    model_name=self.config.llm.model_name
                )
                
                # Initialize form processor
                logger.info("Initializing structured form processor...")
                self.form_processor = StructuredFormProcessor(
                    llm_client=self.llm_client,
                    schema_storage=self.schema_storage,
                    form_storage=self.form_state_storage
                )
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False
    
    def start_system(self) -> bool:
        """Start the clinical documentation system"""
        if self.is_running:
            logger.warning("System is already running")
            return True
        
        if not self.initialize_components():
            logger.error("Failed to initialize components")
            return False
        
        try:
            logger.info("Starting Clinical Documentation System...")
            self.is_running = True
            self.system_stats["start_time"] = time.time()
            
            # Start transcription processing
            self.transcription_queue.start()
            
            # Start mobile interface if enabled
            if self.config.enable_mobile_interface and self.mobile_server:
                self.mobile_server.start()
                logger.info(f"Mobile interface started on {self.config.mobile.host}:{self.config.mobile.port}")
            
            # Start local audio processing if enabled
            elif self.audio_stream:
                self.audio_stream.start_recording()
                self.audio_processing_thread = threading.Thread(
                    target=self._process_audio_stream,
                    daemon=True
                )
                self.audio_processing_thread.start()
                logger.info("Local audio processing started")
            
            # Start background tasks
            self._start_background_tasks()
            
            logger.info("Clinical Documentation System started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            self.is_running = False
            return False
    
    def stop_system(self):
        """Stop the clinical documentation system gracefully"""
        if not self.is_running:
            return
        
        logger.info("Stopping Clinical Documentation System...")
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop audio processing
        if self.audio_stream:
            self.audio_stream.stop_recording()
        
        if self.mobile_server:
            self.mobile_server.stop()
        
        # Stop transcription processing
        if self.transcription_queue:
            self.transcription_queue.stop()
        
        # Close current session if active
        if self.current_session_id:
            self.session_manager.close_session(self.current_session_id)
        
        logger.info("Clinical Documentation System stopped")
    
    def create_new_session(self, session_context: Optional[Dict[str, Any]] = None) -> str:
        """Create a new clinical documentation session"""
        session_id = self.session_manager.create_session()
        
        if session_context:
            # Update session metadata with context
            metadata = self.session_manager.get_session_metadata(session_id)
            metadata.update(session_context)
            self.session_manager.save_session_metadata(session_id, metadata)
        
        self.current_session_id = session_id
        self.system_stats["sessions_created"] += 1
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def _handle_mobile_audio(self, client_id: str, audio_data, client_info: Dict[str, Any]):
        """Handle audio data from mobile client"""
        if not self.current_session_id:
            self.create_new_session({"source": "mobile", "client_id": client_id, "client": client_info})
        
        # audio_data is already a numpy array from mobile interface
        self._process_audio_chunk(audio_data)
    
    def _process_audio_stream(self):
        """Process audio from local audio stream"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                if not self.audio_stream.raw_audio_queue.empty():
                    audio_data = self.audio_stream.raw_audio_queue.get(timeout=1.0)
                    self._process_audio_chunk(audio_data)
                else:
                    time.sleep(0.1)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audio stream: {e}")
    
    def _process_audio_chunk(self, audio_data):
        """Process a chunk of audio data"""
        if not self.current_session_id:
            self.create_new_session({"source": "local_audio"})
        
        # Create audio chunk for transcription
        chunk_id = f"chunk_{int(time.time() * 1000)}"
        audio_chunk = AudioChunk(
            data=audio_data,
            timestamp=time.time(),
            chunk_id=chunk_id,
            sample_rate=self.config.audio.sample_rate
        )
        
        # Save audio segment for zero-loss guarantee
        self.audio_archiver.save_audio_segment(
            session_id=self.current_session_id,
            audio_data=audio_data,
            segment_id=chunk_id,
            timestamp=audio_chunk.timestamp,
            sample_rate=self.config.audio.sample_rate
        )
        
        # Queue for transcription
        success = self.transcription_queue.queue_audio_for_processing(
            audio_chunk, 
            callback=self._handle_transcription_result
        )
        
        if success:
            self.system_stats["audio_segments_processed"] += 1
        else:
            logger.warning(f"Failed to queue audio chunk {chunk_id}")
    
    def _handle_transcription_result(self, transcription_result):
        """Handle completed transcription result"""
        if not transcription_result or not transcription_result.text:
            return
        
        logger.info(f"Transcription: {transcription_result.text}")
        self.system_stats["transcriptions_completed"] += 1
        
        # Clean and process the transcribed text
        cleaned_text = self.text_processor.clean_transcription_text(transcription_result.text)
        
        # Update session with transcription
        if self.current_session_id:
            metadata = self.session_manager.get_session_metadata(self.current_session_id)
            if metadata:
                metadata["transcriptions"].append({
                    "text": cleaned_text,
                    "timestamp": transcription_result.timestamp,
                    "confidence": transcription_result.confidence,
                    "segments": transcription_result.segments
                })
                self.session_manager.save_session_metadata(self.current_session_id, metadata)
        
        # Process with structured form filling if enabled
        if self.config.enable_clinical_ai and self.form_processor:
            try:
                # Process transcript with structured form extraction
                form_results = self.form_processor.process_transcript(
                    transcript=cleaned_text,
                    session_id=self.current_session_id
                )
                
                # Log results
                for result in form_results:
                    if result.success:
                        logger.info(f"Successfully processed {result.form_name} form with {result.confidence:.2f} confidence")
                        filled_fields = sum(1 for v in result.extracted_data.values() if v)
                        logger.info(f"Filled {filled_fields}/{len(result.extracted_data)} fields")
                    else:
                        logger.warning(f"Failed to process {result.form_name} form: {result.error_message}")
                
                self.system_stats["forms_created"] += len([r for r in form_results if r.success])
                
            except Exception as e:
                logger.error(f"Error in form processing: {e}")
    
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Storage cleanup task
        cleanup_thread = threading.Thread(
            target=self._periodic_cleanup,
            daemon=True
        )
        cleanup_thread.start()
        
        # Health monitoring task
        health_thread = threading.Thread(
            target=self._periodic_health_check,
            daemon=True
        )
        health_thread.start()
    
    def _periodic_cleanup(self):
        """Periodic storage cleanup"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Perform cleanup based on storage policy
                # This would be implemented in StoragePolicy
                logger.debug("Performing periodic cleanup")
                time.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    def _periodic_health_check(self):
        """Periodic health monitoring"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Check component health
                if self.llm_client:
                    llm_stats = self.llm_client.get_stats()
                    if llm_stats["error_rate"] > 0.1:
                        logger.warning(f"High LLM error rate: {llm_stats['error_rate']:.2%}")
                
                if self.transcription_queue:
                    queue_stats = self.transcription_queue.get_queue_stats()
                    if queue_stats["queue_size"] > queue_stats["max_queue_size"] * 0.8:
                        logger.warning("Transcription queue near capacity")
                
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in health check: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics"""
        uptime = time.time() - self.system_stats["start_time"] if self.system_stats["start_time"] else 0
        
        status = {
            "is_running": self.is_running,
            "uptime_seconds": uptime,
            "current_session": self.current_session_id,
            "statistics": self.system_stats.copy(),
            "component_status": {}
        }
        
        # Add component-specific status
        if self.transcription_queue:
            status["component_status"]["transcription"] = self.transcription_queue.get_queue_stats()
        
        if self.llm_client:
            status["component_status"]["llm"] = self.llm_client.get_stats()
        
        if self.mobile_server:
            status["component_status"]["mobile"] = {
                "connected_clients": len(getattr(self.mobile_server, 'clients', {})),
                "server_running": self.mobile_server.is_running
            }
        
        if self.form_processor and self.current_session_id:
            try:
                forms_summary = self.form_processor.get_session_forms_summary(self.current_session_id)
                status["current_session_forms"] = forms_summary
            except Exception as e:
                logger.debug(f"Error getting forms summary: {e}")
        
        return status
    
    def _signal_handler(self, signum, _frame):
        """Handle system signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_system()
        sys.exit(0)

def main():
    """Main entry point for the clinical documentation system"""
    
    # Parse command line arguments (basic implementation)
    import argparse
    parser = argparse.ArgumentParser(description="Clinical Documentation System")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--mobile-only", action="store_true", help="Mobile interface only")
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create and start the system
        system = ClinicalDocumentationSystem(config_file=args.config)
        
        # Override config for mobile-only mode
        if args.mobile_only:
            system.config.enable_mobile_interface = True
        
        if not system.start_system():
            logger.error("Failed to start system")
            return 1
        
        # Create initial session
        session_id = system.create_new_session({
            "session_type": "interactive",
            "started_via": "command_line"
        })
        
        logger.info(f"System ready. Session ID: {session_id}")
        logger.info("Press Ctrl+C to stop the system")
        
        # Keep the main thread alive
        try:
            while system.is_running:
                time.sleep(1)
                
                # Print periodic status updates
                if int(time.time()) % 60 == 0:  # Every minute
                    status = system.get_system_status()
                    logger.info(f"System status: {status['statistics']}")
                    
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        
        return 0
        
    except Exception as e:
        logger.error(f"System error: {e}")
        return 1
    
    finally:
        if 'system' in locals():
            system.stop_system()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)