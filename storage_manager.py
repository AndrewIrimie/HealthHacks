# storage_manager.py - Zero-loss Storage Management Module
import os
import json
import time
import threading
import hashlib
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SessionManager:
    """Track audio sessions and form states"""
    
    def __init__(self, base_dir: str = "audio_sessions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.active_sessions = {}
        self.session_lock = threading.Lock()
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session and return session ID"""
        if not session_id:
            session_id = f"session_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        
        session_dir = self.base_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        session_metadata = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "audio_segments": [],
            "forms": {},
            "transcriptions": []
        }
        
        with self.session_lock:
            self.active_sessions[session_id] = {
                "metadata": session_metadata,
                "directory": session_dir,
                "last_activity": time.time()
            }
        
        self.save_session_metadata(session_id, session_metadata)
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def save_session_metadata(self, session_id: str, metadata: Dict[str, Any]):
        """Save session metadata to disk"""
        if session_id not in self.active_sessions:
            logger.error(f"Session {session_id} not found")
            return
        
        metadata_file = self.active_sessions[session_id]["directory"] / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session metadata"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]["metadata"]
        return None
    
    def update_session_activity(self, session_id: str):
        """Update last activity timestamp"""
        with self.session_lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["last_activity"] = time.time()
    
    def close_session(self, session_id: str):
        """Close a session"""
        with self.session_lock:
            if session_id in self.active_sessions:
                metadata = self.active_sessions[session_id]["metadata"]
                metadata["status"] = "closed"
                metadata["closed_at"] = datetime.now().isoformat()
                self.save_session_metadata(session_id, metadata)
                del self.active_sessions[session_id]
                logger.info(f"Closed session: {session_id}")

class StoragePolicy:
    """Configurable retention policies"""
    
    def __init__(self, 
                 max_session_age_hours: int = 24,
                 max_storage_gb: float = 10.0,
                 cleanup_interval_hours: int = 6):
        self.max_session_age_hours = max_session_age_hours
        self.max_storage_gb = max_storage_gb
        self.cleanup_interval_hours = cleanup_interval_hours
        self.last_cleanup = time.time()
    
    def should_cleanup_session(self, session_created_at: str) -> bool:
        """Check if session should be cleaned up based on age"""
        created_time = datetime.fromisoformat(session_created_at)
        age = datetime.now() - created_time
        return age > timedelta(hours=self.max_session_age_hours)
    
    def get_storage_usage_gb(self, base_dir: str) -> float:
        """Calculate current storage usage in GB"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(base_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024**3)  # Convert to GB
    
    def needs_cleanup(self, base_dir: str) -> bool:
        """Check if cleanup is needed"""
        current_time = time.time()
        time_since_cleanup = (current_time - self.last_cleanup) / 3600  # hours
        
        storage_exceeded = self.get_storage_usage_gb(base_dir) > self.max_storage_gb
        time_exceeded = time_since_cleanup > self.cleanup_interval_hours
        
        return storage_exceeded or time_exceeded

class FormStateStorage:
    """Store form instances and field states"""
    
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
    
    def save_form_instance(self, session_id: str, form_type: str, form_data: Dict[str, Any]):
        """Save a form instance to session"""
        metadata = self.session_manager.get_session_metadata(session_id)
        if not metadata:
            logger.error(f"Session {session_id} not found")
            return
        
        if "forms" not in metadata:
            metadata["forms"] = {}
        
        form_id = f"{form_type}_{int(time.time())}"
        metadata["forms"][form_id] = {
            "form_type": form_type,
            "created_at": datetime.now().isoformat(),
            "data": form_data,
            "status": "active"
        }
        
        self.session_manager.save_session_metadata(session_id, metadata)
        self.session_manager.update_session_activity(session_id)
        return form_id
    
    def update_field_values(self, session_id: str, form_id: str, field_updates: Dict[str, Any]):
        """Update specific field values in a form"""
        metadata = self.session_manager.get_session_metadata(session_id)
        if not metadata or form_id not in metadata.get("forms", {}):
            logger.error(f"Form {form_id} not found in session {session_id}")
            return
        
        form_data = metadata["forms"][form_id]["data"]
        form_data.update(field_updates)
        metadata["forms"][form_id]["updated_at"] = datetime.now().isoformat()
        
        self.session_manager.save_session_metadata(session_id, metadata)
        self.session_manager.update_session_activity(session_id)
    
    def get_form_completion_status(self, session_id: str, form_id: str) -> Dict[str, Any]:
        """Get completion status of a form"""
        metadata = self.session_manager.get_session_metadata(session_id)
        if not metadata or form_id not in metadata.get("forms", {}):
            return {"error": "Form not found"}
        
        form = metadata["forms"][form_id]
        form_data = form["data"]
        
        # Calculate completion percentage
        total_fields = len(form_data)
        filled_fields = sum(1 for value in form_data.values() if value and str(value).strip())
        completion_percentage = (filled_fields / total_fields * 100) if total_fields > 0 else 0
        
        return {
            "form_id": form_id,
            "form_type": form["form_type"],
            "completion_percentage": completion_percentage,
            "filled_fields": filled_fields,
            "total_fields": total_fields,
            "status": form["status"]
        }

class SchemaStorage:
    """Manage JSON form schemas"""
    
    def __init__(self, schemas_dir: str = "json_forms"):
        self.schemas_dir = Path(schemas_dir)
        self.schemas_dir.mkdir(exist_ok=True)
        self.cached_schemas = {}
        self.schema_lock = threading.Lock()
    
    def load_form_schemas(self) -> Dict[str, Any]:
        """Load all available form schemas"""
        with self.schema_lock:
            if not self.cached_schemas:
                self._refresh_schema_cache()
            return self.cached_schemas.copy()
    
    def _refresh_schema_cache(self):
        """Refresh the schema cache from disk"""
        self.cached_schemas = {}
        for schema_file in self.schemas_dir.glob("*.json"):
            try:
                with open(schema_file, 'r') as f:
                    schema_data = json.load(f)
                    schema_name = schema_file.stem
                    self.cached_schemas[schema_name] = schema_data
                    logger.info(f"Loaded schema: {schema_name}")
            except Exception as e:
                logger.error(f"Error loading schema {schema_file}: {e}")
    
    def get_schema_for_ai(self, form_type: str) -> Optional[Dict[str, Any]]:
        """Get schema formatted for AI processing"""
        schemas = self.load_form_schemas()
        if form_type not in schemas:
            return None
        
        schema = schemas[form_type]
        # Format for AI with field descriptions and types
        ai_schema = {
            "form_type": form_type,
            "description": schema.get("description", ""),
            "fields": {}
        }
        
        for field_name, field_info in schema.get("fields", {}).items():
            ai_schema["fields"][field_name] = {
                "type": field_info.get("type", "text"),
                "description": field_info.get("description", ""),
                "required": field_info.get("required", False),
                "options": field_info.get("options", [])
            }
        
        return ai_schema
    
    def validate_schema_format(self, schema: Dict[str, Any]) -> bool:
        """Validate schema format"""
        required_keys = ["form_type", "fields"]
        if not all(key in schema for key in required_keys):
            return False
        
        # Validate field structure
        for field_name, field_info in schema["fields"].items():
            if not isinstance(field_info, dict):
                return False
            if "type" not in field_info:
                return False
        
        return True

class AudioArchiver:
    """Persistent audio storage with zero-loss guarantee"""
    
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.archival_lock = threading.Lock()
    
    def save_audio_segment(self, session_id: str, audio_data: np.ndarray, 
                          segment_id: str, timestamp: float, sample_rate: int = 16000) -> str:
        """Save audio segment with zero-loss guarantee"""
        session_metadata = self.session_manager.get_session_metadata(session_id)
        if not session_metadata:
            logger.error(f"Session {session_id} not found")
            return ""
        
        session_dir = self.session_manager.active_sessions[session_id]["directory"]
        audio_dir = session_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        # Save as both WAV and NPY for different use cases
        audio_file_wav = audio_dir / f"{segment_id}.wav"
        audio_file_npy = audio_dir / f"{segment_id}.npy"
        
        try:
            with self.archival_lock:
                # Save as numpy array (lossless)
                np.save(audio_file_npy, audio_data)
                
                # Also save as WAV for compatibility
                import soundfile as sf
                sf.write(audio_file_wav, audio_data, sample_rate)
                
                # Update session metadata
                segment_metadata = {
                    "segment_id": segment_id,
                    "timestamp": timestamp,
                    "file_path_npy": str(audio_file_npy),
                    "file_path_wav": str(audio_file_wav),
                    "sample_rate": sample_rate,
                    "duration": len(audio_data) / sample_rate,
                    "saved_at": datetime.now().isoformat()
                }
                
                session_metadata["audio_segments"].append(segment_metadata)
                self.session_manager.save_session_metadata(session_id, session_metadata)
                self.session_manager.update_session_activity(session_id)
                
                logger.info(f"Saved audio segment {segment_id} for session {session_id}")
                return str(audio_file_npy)
                
        except Exception as e:
            logger.error(f"Error saving audio segment {segment_id}: {e}")
            return ""
    
    def load_audio_segment(self, session_id: str, segment_id: str) -> Optional[np.ndarray]:
        """Load audio segment from storage"""
        session_metadata = self.session_manager.get_session_metadata(session_id)
        if not session_metadata:
            return None
        
        # Find segment in metadata
        segment_metadata = None
        for segment in session_metadata.get("audio_segments", []):
            if segment["segment_id"] == segment_id:
                segment_metadata = segment
                break
        
        if not segment_metadata:
            logger.error(f"Audio segment {segment_id} not found in session {session_id}")
            return None
        
        try:
            audio_file_npy = Path(segment_metadata["file_path_npy"])
            if audio_file_npy.exists():
                return np.load(audio_file_npy)
            else:
                logger.error(f"Audio file not found: {audio_file_npy}")
                return None
        except Exception as e:
            logger.error(f"Error loading audio segment {segment_id}: {e}")
            return None
    
    def compress_for_long_term_storage(self, session_id: str):
        """Compress older audio files for long-term storage"""
        # Implementation for compression (could use FLAC or other lossless compression)
        session_metadata = self.session_manager.get_session_metadata(session_id)
        if not session_metadata:
            return
        
        logger.info(f"Compressing audio for long-term storage: session {session_id}")
        # TODO: Implement compression logic
    
    def ensure_zero_loss_guarantee(self, session_id: str) -> bool:
        """Verify all audio segments are properly stored"""
        session_metadata = self.session_manager.get_session_metadata(session_id)
        if not session_metadata:
            return False
        
        all_segments_valid = True
        for segment in session_metadata.get("audio_segments", []):
            audio_file = Path(segment["file_path_npy"])
            if not audio_file.exists():
                logger.error(f"Missing audio file: {audio_file}")
                all_segments_valid = False
            else:
                # Verify file integrity
                try:
                    audio_data = np.load(audio_file)
                    if len(audio_data) == 0:
                        logger.error(f"Empty audio file: {audio_file}")
                        all_segments_valid = False
                except Exception as e:
                    logger.error(f"Corrupted audio file {audio_file}: {e}")
                    all_segments_valid = False
        
        return all_segments_valid