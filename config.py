# config.py - Centralized Configuration Module
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AudioFormat(Enum):
    """Supported audio formats"""
    WAV = "wav"
    FLAC = "flac"
    MP3 = "mp3"
    OGG = "ogg"

class LLMProvider(Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    # Audio input settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 3.0  # seconds
    buffer_duration: float = 30.0  # seconds
    
    # Voice Activity Detection
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive
    vad_frame_duration: int = 30  # milliseconds (10, 20, or 30)
    silence_timeout: float = 2.0  # seconds
    min_speech_duration: float = 0.5  # seconds
    
    # Audio quality
    noise_gate_threshold: float = 0.01
    auto_gain_control: bool = True
    noise_suppression: bool = True
    
    # Storage settings
    save_format: AudioFormat = AudioFormat.WAV
    compression_level: int = 6  # For FLAC compression (0-8)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        config_dict = asdict(self)
        # Convert enums to their values
        config_dict['save_format'] = self.save_format.value
        return config_dict

@dataclass
class MedicalConfig:
    """Medical documentation configuration"""
    # Compliance settings
    hipaa_compliant: bool = True
    audit_logging: bool = True
    data_encryption: bool = True
    
    # Clinical documentation
    default_documentation_format: str = "SOAP"  # SOAP, DAP, PIE
    require_physician_review: bool = True
    auto_save_interval: int = 30  # seconds
    
    # Medical terminology
    medical_spell_check: bool = True
    drug_interaction_warnings: bool = True
    clinical_decision_support: bool = False
    
    # Patient privacy
    patient_id_anonymization: bool = True
    phi_detection: bool = True  # Protected Health Information
    automatic_redaction: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class StorageConfig:
    """Storage and retention configuration"""
    # Base directories
    base_dir: str = "audio_sessions"
    forms_dir: str = "json_forms"
    temp_dir: str = "temp"
    backup_dir: str = "backups"
    
    # Retention policies
    max_session_age_hours: int = 24
    max_storage_gb: float = 10.0
    cleanup_interval_hours: int = 6
    backup_interval_hours: int = 24
    
    # File management
    auto_cleanup: bool = True
    compress_old_files: bool = True
    verify_file_integrity: bool = True
    
    # Zero-loss guarantees
    redundant_storage: bool = True
    checksum_verification: bool = True
    transaction_logging: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class LLMConfig:
    """LLM service configuration"""
    # Provider settings
    provider: LLMProvider = LLMProvider.OLLAMA
    base_url: str = "http://localhost:11434"
    model_name: str = "llama2"
    api_key: Optional[str] = None
    
    # Request settings
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30
    max_retries: int = 3
    
    # Performance settings
    concurrent_requests: int = 3
    request_queue_size: int = 100
    health_check_interval: int = 30
    
    # Medical optimization
    medical_prompt_templates: bool = True
    clinical_terminology_boost: bool = True
    medical_abbreviation_expansion: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        config_dict = asdict(self)
        # Convert enums to their values
        config_dict['provider'] = self.provider.value
        return config_dict

@dataclass
class TranscriptionConfig:
    """Speech-to-text configuration"""
    # Whisper settings
    model_size: str = "base"  # tiny, base, small, medium, large
    device: str = "cpu"  # cpu, cuda
    language: str = "en"
    
    # Processing settings
    queue_size: int = 100
    concurrent_processing: int = 2
    chunk_overlap: float = 0.5  # seconds
    
    # Medical optimization
    medical_vocabulary_boost: bool = True
    drug_name_recognition: bool = True
    medical_abbreviation_detection: bool = True
    
    # Quality settings
    confidence_threshold: float = 0.6
    retry_low_confidence: bool = True
    post_processing: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class MobileConfig:
    """Mobile app communication configuration"""
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080
    max_connections: int = 10
    
    # Protocol settings
    protocol_version: str = "1.0"
    heartbeat_interval: int = 30  # seconds
    connection_timeout: int = 300  # seconds
    
    # Audio streaming
    buffer_size: int = 8192
    compression: bool = True
    real_time_transcription: bool = True
    
    # Security
    require_authentication: bool = False
    ssl_enabled: bool = False
    certificate_path: Optional[str] = None
    key_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ClinicalWorkflowConfig:
    """Clinical workflow configuration"""
    # Workflow settings
    auto_form_selection: bool = True
    multi_form_processing: bool = True
    smart_field_population: bool = True
    
    # Clinical intelligence
    symptom_analysis: bool = True
    drug_interaction_checking: bool = False
    clinical_decision_support: bool = False
    
    # Integration settings
    ehr_integration: bool = False
    hl7_fhir_support: bool = False
    icd10_coding: bool = False
    
    # Validation
    clinical_validation: bool = True
    require_signatures: bool = False
    audit_trail: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SystemConfig:
    """Main system configuration"""
    # Component configurations
    audio: AudioConfig = None
    medical: MedicalConfig = None
    storage: StorageConfig = None
    llm: LLMConfig = None
    transcription: TranscriptionConfig = None
    mobile: MobileConfig = None
    clinical_workflow: ClinicalWorkflowConfig = None
    
    def __post_init__(self):
        """Initialize default configurations if None"""
        if self.audio is None:
            self.audio = AudioConfig()
        if self.medical is None:
            self.medical = MedicalConfig()
        if self.storage is None:
            self.storage = StorageConfig()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.transcription is None:
            self.transcription = TranscriptionConfig()
        if self.mobile is None:
            self.mobile = MobileConfig()
        if self.clinical_workflow is None:
            self.clinical_workflow = ClinicalWorkflowConfig()
    
    # System settings
    debug_mode: bool = False
    log_level: LogLevel = LogLevel.INFO
    performance_monitoring: bool = True
    
    # Feature flags
    enable_mobile_interface: bool = True
    enable_clinical_ai: bool = True
    enable_multi_form_processing: bool = True
    enable_real_time_suggestions: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire config to dictionary"""
        config_dict = {
            'audio': self.audio.to_dict(),
            'medical': self.medical.to_dict(),
            'storage': self.storage.to_dict(),
            'llm': self.llm.to_dict(),
            'transcription': self.transcription.to_dict(),
            'mobile': self.mobile.to_dict(),
            'clinical_workflow': self.clinical_workflow.to_dict(),
            'debug_mode': self.debug_mode,
            'log_level': self.log_level.value,
            'performance_monitoring': self.performance_monitoring,
            'enable_mobile_interface': self.enable_mobile_interface,
            'enable_clinical_ai': self.enable_clinical_ai,
            'enable_multi_form_processing': self.enable_multi_form_processing,
            'enable_real_time_suggestions': self.enable_real_time_suggestions
        }
        return config_dict

class ConfigManager:
    """Configuration management with environment variable support"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "system_config.json"
        self.config = SystemConfig()
        self._load_environment_overrides()
    
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables"""
        # Audio settings
        if os.getenv('AUDIO_SAMPLE_RATE'):
            self.config.audio.sample_rate = int(os.getenv('AUDIO_SAMPLE_RATE'))
        
        if os.getenv('AUDIO_CHUNK_DURATION'):
            self.config.audio.chunk_duration = float(os.getenv('AUDIO_CHUNK_DURATION'))
        
        # LLM settings
        if os.getenv('LLM_PROVIDER'):
            provider_name = os.getenv('LLM_PROVIDER').upper()
            if hasattr(LLMProvider, provider_name):
                self.config.llm.provider = LLMProvider[provider_name]
        
        if os.getenv('LLM_BASE_URL'):
            self.config.llm.base_url = os.getenv('LLM_BASE_URL')
        
        if os.getenv('LLM_MODEL_NAME'):
            self.config.llm.model_name = os.getenv('LLM_MODEL_NAME')
        
        if os.getenv('LLM_API_KEY'):
            self.config.llm.api_key = os.getenv('LLM_API_KEY')
        
        # Mobile settings
        if os.getenv('MOBILE_HOST'):
            self.config.mobile.host = os.getenv('MOBILE_HOST')
        
        if os.getenv('MOBILE_PORT'):
            self.config.mobile.port = int(os.getenv('MOBILE_PORT'))
        
        # Storage settings
        if os.getenv('STORAGE_BASE_DIR'):
            self.config.storage.base_dir = os.getenv('STORAGE_BASE_DIR')
        
        if os.getenv('STORAGE_MAX_GB'):
            self.config.storage.max_storage_gb = float(os.getenv('STORAGE_MAX_GB'))
        
        # System settings
        if os.getenv('DEBUG_MODE'):
            self.config.debug_mode = os.getenv('DEBUG_MODE').lower() in ('true', '1', 'yes')
        
        if os.getenv('LOG_LEVEL'):
            level_name = os.getenv('LOG_LEVEL').upper()
            if hasattr(LogLevel, level_name):
                self.config.log_level = LogLevel[level_name]
    
    def load_from_file(self, config_file: Optional[str] = None) -> bool:
        """Load configuration from JSON file"""
        file_path = config_file or self.config_file
        
        if not os.path.exists(file_path):
            return False
        
        try:
            import json
            with open(file_path, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration with loaded data
            self._update_config_from_dict(config_data)
            return True
            
        except Exception as e:
            print(f"Error loading config file {file_path}: {e}")
            return False
    
    def save_to_file(self, config_file: Optional[str] = None) -> bool:
        """Save configuration to JSON file"""
        file_path = config_file or self.config_file
        
        try:
            import json
            config_dict = self.config.to_dict()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving config file {file_path}: {e}")
            return False
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary"""
        # This would be a more complex implementation to safely update
        # nested configuration objects from the loaded data
        # For now, we'll just handle a few key settings
        
        if 'audio' in config_data:
            audio_data = config_data['audio']
            if 'sample_rate' in audio_data:
                self.config.audio.sample_rate = audio_data['sample_rate']
            if 'chunk_duration' in audio_data:
                self.config.audio.chunk_duration = audio_data['chunk_duration']
        
        if 'llm' in config_data:
            llm_data = config_data['llm']
            if 'provider' in llm_data:
                provider_name = llm_data['provider'].upper()
                if hasattr(LLMProvider, provider_name):
                    self.config.llm.provider = LLMProvider[provider_name]
            if 'base_url' in llm_data:
                self.config.llm.base_url = llm_data['base_url']
            if 'model_name' in llm_data:
                self.config.llm.model_name = llm_data['model_name']
    
    def get_config(self) -> SystemConfig:
        """Get current configuration"""
        return self.config
    
    def validate_config(self) -> list[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate audio settings
        if self.config.audio.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            issues.append("Invalid audio sample rate")
        
        if self.config.audio.chunk_duration <= 0:
            issues.append("Audio chunk duration must be positive")
        
        # Validate storage settings
        if self.config.storage.max_storage_gb <= 0:
            issues.append("Max storage must be positive")
        
        # Validate LLM settings
        if not self.config.llm.base_url:
            issues.append("LLM base URL is required")
        
        if not self.config.llm.model_name:
            issues.append("LLM model name is required")
        
        # Validate mobile settings
        if not (1024 <= self.config.mobile.port <= 65535):
            issues.append("Mobile port must be between 1024 and 65535")
        
        return issues
    
    def create_directories(self):
        """Create necessary directories based on configuration"""
        directories = [
            self.config.storage.base_dir,
            self.config.storage.forms_dir,
            self.config.storage.temp_dir,
            self.config.storage.backup_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

# Global configuration instance
_config_manager = None

def get_config() -> SystemConfig:
    """Get global configuration instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        _config_manager.load_from_file()
    return _config_manager.get_config()

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        _config_manager.load_from_file()
    return _config_manager

def init_config(config_file: Optional[str] = None) -> ConfigManager:
    """Initialize configuration with optional config file"""
    global _config_manager
    _config_manager = ConfigManager(config_file)
    _config_manager.load_from_file()
    _config_manager.create_directories()
    return _config_manager