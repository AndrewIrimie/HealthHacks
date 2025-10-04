# llm_service.py - LLM Integration Service Module
import threading
import queue
import time
import json
import requests
from typing import Optional, Dict, Any, List, Callable
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

@dataclass
class LLMRequest:
    """LLM request container"""
    prompt: str
    context: Dict[str, Any]
    request_id: str
    temperature: float = 0.7
    max_tokens: int = 2000
    system_prompt: Optional[str] = None
    timestamp: float = time.time()

@dataclass
class LLMResponse:
    """LLM response container"""
    text: str
    request_id: str
    provider: str
    processing_time: float
    confidence: float
    metadata: Dict[str, Any]
    timestamp: float

class LLMClient:
    """LLM interface for content formatting and clinical analysis"""
    
    def __init__(self, provider: LLMProvider = LLMProvider.OLLAMA, 
                 base_url: str = "http://localhost:11434",
                 model_name: str = "llama2"):
        self.provider = provider
        self.base_url = base_url
        self.model_name = model_name
        self.session = requests.Session()
        self.is_healthy = False
        self.last_health_check = 0
        self.health_check_interval = 30  # seconds
        
        # Connection settings
        self.timeout = 30
        self.max_retries = 3
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.avg_response_time = 0.0
        self.stats_lock = threading.Lock()
        
        # Initialize health check
        self._check_health()
    
    def _check_health(self) -> bool:
        """Check if LLM service is healthy"""
        current_time = time.time()
        if current_time - self.last_health_check < self.health_check_interval:
            return self.is_healthy
        
        try:
            if self.provider == LLMProvider.OLLAMA:
                response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
                self.is_healthy = response.status_code == 200
            else:
                # Health check for other providers
                self.is_healthy = True  # Placeholder
            
            self.last_health_check = current_time
            if self.is_healthy:
                logger.info(f"LLM service health check passed: {self.provider.value}")
            else:
                logger.warning(f"LLM service health check failed: {self.provider.value}")
                
        except Exception as e:
            logger.error(f"LLM health check error: {e}")
            self.is_healthy = False
        
        return self.is_healthy
    
    def send_formatting_prompt(self, llm_request: LLMRequest) -> LLMResponse:
        """Send formatting prompt to LLM with retries"""
        if not self._check_health():
            return self._create_error_response(llm_request, "LLM service unhealthy")
        
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if self.provider == LLMProvider.OLLAMA:
                    response = self._send_ollama_request(llm_request)
                elif self.provider == LLMProvider.OPENAI:
                    response = self._send_openai_request(llm_request)
                else:
                    response = self._create_error_response(llm_request, "Unsupported provider")
                
                # Update statistics
                processing_time = time.time() - start_time
                with self.stats_lock:
                    self.request_count += 1
                    if self.avg_response_time == 0:
                        self.avg_response_time = processing_time
                    else:
                        self.avg_response_time = (self.avg_response_time * 0.9 + processing_time * 0.1)
                
                return response
                
            except Exception as e:
                last_error = e
                logger.warning(f"LLM request attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # All retries failed
        with self.stats_lock:
            self.error_count += 1
        
        return self._create_error_response(llm_request, f"All retries failed: {last_error}")
    
    def _send_ollama_request(self, llm_request: LLMRequest) -> LLMResponse:
        """Send request to Ollama service"""
        payload = {
            "model": self.model_name,
            "prompt": llm_request.prompt,
            "system": llm_request.system_prompt or "",
            "options": {
                "temperature": llm_request.temperature,
                "num_predict": llm_request.max_tokens
            },
            "stream": False
        }
        
        # Add format parameter if specified in context
        if "json_schema" in llm_request.context:
            payload["format"] = llm_request.context["json_schema"]
        
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama request failed: {response.status_code} - {response.text}")
        
        result = response.json()
        processing_time = time.time() - llm_request.timestamp
        
        return LLMResponse(
            text=result.get("response", ""),
            request_id=llm_request.request_id,
            provider=self.provider.value,
            processing_time=processing_time,
            confidence=0.8,  # Default confidence for Ollama
            metadata={
                "model": self.model_name,
                "eval_count": result.get("eval_count", 0),
                "eval_duration": result.get("eval_duration", 0)
            },
            timestamp=time.time()
        )
    
    def _send_openai_request(self, llm_request: LLMRequest) -> LLMResponse:
        """Send request to OpenAI API (placeholder)"""
        # Placeholder implementation for OpenAI integration
        processing_time = time.time() - llm_request.timestamp
        
        return LLMResponse(
            text="[OpenAI integration not yet implemented]",
            request_id=llm_request.request_id,
            provider=self.provider.value,
            processing_time=processing_time,
            confidence=0.0,
            metadata={"status": "not_implemented"},
            timestamp=time.time()
        )
    
    def _create_error_response(self, llm_request: LLMRequest, error_message: str) -> LLMResponse:
        """Create error response"""
        processing_time = time.time() - llm_request.timestamp
        
        return LLMResponse(
            text=f"[Error: {error_message}]",
            request_id=llm_request.request_id,
            provider=self.provider.value,
            processing_time=processing_time,
            confidence=0.0,
            metadata={"error": error_message},
            timestamp=time.time()
        )
    
    def send_structured_request(self, llm_request: LLMRequest, json_schema: Dict[str, Any]) -> LLMResponse:
        """Send request with structured JSON output format"""
        # Add JSON schema to request context
        llm_request.context["json_schema"] = json_schema
        return self.send_formatting_prompt(llm_request)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        with self.stats_lock:
            return {
                "provider": self.provider.value,
                "model": self.model_name,
                "is_healthy": self.is_healthy,
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(1, self.request_count),
                "avg_response_time": self.avg_response_time
            }

# Removed complex formatting classes - using simplified structured output approach