# form_processor.py - Simplified Form Processing with Ollama Structured Output
import json
import time
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass

from llm_service import LLMClient, LLMRequest
from storage_manager import SchemaStorage, FormStateStorage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FormResult:
    """Result of form processing"""
    form_name: str
    extracted_data: Dict[str, Any]
    confidence: float
    processing_time: float
    success: bool
    error_message: Optional[str] = None

class StructuredFormProcessor:
    """Simplified form processor using Ollama structured output"""
    
    def __init__(self, llm_client: LLMClient, schema_storage: SchemaStorage, 
                 form_storage: FormStateStorage):
        self.llm_client = llm_client
        self.schema_storage = schema_storage
        self.form_storage = form_storage
        
        # Processing settings
        self.temperature = 0.2  # Low temperature for consistent output
        self.max_tokens = 1500
        
        # Cache for JSON schemas
        self.json_schema_cache = {}
    
    def process_transcript(self, transcript: str, session_id: str, 
                         target_forms: Optional[List[str]] = None) -> List[FormResult]:
        """Process transcript and extract structured data for forms"""
        
        if not transcript or not transcript.strip():
            return []
        
        # Get available forms
        available_forms = self.schema_storage.load_form_schemas()
        if not available_forms:
            logger.warning("No form schemas available")
            return []
        
        # Determine which forms to process
        forms_to_process = target_forms if target_forms else list(available_forms.keys())
        results = []
        
        for form_name in forms_to_process:
            if form_name not in available_forms:
                logger.warning(f"Form schema not found: {form_name}")
                continue
            
            try:
                result = self._process_single_form(transcript, form_name, 
                                                 available_forms[form_name], session_id)
                results.append(result)
                
                # Update form storage if successful
                if result.success and result.extracted_data:
                    self._update_form_storage(session_id, form_name, result.extracted_data)
                    
            except Exception as e:
                logger.error(f"Error processing form {form_name}: {e}")
                results.append(FormResult(
                    form_name=form_name,
                    extracted_data={},
                    confidence=0.0,
                    processing_time=0.0,
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    def _process_single_form(self, transcript: str, form_name: str, 
                           form_schema: Dict[str, Any], session_id: str) -> FormResult:
        """Process transcript for a single form using structured output"""
        
        start_time = time.time()
        
        # Convert form schema to JSON schema format
        json_schema = self._convert_to_json_schema(form_name, form_schema)
        
        # Create structured prompt
        prompt = self._create_structured_prompt(transcript, form_name, form_schema)
        
        # Create LLM request with structured output
        request = LLMRequest(
            prompt=prompt,
            context={"session_id": session_id, "form_name": form_name},
            request_id=f"form_{form_name}_{int(time.time())}",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt=self._get_system_prompt()
        )
        
        # Send request with structured output format
        response = self._send_structured_request(request, json_schema)
        
        processing_time = time.time() - start_time
        
        # Parse and validate response
        if response and response.text:
            extracted_data, confidence = self._parse_response(response.text, form_schema)
            
            return FormResult(
                form_name=form_name,
                extracted_data=extracted_data,
                confidence=confidence,
                processing_time=processing_time,
                success=len(extracted_data) > 0
            )
        else:
            return FormResult(
                form_name=form_name,
                extracted_data={},
                confidence=0.0,
                processing_time=processing_time,
                success=False,
                error_message="No response from LLM"
            )
    
    def _convert_to_json_schema(self, form_name: str, form_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert form schema to JSON schema format for Ollama"""
        
        if form_name in self.json_schema_cache:
            return self.json_schema_cache[form_name]
        
        json_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for field_id, field_info in form_schema.items():
            field_type = field_info.get("ResponseType", "Text").lower()
            description = field_info.get("Description", "")
            
            # Map response types to JSON schema types
            if field_type == "boolean":
                json_schema["properties"][field_id] = {
                    "type": "boolean",
                    "description": description
                }
            elif field_type in ["number", "integer"]:
                json_schema["properties"][field_id] = {
                    "type": "number",
                    "description": description
                }
            else:  # Default to string for text, select, etc.
                json_schema["properties"][field_id] = {
                    "type": "string",
                    "description": description
                }
        
        # Cache the schema
        self.json_schema_cache[form_name] = json_schema
        return json_schema
    
    def _create_structured_prompt(self, transcript: str, form_name: str, 
                                form_schema: Dict[str, Any]) -> str:
        """Create a structured prompt for form extraction"""
        
        # Build field descriptions
        field_descriptions = []
        for field_id, field_info in form_schema.items():
            description = field_info.get("Description", "")
            response_type = field_info.get("ResponseType", "Text")
            field_descriptions.append(f"- {field_id}: {description} (Type: {response_type})")
        
        fields_text = "\n".join(field_descriptions)
        
        prompt = f"""Extract medical information from the following conversation transcript to fill out a {form_name} form.

CONVERSATION TRANSCRIPT:
{transcript}

FORM FIELDS TO EXTRACT:
{fields_text}

INSTRUCTIONS:
1. Extract information ONLY if explicitly mentioned in the conversation
2. For Boolean fields: Use true/false based on whether the condition is mentioned as present
3. For Text fields: Extract exact phrases or summarize relevant information
4. If information is not mentioned, use null for that field
5. Be precise and only include information that is clearly stated

Return the extracted information as a JSON object with the exact field IDs as keys."""
        
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for medical form extraction"""
        return """You are a medical documentation assistant that extracts structured information from clinical conversations. 

Key guidelines:
- Only extract information explicitly mentioned in the conversation
- Use precise medical terminology when present
- For Boolean fields, return true only if the condition is explicitly mentioned as present
- For missing information, use null values
- Maintain patient privacy and medical accuracy
- Return valid JSON matching the requested schema"""
    
    def _send_structured_request(self, request: LLMRequest, json_schema: Dict[str, Any]):
        """Send request to Ollama with structured output format"""
        
        # Use the structured request method we added to LLMClient
        return self.llm_client.send_structured_request(request, json_schema)
    
    def _parse_response(self, response_text: str, form_schema: Dict[str, Any]) -> tuple[Dict[str, Any], float]:
        """Parse LLM response and extract structured data"""
        
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                parsed_data = json.loads(json_text)
                
                # Validate against form schema
                validated_data = self._validate_extracted_data(parsed_data, form_schema)
                
                # Calculate confidence based on completeness and validity
                confidence = self._calculate_confidence(validated_data, form_schema)
                
                return validated_data, confidence
            else:
                logger.warning("No JSON found in response")
                return {}, 0.0
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {}, 0.0
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {}, 0.0
    
    def _validate_extracted_data(self, data: Dict[str, Any], 
                                form_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted data against form schema"""
        
        validated = {}
        
        for field_id, field_info in form_schema.items():
            if field_id in data:
                value = data[field_id]
                response_type = field_info.get("ResponseType", "Text").lower()
                
                # Type validation and conversion
                if response_type == "boolean":
                    validated[field_id] = bool(value) if value is not None else None
                elif response_type in ["number", "integer"]:
                    try:
                        validated[field_id] = float(value) if value is not None else None
                    except (ValueError, TypeError):
                        validated[field_id] = None
                else:  # Text or other types
                    validated[field_id] = str(value) if value is not None else None
            else:
                validated[field_id] = None
        
        return validated
    
    def _calculate_confidence(self, extracted_data: Dict[str, Any], 
                            form_schema: Dict[str, Any]) -> float:
        """Calculate confidence score for extraction"""
        
        if not form_schema:
            return 0.0
        
        total_fields = len(form_schema)
        filled_fields = sum(1 for value in extracted_data.values() 
                          if value is not None and str(value).strip())
        
        # Base confidence on completeness
        completeness_score = filled_fields / total_fields if total_fields > 0 else 0.0
        
        # Boost confidence if critical fields are filled
        critical_keywords = ["chief complaint", "history", "symptoms", "medication"]
        critical_filled = 0
        critical_total = 0
        
        for field_id, field_info in form_schema.items():
            description = field_info.get("Description", "").lower()
            if any(keyword in description for keyword in critical_keywords):
                critical_total += 1
                if extracted_data.get(field_id) is not None:
                    critical_filled += 1
        
        critical_score = critical_filled / critical_total if critical_total > 0 else 1.0
        
        # Combined confidence score
        confidence = (completeness_score * 0.6) + (critical_score * 0.4)
        return min(1.0, confidence)
    
    def _update_form_storage(self, session_id: str, form_name: str, 
                           extracted_data: Dict[str, Any]):
        """Update form storage with extracted data"""
        
        try:
            # Check if form already exists for this session
            existing_form_id = self._find_existing_form(session_id, form_name)
            
            if existing_form_id:
                # Update existing form
                self.form_storage.update_field_values(session_id, existing_form_id, extracted_data)
                logger.info(f"Updated existing form {existing_form_id} with new data")
            else:
                # Create new form instance
                form_id = self.form_storage.save_form_instance(session_id, form_name, extracted_data)
                logger.info(f"Created new form instance: {form_id}")
                
        except Exception as e:
            logger.error(f"Error updating form storage: {e}")
    
    def _find_existing_form(self, session_id: str, form_name: str) -> Optional[str]:
        """Find existing form instance of given type in session"""
        
        session_metadata = self.form_storage.session_manager.get_session_metadata(session_id)
        if not session_metadata:
            return None
        
        for form_id, form_info in session_metadata.get("forms", {}).items():
            if form_info.get("form_type") == form_name:
                return form_id
        
        return None
    
    def get_session_forms_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of all forms for a session"""
        
        session_metadata = self.form_storage.session_manager.get_session_metadata(session_id)
        if not session_metadata:
            return {"error": "Session not found"}
        
        forms_summary = {}
        for form_id, form_info in session_metadata.get("forms", {}).items():
            completion_status = self.form_storage.get_form_completion_status(session_id, form_id)
            forms_summary[form_id] = {
                "form_type": form_info.get("form_type"),
                "completion_percentage": completion_status.get("completion_percentage", 0),
                "last_updated": form_info.get("updated_at", form_info.get("created_at"))
            }
        
        return {
            "session_id": session_id,
            "total_forms": len(forms_summary),
            "forms": forms_summary
        }