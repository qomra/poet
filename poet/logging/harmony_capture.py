from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Union, Callable
from datetime import datetime
import json
import uuid
from pathlib import Path
from contextlib import contextmanager
import functools
import asyncio

@dataclass
class CapturedCall:
    """Captures a single function/method call with inputs and outputs"""
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # What was called
    component_name: str = ""  # e.g., "ConstraintParser"
    method_name: str = ""      # e.g., "parse_constraints"
    call_type: str = ""        # "parse", "enrich", "generate", "validate", "refine"
    
    # Input/Output
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Any = None
    error: Optional[str] = None
    
    # LLM details if applicable
    llm_provider: Optional[str] = None
    model_name: Optional[str] = None
    prompt: Optional[str] = None
    response: Optional[str] = None
    tokens_used: Optional[int] = None
    
    # Execution details
    duration_ms: Optional[int] = None
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "timestamp": self.timestamp.isoformat(),
            "component_name": self.component_name,
            "method_name": self.method_name,
            "call_type": self.call_type,
            "inputs": self._serialize_value(self.inputs),
            "outputs": self._serialize_value(self.outputs),
            "error": self.error,
            "llm_provider": self.llm_provider,
            "model_name": self.model_name,
            "prompt": self.prompt,
            "response": self.response,
            "tokens_used": self.tokens_used,
            "duration_ms": self.duration_ms,
            "success": self.success
        }
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize values, handling custom objects"""
        if hasattr(value, 'to_dict'):
            return value.to_dict()
        elif hasattr(value, '__dict__'):
            return {k: self._serialize_value(v) for k, v in value.__dict__.items() 
                   if not k.startswith('_')}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        else:
            return value

@dataclass
class PipelineExecution:
    """Captures an entire pipeline execution"""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=datetime.now)
    
    # User input
    user_prompt: str = ""
    initial_constraints: Optional[Dict[str, Any]] = None
    
    # Captured calls in order
    calls: List[CapturedCall] = field(default_factory=list)
    
    # Final outputs
    final_poem: Optional[Any] = None
    quality_assessment: Optional[Any] = None
    
    # Execution metadata
    completed_at: Optional[datetime] = None
    total_duration_ms: Optional[int] = None
    total_llm_calls: int = 0
    total_tokens: int = 0
    
    def add_call(self, call: CapturedCall):
        """Add a captured call to the execution"""
        self.calls.append(call)
        if call.llm_provider:
            self.total_llm_calls += 1
        if call.tokens_used:
            self.total_tokens += call.tokens_used
    
    def complete(self, final_poem: Any = None, quality_assessment: Any = None):
        """Mark execution as complete"""
        self.completed_at = datetime.now()
        self.total_duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)
        self.final_poem = final_poem
        self.quality_assessment = quality_assessment
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "started_at": self.started_at.isoformat(),
            "user_prompt": self.user_prompt,
            "initial_constraints": self.initial_constraints,
            "calls": [call.to_dict() for call in self.calls],
            "final_poem": self._serialize_value(self.final_poem),
            "quality_assessment": self._serialize_value(self.quality_assessment),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_ms": self.total_duration_ms,
            "total_llm_calls": self.total_llm_calls,
            "total_tokens": self.total_tokens
        }
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize values, handling custom objects"""
        if hasattr(value, 'to_dict'):
            return value.to_dict()
        elif hasattr(value, '__dict__'):
            return {k: self._serialize_value(v) for k, v in value.__dict__.items() 
                   if not k.startswith('_')}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        else:
            return value

class ExecutionCapture:
    """
    Non-intrusive capture system that records pipeline execution
    without modifying the core logic
    """
    
    def __init__(self):
        self.current_execution: Optional[PipelineExecution] = None
        self.current_call: Optional[CapturedCall] = None
        self._capture_enabled = True
    
    def start_execution(self, user_prompt: str, initial_constraints: Dict[str, Any] = None):
        """Start capturing a new execution"""
        self.current_execution = PipelineExecution(
            user_prompt=user_prompt,
            initial_constraints=initial_constraints
        )
        return self.current_execution
    
    @contextmanager
    def capture_call(self, component_name: str, method_name: str, 
                    call_type: str, inputs: Dict[str, Any] = None):
        """Context manager to capture a single call"""
        if not self._capture_enabled or not self.current_execution:
            yield None
            return
        
        call = CapturedCall(
            component_name=component_name,
            method_name=method_name,
            call_type=call_type,
            inputs=inputs or {}
        )
        start_time = datetime.now()
        
        try:
            self.current_call = call
            yield call
            call.success = True
        except Exception as e:
            call.success = False
            call.error = str(e)
            raise
        finally:
            call.duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.current_execution.add_call(call)
            self.current_call = None
    
    def capture_llm_details(self, provider: str, model: str, 
                           prompt: str = None, response: str = None, 
                           tokens: int = None):
        """Add LLM details to current call"""
        if self.current_call:
            self.current_call.llm_provider = provider
            self.current_call.model_name = model
            self.current_call.prompt = prompt
            self.current_call.response = response
            self.current_call.tokens_used = tokens
    
    def capture_output(self, output: Any):
        """Capture the output of current call"""
        if self.current_call:
            self.current_call.outputs = output
    
    def complete_execution(self, final_poem: Any = None, quality_assessment: Any = None):
        """Complete the current execution"""
        if self.current_execution:
            self.current_execution.complete(final_poem, quality_assessment)
    
    def get_execution(self) -> Optional[PipelineExecution]:
        """Get the current execution"""
        return self.current_execution
    
    def export_execution(self, output_file: Path = None) -> str:
        """Export execution as JSON"""
        if not self.current_execution:
            return "{}"
        
        json_str = json.dumps(self.current_execution.to_dict(), 
                             indent=2, ensure_ascii=False)
        
        if output_file:
            output_file.write_text(json_str, encoding='utf-8')
        
        return json_str

# Global capture instance (singleton)
_capture = ExecutionCapture()

def get_capture() -> ExecutionCapture:
    """Get the global capture instance"""
    return _capture

# Decorator for capturing method calls
def capture_method(component_name: str, call_type: str):
    """
    Decorator to capture method calls non-intrusively
    
    Usage:
        @capture_method("ConstraintParser", "parse")
        def parse_constraints(self, prompt: str) -> Constraints:
            # ... method implementation
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Get the capture instance
                capture = get_capture()
                
                # Extract inputs (skip 'self' if it's a method)
                inputs = {}
                if args and hasattr(args[0], '__class__'):
                    # It's a method, skip self
                    if len(args) > 1:
                        inputs['args'] = args[1:]
                else:
                    inputs['args'] = args
                inputs['kwargs'] = kwargs
                
                # Capture the call
                with capture.capture_call(component_name, func.__name__, 
                                         call_type, inputs) as call:
                    result = await func(*args, **kwargs)
                    if call:
                        capture.capture_output(result)
                    return result
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Get the capture instance
                capture = get_capture()
                
                # Extract inputs (skip 'self' if it's a method)
                inputs = {}
                if args and hasattr(args[0], '__class__'):
                    # It's a method, skip self
                    if len(args) > 1:
                        inputs['args'] = args[1:]
                else:
                    inputs['args'] = args
                inputs['kwargs'] = kwargs
                
                # Capture the call
                with capture.capture_call(component_name, func.__name__, 
                                         call_type, inputs) as call:
                    result = func(*args, **kwargs)
                    if call:
                        capture.capture_output(result)
                    return result
            
            return sync_wrapper
    
    return decorator



