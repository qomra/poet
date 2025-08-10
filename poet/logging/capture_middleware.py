from typing import Any, Callable, Dict
import functools
import asyncio
from poet.logging.harmony_capture import get_capture


class CaptureMiddleware:
    """
    Middleware that wraps any class to add capture functionality
    without modifying the original code at all
    """
    
    def __init__(self, wrapped_instance: Any, component_name: str):
        self.wrapped = wrapped_instance
        self.component_name = component_name
        self.capture = get_capture()
    
    def __getattr__(self, name: str) -> Any:
        """Intercept method calls and wrap them with capture"""
        attr = getattr(self.wrapped, name)
        
        # Only wrap callable methods (not properties)
        if not callable(attr) or name.startswith('_'):
            return attr
        
        # Determine if it's async
        if asyncio.iscoroutinefunction(attr):
            return self._wrap_async_method(attr, name)
        else:
            return self._wrap_sync_method(attr, name)
    
    def _wrap_sync_method(self, method: Callable, method_name: str) -> Callable:
        """Wrap synchronous method with capture"""
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            with self.capture.capture_call(
                component_name=self.component_name,
                method_name=method_name,
                call_type=self._infer_call_type(method_name),
                inputs=self._extract_inputs(args, kwargs)
            ) as call:
                result = method(*args, **kwargs)
                if call:
                    self.capture.capture_output(result)
                return result
        return wrapper
    
    def _wrap_async_method(self, method: Callable, method_name: str) -> Callable:
        """Wrap asynchronous method with capture"""
        @functools.wraps(method)
        async def wrapper(*args, **kwargs):
            with self.capture.capture_call(
                component_name=self.component_name,
                method_name=method_name,
                call_type=self._infer_call_type(method_name),
                inputs=self._extract_inputs(args, kwargs)
            ) as call:
                result = await method(*args, **kwargs)
                if call:
                    self.capture.capture_output(result)
                return result
        return wrapper
    
    def _infer_call_type(self, method_name: str) -> str:
        """Infer the call type from method name"""
        if 'parse' in method_name.lower():
            return 'parse'
        elif 'refine' in method_name.lower():
            return 'refine'
        elif 'evaluate' in method_name.lower():
            return 'evaluate'
        elif 'generate' in method_name.lower():
            return 'generate'
        else:
            return 'process'
    
    def _extract_inputs(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract meaningful inputs for capture"""
        inputs = {}
        
        # Add kwargs
        inputs.update(kwargs)
        
        # Add positional args with generic names
        for i, arg in enumerate(args):
            # Try to extract meaningful data
            if hasattr(arg, 'to_dict'):
                inputs[f'arg_{i}'] = arg.to_dict()
            elif hasattr(arg, 'verses'):
                inputs[f'poem_{i}'] = arg.verses
            elif isinstance(arg, (str, int, float, bool, list, dict)):
                inputs[f'arg_{i}'] = arg
            else:
                inputs[f'arg_{i}'] = str(type(arg).__name__)
        
        return inputs


def capture_component(component: Any, name: str) -> Any:
    """
    Factory function to wrap any component with capture
    
    Usage:
        refiner_chain = RefinerChain(refiners, llm)
        captured_chain = capture_component(refiner_chain, "RefinerChain")
        # Now use captured_chain exactly like refiner_chain
    """
    return CaptureMiddleware(component, name)