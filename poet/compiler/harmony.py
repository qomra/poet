from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import re
from poet.llm.base_llm import BaseLLM
from poet.prompts.prompt_manager import PromptManager
from poet.logging.harmony_capture import PipelineExecution

class HarmonyCompiler:
    """
    Takes captured pipeline execution and generates Harmony-formatted
    reasoning that reconstructs the entire process as a coherent narrative
    """
    
    def __init__(self, llm: BaseLLM, prompt_manager: PromptManager = None):
        self.llm = llm
        self.prompt_manager = prompt_manager or PromptManager()
    
    def generate_structured_harmony(self, execution: PipelineExecution) -> dict:
        """
        Generate structured Harmony data compatible with openai_harmony library
        
        Returns:
            Dictionary with conversation structure that can be used with openai_harmony
        """
        # Get the prompt template from prompt manager
        template = self.prompt_manager.get_template("harmony_structured")
        
        # Format the prompt with execution data
        prompt = template.format(
            user_prompt=execution.user_prompt,
            initial_constraints=json.dumps(self._serialize_output(execution.initial_constraints), ensure_ascii=False, indent=2),
            execution_steps=self._format_execution_steps(execution),
            final_poem=json.dumps(self._serialize_output(execution.final_poem), ensure_ascii=False, indent=2),
            quality_assessment=json.dumps(self._serialize_output(execution.quality_assessment), ensure_ascii=False, indent=2),
            conversation_start_date=execution.started_at[:10] if isinstance(execution.started_at, str) else execution.started_at.strftime('%Y-%m-%d'),
            cursor="1",
            toolname="browser",
            line_start="1",
            line_end="10",
            name="browser_tool",
            output="search_results",
            id="1",
            long_chain_of_thought="تفكير متسلسل طويل حول توليد الشعر العربي"
        )
        
        # Generate structured response
        response = self.llm.generate(prompt)
        
        # Debug: Print the raw response for troubleshooting
        print(f"🔍 Raw LLM response (first 500 chars): {response[:500]}...")
        
        # Parse the Harmony format response and convert to structured data
        try:
            structured_data = self._parse_harmony_response(response)
            print(f"✅ Successfully parsed Harmony response with {len(structured_data.get('messages', []))} messages")
            return structured_data
        except Exception as e:
            # Fallback to raw response if parsing fails
            print(f"❌ Failed to parse Harmony response: {str(e)}")
            print(f"🔍 Raw response: {response}")
            raise ValueError(f"Failed to parse Harmony response: {str(e)}")
    
    def _parse_harmony_response(self, response: str) -> dict:
        """
        Parse Harmony format response and convert to structured data
        
        Args:
            response: Raw Harmony format response from LLM
            
        Returns:
            Dictionary with structured conversation data
        """
        # Initialize structured data
        structured_data = {
            "system_message": {
                "model_identity": "أنت وكيل توليد الشعر العربي",
                "reasoning_effort": "high",
                "conversation_start_date": "2025-01-01",
                "knowledge_cutoff": "2024-06",
                "required_channels": ["analysis", "final"],
                "tools": [
                    {
                        "type": "browser",
                        "description": "أداة للتصفح والبحث على الويب"
                    }
                ]
            },
            "developer_message": {
                "instructions": "أنت نظام متخصص في توليد الشعر العربي"
            },
            "messages": []
        }
        
        # Parse the Harmony format response
        try:
            # Split by message boundaries
            message_parts = response.split('<|start|>')
            
            for part in message_parts:
                if not part.strip():
                    continue
                    
                # Extract role and channel
                role = None
                channel = None
                content = ""
                
                # Check for role
                if part.startswith('system'):
                    role = 'system'
                elif part.startswith('developer'):
                    role = 'developer'
                elif part.startswith('user'):
                    role = 'user'
                elif part.startswith('assistant'):
                    role = 'assistant'
                elif part.startswith('browser'):
                    role = 'tool'
                
                # Extract channel if present - look for channel information in the content
                if '<|channel|>' in part:
                    channel_start = part.find('<|channel|>') + len('<|channel|>')
                    channel_end = part.find('<|message|>')
                    if channel_end > channel_start:
                        channel = part[channel_start:channel_end].strip()
                else:
                    # Look for channel info in the content (like "# Valid channels: analysis, final")
                    if 'channels:' in part or 'channels' in part:
                        # Extract channel from text like "# Valid channels: analysis, final"
                        channel_match = re.search(r'channels?:\s*([^.\n]+)', part, re.IGNORECASE)
                        if channel_match:
                            channels_text = channel_match.group(1).strip()
                            # Extract first channel as default
                            if 'analysis' in channels_text:
                                channel = 'analysis'
                            elif 'final' in channels_text:
                                channel = 'final'
                
                # Extract message content
                if '<|message|>' in part:
                    message_start = part.find('<|message|>') + len('<|message|>')
                    message_end = part.find('<|end|>')
                    if message_end > message_start:
                        content = part[message_start:message_end].strip()
                else:
                    # If no <|message|> tag, extract content after role
                    if role:
                        # Find the end of the role
                        role_end = len(role)
                        # Look for <|end|> tag
                        end_tag = part.find('<|end|>')
                        if end_tag > role_end:
                            content = part[role_end:end_tag].strip()
                        else:
                            # If no end tag, take everything after role
                            content = part[role_end:].strip()
                
                # Only add messages with content and valid roles
                if content and role and role not in ['system', 'developer']:
                    message_data = {
                        "role": role,
                        "content": content
                    }
                    
                    if channel:
                        message_data["channel"] = channel
                    
                    # Add recipient if present (for tool calls)
                    if ' to=' in part:
                        recipient_start = part.find(' to=') + 4
                        recipient_end = part.find(' ', recipient_start)
                        if recipient_end == -1:
                            recipient_end = part.find('<', recipient_start)
                        if recipient_end > recipient_start:
                            recipient = part[recipient_start:recipient_end].strip()
                            message_data["recipient"] = recipient
                    
                    structured_data["messages"].append(message_data)
            
            # If we successfully parsed messages, return the structured data
            if structured_data["messages"]:
                return structured_data
            else:
                # If no messages found, check if we have a system message and create conversation from it
                system_content = ""
                for part in message_parts:
                    if part.startswith('system'):
                        if '<|message|>' in part:
                            message_start = part.find('<|message|>') + len('<|message|>')
                            message_end = part.find('<|end|>')
                            if message_end > message_start:
                                system_content = part[message_start:message_end].strip()
                        else:
                            # Extract content after 'system'
                            system_content = part[6:].strip()
                            if '<|end|>' in system_content:
                                system_content = system_content[:system_content.find('<|end|>')].strip()
                        break
                
                if system_content:
                    # Create conversation structure from system content
                    structured_data["messages"].append({
                        "role": "user",
                        "content": "أريد توليد شعر عربي",
                        "channel": "analysis"
                    })
                    
                    # Parse the system content to extract channel info
                    if 'channels:' in system_content:
                        channel_match = re.search(r'channels?:\s*([^.\n]+)', system_content, re.IGNORECASE)
                        if channel_match:
                            channels_text = channel_match.group(1).strip()
                            if 'analysis' in channels_text:
                                channel = 'analysis'
                            elif 'final' in channels_text:
                                channel = 'final'
                            else:
                                channel = 'analysis'
                        else:
                            channel = 'analysis'
                    
                    structured_data["messages"].append({
                        "role": "assistant",
                        "content": system_content,
                        "channel": channel
                    })
                    
                    return structured_data
                
                raise ValueError("No valid messages found in response")
                
        except Exception as e:
            raise ValueError(f"Failed to parse Harmony response: {str(e)}")
    
    def create_harmony_conversation(self, structured_data: dict):
        """
        Convert structured data to openai_harmony Conversation object
        
        Args:
            structured_data: Dictionary from generate_structured_harmony
            
        Returns:
            openai_harmony Conversation object
        """
        try:
            from openai_harmony import (
                Author, Conversation, DeveloperContent, Message, Role, 
                SystemContent, ToolDescription, ReasoningEffort
            )
            
            # Create system message
            sys_data = structured_data.get("system_message", {})
            system_message = (
                Message.from_role_and_content(Role.SYSTEM, sys_data.get("model_identity", "You are an Arabic Poetry Generation Agent."))
                .with_channel("analysis")
            )
            
            # Create developer message
            dev_data = structured_data.get("developer_message", {})
            developer_message = (
                Message.from_role_and_content(Role.ASSISTANT, dev_data.get("instructions", "You are a specialized Arabic poetry generation system."))
                .with_channel("analysis")
            )
            
            # Create messages from structured data
            messages = [system_message, developer_message]
            
            # Add user and assistant messages
            for msg_data in structured_data.get("messages", []):
                role = Role.USER if msg_data.get("role") == "user" else Role.ASSISTANT
                content = msg_data.get("content", "No content provided")
                message = Message.from_role_and_content(role, content)
                
                # Add channel if specified
                if "channel" in msg_data:
                    message = message.with_channel(msg_data["channel"])
                
                # Add recipient if specified
                if "recipient" in msg_data:
                    message = message.with_recipient(msg_data["recipient"])
                
                messages.append(message)
            
            # Create conversation
            conversation = Conversation.from_messages(messages)
            return conversation
            
        except ImportError:
            return {"error": "openai_harmony library not available"}
        except Exception as e:
            return {"error": f"Failed to create conversation: {str(e)}"}
    
    def _format_execution_steps(self, execution: PipelineExecution) -> str:
        """Format execution steps for the prompt"""
        steps = []
        
        for i, call in enumerate(execution.calls, 1):
            step_info = f"""
Step {i}: {call.component_name}.{call.method_name}
Type: {call.call_type}
Inputs: {json.dumps(self._serialize_output(call.inputs), ensure_ascii=False, indent=2)}
Outputs: {json.dumps(self._serialize_output(call.outputs), ensure_ascii=False, indent=2)}
Duration: {call.duration_ms}ms
Success: {call.success}
"""
            if call.error:
                step_info += f"Error: {call.error}\n"
            
            if call.llm_provider:
                step_info += f"""
LLM Call:
  Provider: {call.llm_provider}
  Model: {call.model_name}
  Tokens: {call.tokens_used}
  Prompt: {call.prompt[:200]}... (truncated)
  Response: {call.response[:200]}... (truncated)
"""
            
            steps.append(step_info)
        
        return "\n".join(steps)
    
    def _serialize_output(self, output: Any) -> Any:
        """Serialize output for JSON"""
        if hasattr(output, 'to_dict'):
            return output.to_dict()
        elif hasattr(output, '__dict__'):
            return {k: self._serialize_output(v) for k, v in output.__dict__.items() if not k.startswith('_')}
        elif isinstance(output, (list, tuple)):
            return [self._serialize_output(item) for item in output]
        elif isinstance(output, dict):
            return {k: self._serialize_output(v) for k, v in output.items()}
        else:
            return output
    
    def save_harmony_reasoning(self, reasoning: str, output_file: Path):
        """Save generated reasoning to file"""
        output_file.write_text(reasoning, encoding='utf-8')



