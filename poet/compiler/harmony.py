import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import json
import re
from poet.llm.base_llm import BaseLLM
from poet.prompts import get_global_prompt_manager
from poet.logging.harmony_capture import PipelineExecution
from poet.core.node import Node

class HarmonyCompiler:
    """
    Takes captured pipeline execution and generates Harmony-formatted
    reasoning that reconstructs the entire process as a coherent narrative
    """
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.prompt_manager = get_global_prompt_manager()
    
    def generate_structured_harmony(self, execution: Union[PipelineExecution, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate structured Harmony data from pipeline execution
        
        Args:
            execution: PipelineExecution object or its serialized dictionary
            
        Returns:
            Dictionary with structured conversation data
        """
        # Convert execution to dict if it's a PipelineExecution object
        if hasattr(execution, 'to_dict'):
            execution = execution.to_dict()
        
        # Get the prompt template from prompt manager
        template = self.prompt_manager.get_template("harmony_structured")
        

        # Get initial constraints from the execution data
        initial_constraints = execution.get('initial_constraints')
   
        serialized_constraints = self._serialize_output(initial_constraints)
        
        # Format execution steps with detailed information
        execution_steps = self._format_execution_steps(execution)
        
        prompt = template.format(
            user_prompt=execution.get('user_prompt', ''),
            initial_constraints=json.dumps(serialized_constraints, ensure_ascii=False, indent=2),
            execution_steps=execution_steps,
            final_poem=json.dumps(self._serialize_output(execution.get('final_poem')), ensure_ascii=False, indent=2),
            quality_assessment=json.dumps(self._serialize_output(execution.get('quality_assessment')), ensure_ascii=False, indent=2),
            conversation_start_date=execution.get('started_at', '')[:10] if isinstance(execution.get('started_at'), str) else execution.get('started_at', '').strftime('%Y-%m-%d'),
            cursor="1",
            toolname="browser",
            line_start="1",
            line_end="10",
            name="browser_tool",
            output="search_results",
            id="1",
            long_chain_of_thought="تفكير متسلسل طويل حول توليد الشعر العربي"
        )
        
        # # Debug: Print the formatted prompt
        # print("🔍 === HARMONY PROMPT BEING SENT TO LLM ===")
        # print(prompt)
        # print("🔍 === END OF HARMONY PROMPT ===")
        
        # Generate structured response
        response = self.llm.generate(prompt)
        
        
        # Parse the Harmony format response and convert to structured data
        try:
            structured_data = self._parse_harmony_response(response)
            
            # Ensure there is a final message; if missing, append one from execution facts
            has_final = any(
                msg.get("role") == "assistant" and msg.get("channel") == "final"
                for msg in structured_data.get("messages", [])
            )
            if not has_final:
                try:
                    
                    final_poem_json = json.dumps(
                        self._serialize_output(execution.get('final_poem')), ensure_ascii=False, indent=2
                    )
                    quality_json = json.dumps(
                        self._serialize_output(execution.get('quality_assessment')), ensure_ascii=False, indent=2
                    )
                    fallback_final = {
                        "role": "assistant",
                        "channel": "final",
                        "content": (
                            "Here is the final poem and its quality assessment (auto-appended):\n\n"
                            f"Final Poem:\n{final_poem_json}\n\n"
                        ),
                    }
                    structured_data.setdefault("messages", []).append(fallback_final)
                except Exception:
                    # If even fallback cannot be formed, proceed without it
                    pass
            
            print(f"✅ Successfully parsed Harmony response with {len(structured_data.get('messages', []))} messages")
            return structured_data
        except Exception as e:
            # Fallback to raw response if parsing fails
            print(f"❌ Failed to parse Harmony response: {str(e)}")
            print(f"🔍 Raw response: {response}")
            raise ValueError(f"Failed to parse Harmony response: {str(e)}")
    
    def _parse_harmony_response(self, response: str) -> dict:
        """
        Parse the LLM response and extract structured data
        
        Args:
            response: Raw LLM response
            
        Returns:
            Structured data dictionary
        """
        # Initialize structured data
        structured_data = {
            "messages": []
        }
        
        # Parse JSON response
        try:
            import json
            import re
            
            # First try to extract JSON from code blocks
            json_block_pattern = r'```(?:json)?\s*\n(.*?)\n```'
            json_blocks = re.findall(json_block_pattern, response, re.DOTALL)
            
            if json_blocks:
                json_str = json_blocks[0]
            else:
                # Fallback: find JSON between { and }
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start == -1 or json_end == 0 or json_end <= json_start:
                    raise ValueError("No JSON found in response")
                json_str = response[json_start:json_end]
            
            # Parse the JSON
            data = json.loads(json_str)
            
            # Convert to our expected format
            if 'analysis' in data:
                for step in data['analysis']:
                    message_data = {
                        "role": "assistant",
                        "channel": "analysis",
                        "content": step.get('explanation', '')
                    }
                    structured_data["messages"].append(message_data)
            
            if 'final_poem' in data:
                message_data = {
                    "role": "assistant", 
                    "channel": "final",
                    "content": data['final_poem']
                }
                structured_data["messages"].append(message_data)
            
            if structured_data["messages"]:
                return structured_data
            else:
                raise ValueError("No valid messages found in JSON response")
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to parse harmony response: {str(e)}")
    
    def _clean_message_content(self, content: str) -> str:
        """
        Clean up message content by removing only obvious artifacts while preserving analysis content
        
        Args:
            content: Raw message content
            
        Returns:
            Cleaned content string
        """
        if not content:
            return ""
        
        # Only remove very specific artifacts, not the actual analysis content
        
        # Remove only complete JSON code blocks that are clearly not part of the analysis
        # But preserve JSON that might be embedded in the analysis text
        content = re.sub(r'```json\s*\n.*?\n```', '', content, flags=re.DOTALL)
        
        # Remove only complete code blocks that are clearly not analysis
        content = re.sub(r'```\s*\n.*?\n```', '', content, flags=re.DOTALL)
        
        # Remove only very specific quality assessment patterns that are clearly artifacts
        # But preserve any analysis about quality
        content = re.sub(r'\*\*Quality Assessment:\*\*.*?(?=\n\n|\n$|$)', '', content, flags=re.DOTALL)
        
        # Remove only very specific conclusion patterns that are clearly artifacts
        # But preserve any analysis conclusions
        content = re.sub(r'\*\*Conclusion:\*\*.*?(?=\n\n|\n$|$)', '', content, flags=re.DOTALL)
        
        # Clean up excessive whitespace but preserve meaningful spacing
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = content.strip()
        
        return content
    
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
    
    def _format_execution_steps(self, execution: Union[PipelineExecution, Dict[str, Any]]) -> str:
        """Format execution steps with comprehensive information from each call type"""
        
        # Convert execution to dict if it's a PipelineExecution object
        if hasattr(execution, 'to_dict'):
            execution = execution.to_dict()
        
        calls = execution.get('calls', [])
        
        if not calls:
            return "The creative journey begins with understanding and planning."
        
        # Build comprehensive step-by-step narrative
        steps_narrative = []
        
        for i, call in enumerate(calls, 1):
            component = call.get('component_name', 'Unknown')
            method = call.get('method_name', 'Unknown')
            inputs = call.get('inputs', {})
            outputs = call.get('outputs', {})
            success = call.get('success', True)
            
            # Get detailed step information
            step_details = self._extract_meaningful_step_info(
                component, method, inputs, outputs, success, i, len(calls)
            )
            
            if step_details:
                steps_narrative.append(step_details)
        
        # Join all steps with clear separators
        result = "\n\n".join(steps_narrative)
        return result
    
    def _extract_meaningful_step_info(self, component: str, method: str, inputs: dict, outputs: dict, success: bool, step_num: int, total_steps: int) -> str:
        """Extract essential information for each step in clean, structured format"""
        
        step_info = []
        
        # Add step header
        step_info.append(f"Step {step_num}: {component}")
        
        # Add success/failure status
        status = "✅ SUCCESS" if success else "❌ FAILED"
        step_info.append(f"Status: {status}")
        
        # Extract only relevant inputs (filter out LLM configs and technical details)
        relevant_inputs = self._extract_relevant_inputs(component, inputs)
        if relevant_inputs:
            step_info.append(f"Input: {relevant_inputs}")
        else:
            step_info.append("Input: None")
        
        # Extract only relevant outputs
        relevant_outputs = self._extract_relevant_outputs(component, outputs)
        if relevant_outputs:
            step_info.append(f"Output: {relevant_outputs}")
        else:
            step_info.append("Output: None")
        
        return "\n".join(step_info)
    
    def _extract_relevant_inputs(self, component: str, inputs: dict) -> str:
        """Extract only the relevant input components for each step type based on actual component implementations"""
        
        if component == 'ConstraintParser':
            # Based on actual implementation: takes user_prompt and returns constraints
            user_prompt = inputs.get('arg_0', {}).get('user_prompt', '')
            if user_prompt:
                return f"User poetry request: {user_prompt}"
            return "User poetry request"
        
        elif component == 'QafiyaSelector':
            # Based on actual implementation: takes constraints and user_prompt, returns enhanced constraints
            constraints = inputs.get('arg_0', {}).get('constraints', {})
            if isinstance(constraints, dict):
                theme = constraints.get('theme', 'unspecified')
                meter = constraints.get('meter', 'unspecified')
                qafiya = constraints.get('qafiya', 'unspecified')
                line_count = constraints.get('line_count', 'unspecified')
                return f"Constraints: {theme} theme, {meter} meter, {qafiya} rhyme, {line_count} lines"
            return "Poetry constraints"
        
        elif component == 'BahrSelector':
            # Based on actual implementation: takes constraints and user_prompt, returns enhanced constraints
            constraints = inputs.get('arg_0', {}).get('constraints', {})
            if isinstance(constraints, dict):
                theme = constraints.get('theme', 'unspecified')
                qafiya = constraints.get('qafiya', 'unspecified')
                meter = constraints.get('meter', 'unspecified')
                line_count = constraints.get('line_count', 'unspecified')
                return f"Constraints: {theme} theme, {meter} meter, {qafiya} rhyme, {line_count} lines"
            return "Poetry constraints"
        
        elif component == 'SimplePoemGenerator':
            # Based on actual implementation: takes constraints, returns poem
            constraints = inputs.get('arg_0', {}).get('constraints', {})
            if isinstance(constraints, dict):
                theme = constraints.get('theme', 'unspecified')
                meter = constraints.get('meter', 'unspecified')
                qafiya = constraints.get('qafiya', 'unspecified')
                line_count = constraints.get('line_count', 'unspecified')
                tone = constraints.get('tone', 'unspecified')
                return f"Requirements: {theme} theme, {tone} tone, {meter} meter, {qafiya} rhyme, {line_count} lines"
            return "Poem generation requirements"
        
        elif component == 'PoemEvaluator':
            # Based on actual implementation: takes poem and constraints, returns evaluation
            poem = inputs.get('arg_0', {}).get('poem', {})
            if isinstance(poem, dict) and 'verses' in poem:
                verses = poem['verses']
                if isinstance(verses, list):
                    return f"Poem to evaluate: {len(verses)} verses"
            return "Generated poem"
        
        elif component == 'prosody_refiner':
            # Based on actual implementation: takes poem, constraints, and evaluation, returns refined poem
            poem = inputs.get('arg_0', {}).get('poem', {})
            if isinstance(poem, dict) and 'verses' in poem:
                verses = poem['verses']
                if isinstance(verses, list):
                    return f"Poem with {len(verses)} verses needing prosody refinement"
            return "Poem needing prosody refinement"
        
        elif component == 'qafiya_refiner':
            # Based on actual implementation: takes poem, constraints, and evaluation, returns refined poem
            poem = inputs.get('arg_0', {}).get('poem', {})
            if isinstance(poem, dict) and 'verses' in poem:
                verses = poem['verses']
                if isinstance(verses, list):
                    return f"Poem with {len(verses)} verses needing qafiya refinement"
            return "Poem needing qafiya refinement"
        
        elif component == 'RefinerChain':
            # Based on actual implementation: takes poem and constraints, returns refined poem
            poem = inputs.get('arg_0', {}).get('poem', {})
            if isinstance(poem, dict) and 'verses' in poem:
                verses = poem['verses']
                if isinstance(verses, list):
                    return f"Poem with {len(verses)} verses for refinement chain"
            return "Poem for refinement chain"
        
        elif component == 'KnowledgeRetriever':
            # Based on actual implementation: takes constraints, returns retrieval results
            constraints = inputs.get('arg_0', {}).get('constraints', {})
            if isinstance(constraints, dict):
                theme = constraints.get('theme', 'unspecified')
                meter = constraints.get('meter', 'unspecified')
                return f"Knowledge search for: {theme} theme, {meter} meter"
            return "Knowledge retrieval request"
        
        else:
            # Generic case - extract first meaningful input
            if inputs is None:
                return "Processing request"
            for key, value in inputs.items():
                if key.startswith('arg_') and isinstance(value, dict):
                    if 'user_prompt' in value:
                        return f"User request: {value['user_prompt']}"
                    elif 'constraints' in value:
                        return "Poetry constraints"
                    elif 'poem' in value:
                        return "Poem data"
            return "Processing request"
    
    def _extract_relevant_outputs(self, component: str, outputs: dict) -> str:
        """Extract only the relevant output components for each step type based on actual component implementations"""
        
        if component == 'ConstraintParser':
            # Based on actual implementation: returns constraints and parsed_constraints flag
            if outputs and 'constraints' in outputs:
                constraints = outputs['constraints']
                
                if hasattr(constraints, 'theme') and hasattr(constraints, 'meter'):
                    theme = getattr(constraints, 'theme', 'unspecified')
                    meter = getattr(constraints, 'meter', 'unspecified')
                    qafiya = getattr(constraints, 'qafiya', 'unspecified')
                    line_count = getattr(constraints, 'line_count', 'unspecified')
                    tone = getattr(constraints, 'tone', 'unspecified')
                    return f"Parsed: {theme} theme, {tone} tone, {meter} meter, {qafiya} rhyme, {line_count} lines"
                elif isinstance(constraints, dict):
                    theme = constraints.get('theme', 'unspecified')
                    meter = constraints.get('meter', 'unspecified')
                    qafiya = constraints.get('qafiya', 'unspecified')
                    line_count = constraints.get('line_count', 'unspecified')
                    tone = constraints.get('tone', 'unspecified')
                    return f"Parsed: {theme} theme, {tone} tone, {meter} meter, {qafiya} rhyme, {line_count} lines"
            return "Parsed poetry constraints"
        
        elif component == 'QafiyaSelector':
            # Based on actual implementation: returns enhanced constraints and qafiya_selected flag
            if outputs and 'constraints' in outputs:
                constraints = outputs['constraints']
                
                if hasattr(constraints, 'qafiya') and hasattr(constraints, 'qafiya_type'):
                    qafiya = getattr(constraints, 'qafiya', 'unknown')
                    qafiya_type = getattr(constraints, 'qafiya_type', 'unknown')
                    if hasattr(qafiya_type, 'value'):
                        qafiya_type = qafiya_type.value
                    qafiya_harakah = getattr(constraints, 'qafiya_harakah', 'unknown')
                    
                    # Add detailed rhyme pattern information
                    rhyme_info = f"Selected: {qafiya} rhyme ({qafiya_type} pattern, {qafiya_harakah} harakah)"
                    
                    # Add qafiya type description and examples if available
                    if hasattr(constraints, 'qafiya_type_description_and_examples') and constraints.qafiya_type_description_and_examples:
                        qafiya_desc = getattr(constraints, 'qafiya_type_description_and_examples', '')
                        if qafiya_desc:
                            rhyme_info += f"\nRhyme pattern details: {qafiya_desc}"
                    
                    return rhyme_info
                elif isinstance(constraints, dict):
                    qafiya = constraints.get('qafiya', 'unknown')
                    qafiya_type = constraints.get('qafiya_type', 'unknown')
                    qafiya_harakah = constraints.get('qafiya_harakah', 'unknown')
                    
                    # Add detailed rhyme pattern information
                    rhyme_info = f"Selected: {qafiya} rhyme ({qafiya_type} pattern, {qafiya_harakah} harakah)"
                    
                    # Add qafiya type description and examples if available
                    if 'qafiya_type_description_and_examples' in constraints and constraints['qafiya_type_description_and_examples']:
                        qafiya_desc = constraints['qafiya_type_description_and_examples']
                        if qafiya_desc:
                            rhyme_info += f"\nRhyme pattern details: {qafiya_desc}"
                    
                    return rhyme_info
            return "Rhyme scheme selected"
        
        elif component == 'BahrSelector':
            # Based on actual implementation: returns enhanced constraints and bahr_selected flag
            if outputs and 'constraints' in outputs:
                constraints = outputs['constraints']
                
                if hasattr(constraints, 'meter') and hasattr(constraints, 'meeter_tafeelat'):
                    meter = getattr(constraints, 'meter', 'unknown')
                    tafeelat = getattr(constraints, 'meeter_tafeelat', 'unknown')
                    return f"Selected: {meter} meter with {tafeelat} pattern"
                elif isinstance(constraints, dict):
                    meter = constraints.get('meter', 'unknown')
                    tafeelat = constraints.get('meeter_tafeelat', 'unknown')
                    return f"Selected: {meter} meter with {tafeelat} pattern"
            return "Meter pattern selected"
        
        elif component == 'SimplePoemGenerator':
            # Based on actual implementation: returns poem and poem_generated flag
            if outputs and 'poem' in outputs:
                poem = outputs['poem']
                
                if hasattr(poem, 'verses'):
                    verses = poem.verses
                    if isinstance(verses, list) and len(verses) > 0:
                        # Show all verses, not just the first one
                        verses_text = "\n".join([f"Verse {i+1}: {verse}" for i, verse in enumerate(verses)])
                        return f"Generated: {len(verses)} verses\n{verses_text}"
                elif isinstance(poem, dict) and 'verses' in poem:
                    verses = poem['verses']
                    if isinstance(verses, list) and len(verses) > 0:
                        # Show all verses, not just the first one
                        verses_text = "\n".join([f"Verse {i+1}: {verse}" for i, verse in enumerate(verses)])
                        return f"Generated: {len(verses)} verses\n{verses_text}"
            return "Poem verses created"
        
        elif component == 'PoemEvaluator':
            # Based on actual implementation: returns evaluation, evaluated flag, and updated poem
            if outputs and 'poem' in outputs:
                poem = outputs['poem']
                eval_info = outputs.get('evaluation', {})
                
                # Get basic scores from evaluation
                overall = eval_info.get('overall_score', 'unknown')
                prosody = eval_info.get('prosody_score', 'unknown')
                qafiya_score = eval_info.get('qafiya_score', 'unknown')
                
                # Show detailed evaluation feedback
                feedback = f"Scores: overall {overall}, prosody {prosody}, qafiya {qafiya_score}"
                
                # Extract detailed validation information from poem.quality
                if hasattr(poem, 'quality') and poem.quality:
                    quality = poem.quality
                    
                    # Add specific issues found with actual error details
                    if hasattr(quality, 'prosody_issues') and quality.prosody_issues:
                        issues = quality.prosody_issues
                        if isinstance(issues, list) and len(issues) > 0:
                            feedback += f"\nProsody issues ({len(issues)} found):"
                            for i, issue in enumerate(issues[:3]):  # Show first 3 issues
                                feedback += f"\n  - Issue {i+1}: {issue}"
                    
                    if hasattr(quality, 'qafiya_issues') and quality.qafiya_issues:
                        issues = quality.qafiya_issues
                        if isinstance(issues, list) and len(issues) > 0:
                            feedback += f"\nQafiya issues ({len(issues)} found):"
                            for i, issue in enumerate(issues[:3]):  # Show first 3 issues
                                feedback += f"\n  - Issue {i+1}: {issue}"
                    
                    if hasattr(quality, 'line_count_issues') and quality.line_count_issues:
                        issues = quality.line_count_issues
                        if isinstance(issues, list) and len(issues) > 0:
                            feedback += f"\nLine count issues ({len(issues)} found):"
                            for i, issue in enumerate(issues[:3]):  # Show first 3 issues
                                feedback += f"\n  - Issue {i+1}: {issue}"
                    
                    if hasattr(quality, 'tashkeel_issues') and quality.tashkeel_issues:
                        issues = quality.tashkeel_issues
                        if isinstance(issues, list) and len(issues) > 0:
                            feedback += f"\nTashkeel issues ({len(issues)} found):"
                            for i, issue in enumerate(issues[:3]):  # Show first 3 issues
                                feedback += f"\n  - Issue {i+1}: {issue}"
                    
                    # Add detailed validation results if available
                    if hasattr(quality, 'prosody_validation') and quality.prosody_validation:
                        prosody_val = quality.prosody_validation
                        if hasattr(prosody_val, 'overall_valid'):
                            valid_baits = getattr(prosody_val, 'valid_baits', 0)
                            total_baits = getattr(prosody_val, 'total_baits', 0)
                            feedback += f"\nProsody validation: {valid_baits}/{total_baits} baits correct"
                            # Add detailed prosody validation results
                            if hasattr(prosody_val, 'bait_results') and prosody_val.bait_results:
                                feedback += f"\n  Prosody details:"
                                for k, bait_result in enumerate(prosody_val.bait_results):
                                    is_valid = getattr(bait_result, 'is_valid', False)
                                    error_details = getattr(bait_result, 'error_details', '')
                                    pattern = getattr(bait_result, 'pattern', '')
                                    feedback += f"\n    Bait {k+1}: {'✅' if is_valid else '❌'} {pattern}"
                                    if not is_valid and error_details:
                                        feedback += f" - {error_details}"
                            if hasattr(prosody_val, 'validation_summary'):
                                summary = getattr(prosody_val, 'validation_summary', '')
                                if summary:
                                    feedback += f"\n  Summary: {summary}"
                    
                    if hasattr(quality, 'qafiya_validation') and quality.qafiya_validation:
                        qafiya_val = quality.qafiya_validation
                        if hasattr(qafiya_val, 'overall_valid'):
                            valid_baits = getattr(qafiya_val, 'valid_baits', 0)
                            total_baits = getattr(qafiya_val, 'total_baits', 0)
                            feedback += f"\nQafiya validation: {valid_baits}/{total_baits} baits correct"
                            # Add detailed qafiya validation results
                            if hasattr(qafiya_val, 'bait_results') and qafiya_val.bait_results:
                                feedback += f"\n  Qafiya details:"
                                for k, bait_result in enumerate(qafiya_val.bait_results):
                                    is_valid = getattr(bait_result, 'is_valid', False)
                                    error_details = getattr(bait_result, 'error_details', '')
                                    feedback += f"\n    Bait {k+1}: {'✅' if is_valid else '❌'}"
                                    if not is_valid and error_details:
                                        feedback += f" - {error_details}"
                            if hasattr(qafiya_val, 'validation_summary'):
                                summary = getattr(qafiya_val, 'validation_summary', '')
                                if summary:
                                    feedback += f"\n  Summary: {summary}"
                    
                    # Add line count validation if available
                    if hasattr(quality, 'line_count_validation') and quality.line_count_validation:
                        line_count_val = quality.line_count_validation
                        if hasattr(line_count_val, 'is_valid'):
                            is_valid = getattr(line_count_val, 'is_valid', False)
                            validation_summary = getattr(line_count_val, 'validation_summary', '')
                            feedback += f"\nLine count validation: {'✅ Valid' if is_valid else '❌ Invalid'}"
                            if validation_summary:
                                feedback += f"\n  Summary: {validation_summary}"
                    
                    # Add tashkeel validation if available
                    if hasattr(quality, 'tashkeel_validation') and quality.tashkeel_validation:
                        tashkeel_val = quality.tashkeel_validation
                        if hasattr(tashkeel_val, 'is_valid'):
                            is_valid = getattr(tashkeel_val, 'is_valid', False)
                            validation_summary = getattr(tashkeel_val, 'validation_summary', '')
                            feedback += f"\nTashkeel validation: {'✅ Valid' if is_valid else '❌ Invalid'}"
                            if validation_summary:
                                feedback += f"\n  Summary: {validation_summary}"
                    
                    # Add recommendations if available
                    if hasattr(quality, 'recommendations') and quality.recommendations:
                        recs = quality.recommendations
                        if isinstance(recs, list) and len(recs) > 0:
                            feedback += f"\nRecommendations:"
                            for i, rec in enumerate(recs[:3]):  # Show first 3 recommendations
                                feedback += f"\n  - Rec {i+1}: {rec}"
                
                # Also handle dictionary-based poem structure
                elif isinstance(poem, dict) and 'quality' in poem and poem['quality']:
                    quality = poem['quality']
                    
                    # Add specific issues found with actual error details
                    if 'prosody_issues' in quality and quality['prosody_issues']:
                        issues = quality['prosody_issues']
                        if isinstance(issues, list) and len(issues) > 0:
                            feedback += f"\nProsody issues ({len(issues)} found):"
                            for i, issue in enumerate(issues[:3]):  # Show first 3 issues
                                feedback += f"\n  - Issue {i+1}: {issue}"
                    
                    if 'qafiya_issues' in quality and quality['qafiya_issues']:
                        issues = quality['qafiya_issues']
                        if isinstance(issues, list) and len(issues) > 0:
                            feedback += f"\nQafiya issues ({len(issues)} found):"
                            for i, issue in enumerate(issues[:3]):  # Show first 3 issues
                                feedback += f"\n  - Issue {i+1}: {issue}"
                    
                    if 'line_count_issues' in quality and quality['line_count_issues']:
                        issues = quality['line_count_issues']
                        if isinstance(issues, list) and len(issues) > 0:
                            feedback += f"\nLine count issues ({len(issues)} found):"
                            for i, issue in enumerate(issues[:3]):  # Show first 3 issues
                                feedback += f"\n  - Issue {i+1}: {issue}"
                    
                    if 'tashkeel_issues' in quality and quality['tashkeel_issues']:
                        issues = quality['tashkeel_issues']
                        if isinstance(issues, list) and len(issues) > 0:
                            feedback += f"\nTashkeel issues ({len(issues)} found):"
                            for i, issue in enumerate(issues[:3]):  # Show first 3 issues
                                feedback += f"\n  - Issue {i+1}: {issue}"
                    
                    # Add detailed validation results if available
                    if 'prosody_validation' in quality and quality['prosody_validation']:
                        prosody_val = quality['prosody_validation']
                        valid_baits = prosody_val.get('valid_baits', 0)
                        total_baits = prosody_val.get('total_baits', 0)
                        feedback += f"\nProsody validation: {valid_baits}/{total_baits} baits correct"
                        # Add detailed prosody validation results
                        if 'bait_results' in prosody_val and prosody_val['bait_results']:
                            feedback += f"\n  Prosody details:"
                            for k, bait_result in enumerate(prosody_val['bait_results']):
                                is_valid = bait_result.get('is_valid', False)
                                error_details = bait_result.get('error_details', '')
                                pattern = bait_result.get('pattern', '')
                                feedback += f"\n    Bait {k+1}: {'✅' if is_valid else '❌'} {pattern}"
                                if not is_valid and error_details:
                                    feedback += f" - {error_details}"
                        if 'validation_summary' in prosody_val:
                            summary = prosody_val['validation_summary']
                            if summary:
                                feedback += f"\n  Summary: {summary}"
                    
                    if 'qafiya_validation' in quality and quality['qafiya_validation']:
                        qafiya_val = quality['qafiya_validation']
                        valid_baits = qafiya_val.get('valid_baits', 0)
                        total_baits = qafiya_val.get('total_baits', 0)
                        feedback += f"\nQafiya validation: {valid_baits}/{total_baits} baits correct"
                        # Add detailed qafiya validation results
                        if 'bait_results' in qafiya_val and qafiya_val['bait_results']:
                            feedback += f"\n  Qafiya details:"
                            for k, bait_result in enumerate(qafiya_val['bait_results']):
                                is_valid = bait_result.get('is_valid', False)
                                error_details = bait_result.get('error_details', '')
                                feedback += f"\n    Bait {k+1}: {'✅' if is_valid else '❌'}"
                                if not is_valid and error_details:
                                    feedback += f" - {error_details}"
                        if 'validation_summary' in qafiya_val:
                            summary = qafiya_val['validation_summary']
                            if summary:
                                feedback += f"\n  Summary: {summary}"
                    
                    # Add line count validation if available
                    if 'line_count_validation' in quality and quality['line_count_validation']:
                        line_count_val = quality['line_count_validation']
                        is_valid = line_count_val.get('is_valid', False)
                        validation_summary = line_count_val.get('validation_summary', '')
                        feedback += f"\nLine count validation: {'✅ Valid' if is_valid else '❌ Invalid'}"
                        if validation_summary:
                            feedback += f"\n  Summary: {validation_summary}"
                    
                    # Add tashkeel validation if available
                    if 'tashkeel_validation' in quality and quality['tashkeel_validation']:
                        tashkeel_val = quality['tashkeel_validation']
                        is_valid = tashkeel_val.get('is_valid', False)
                        validation_summary = tashkeel_val.get('validation_summary', '')
                        feedback += f"\nTashkeel validation: {'✅ Valid' if is_valid else '❌ Invalid'}"
                        if validation_summary:
                            feedback += f"\n  Summary: {validation_summary}"
                    
                    # Add recommendations if available
                    if 'recommendations' in quality and quality['recommendations']:
                        recs = quality['recommendations']
                        if isinstance(recs, list) and len(recs) > 0:
                            feedback += f"\nRecommendations:"
                            for i, rec in enumerate(recs[:3]):  # Show first 3 recommendations
                                feedback += f"\n  - Rec {i+1}: {rec}"
                
                return feedback
            return "Poem quality evaluated"
        
        elif component == 'prosody_refiner':
            # Based on actual implementation: returns refined poem, refined flag, and iterations count
            if outputs and 'poem' in outputs:
                poem = outputs['poem']
                refined = outputs.get('refined', False)
                iterations = outputs.get('refinement_iterations', 0)
                
                if refined:
                    # Show the refined poem content with quality information
                    if hasattr(poem, 'verses'):
                        verses = poem.verses
                        if isinstance(verses, list) and len(verses) > 0:
                            verses_text = "\n".join([f"Refined Verse {i+1}: {verse}" for i, verse in enumerate(verses)])
                            quality_info = ""
                            if hasattr(poem, 'quality') and poem.quality:
                                if hasattr(poem.quality, 'overall_score'):
                                    quality_info = f"\nQuality score: {poem.quality.overall_score}"
                                    # Add detailed validation details if available
                                    if hasattr(poem.quality, 'prosody_validation') and poem.quality.prosody_validation:
                                        prosody_val = poem.quality.prosody_validation
                                        if hasattr(prosody_val, 'overall_valid'):
                                            valid_baits = getattr(prosody_val, 'valid_baits', 0)
                                            total_baits = getattr(prosody_val, 'total_baits', 0)
                                            quality_info += f"\nProsody: {valid_baits}/{total_baits} baits correct"
                                            # Add detailed prosody validation results
                                            if hasattr(prosody_val, 'bait_results') and prosody_val.bait_results:
                                                quality_info += f"\n  Prosody details:"
                                                for k, bait_result in enumerate(prosody_val.bait_results):
                                                    is_valid = getattr(bait_result, 'is_valid', False)
                                                    error_details = getattr(bait_result, 'error_details', '')
                                                    pattern = getattr(bait_result, 'pattern', '')
                                                    quality_info += f"\n    Bait {k+1}: {'✅' if is_valid else '❌'} {pattern}"
                                                    if not is_valid and error_details:
                                                        quality_info += f" - {error_details}"
                                            if hasattr(prosody_val, 'validation_summary'):
                                                summary = getattr(prosody_val, 'validation_summary', '')
                                                if summary:
                                                    quality_info += f"\n  Summary: {summary}"
                                    
                                    if hasattr(poem.quality, 'qafiya_validation') and poem.quality.qafiya_validation:
                                        qafiya_val = poem.quality.qafiya_validation
                                        if hasattr(qafiya_val, 'overall_valid'):
                                            valid_baits = getattr(qafiya_val, 'valid_baits', 0)
                                            total_baits = getattr(qafiya_val, 'total_baits', 0)
                                            quality_info += f"\nQafiya: {valid_baits}/{total_baits} baits correct"
                                            # Add detailed qafiya validation results
                                            if hasattr(qafiya_val, 'bait_results') and qafiya_val.bait_results:
                                                quality_info += f"\n  Qafiya details:"
                                                for k, bait_result in enumerate(qafiya_val.bait_results):
                                                    is_valid = getattr(bait_result, 'is_valid', False)
                                                    error_details = getattr(bait_result, 'error_details', '')
                                                    quality_info += f"\n    Bait {k+1}: {'✅' if is_valid else '❌'}"
                                                    if not is_valid and error_details:
                                                        quality_info += f" - {error_details}"
                                            if hasattr(qafiya_val, 'validation_summary'):
                                                summary = getattr(qafiya_val, 'validation_summary', '')
                                                if summary:
                                                    quality_info += f"\n  Summary: {summary}"
                                    
                                    # Add overall quality issues if available
                                    if hasattr(poem.quality, 'prosody_issues') and poem.quality.prosody_issues:
                                        issues = poem.quality.prosody_issues
                                        if isinstance(issues, list) and len(issues) > 0:
                                            quality_info += f"\nProsody issues:"
                                            for issue in issues[:3]:  # Show first 3 issues
                                                quality_info += f"\n  - {issue}"
                                    
                                    if hasattr(poem.quality, 'qafiya_issues') and poem.quality.qafiya_issues:
                                        issues = poem.quality.qafiya_issues
                                        if isinstance(issues, list) and len(issues) > 0:
                                            quality_info += f"\nQafiya issues:"
                                            for issue in issues[:3]:  # Show first 3 issues
                                                quality_info += f"\n  - {issue}"
                            
                            return f"Prosody refined through {iterations} iteration(s){quality_info}\n{verses_text}"
                    elif isinstance(poem, dict) and 'verses' in poem:
                        verses = poem['verses']
                        if isinstance(verses, list) and len(verses) > 0:
                            verses_text = "\n".join([f"Refined Verse {i+1}: {verse}" for i, verse in enumerate(verses)])
                            quality_info = ""
                            if 'quality' in poem and poem['quality']:
                                quality_score = poem['quality'].get('overall_score', 'unknown')
                                quality_info = f"\nQuality score: {quality_score}"
                                # Add detailed validation details if available
                                if 'prosody_validation' in poem['quality'] and poem['quality']['prosody_validation']:
                                    prosody_val = poem['quality']['prosody_validation']
                                    valid_baits = prosody_val.get('valid_baits', 0)
                                    total_baits = prosody_val.get('total_baits', 0)
                                    quality_info += f"\nProsody: {valid_baits}/{total_baits} baits correct"
                                    # Add detailed prosody validation results
                                    if 'bait_results' in prosody_val and prosody_val['bait_results']:
                                        quality_info += f"\n  Prosody details:"
                                        for k, bait_result in enumerate(prosody_val['bait_results']):
                                            is_valid = bait_result.get('is_valid', False)
                                            error_details = bait_result.get('error_details', '')
                                            pattern = bait_result.get('pattern', '')
                                            quality_info += f"\n    Bait {k+1}: {'✅' if is_valid else '❌'} {pattern}"
                                            if not is_valid and error_details:
                                                quality_info += f" - {error_details}"
                                    if 'validation_summary' in prosody_val:
                                        summary = prosody_val['validation_summary']
                                        if summary:
                                            quality_info += f"\n  Summary: {summary}"
                                
                                if 'qafiya_validation' in poem['quality'] and poem['quality']['qafiya_validation']:
                                    qafiya_val = poem['quality']['qafiya_validation']
                                    valid_baits = qafiya_val.get('valid_baits', 0)
                                    total_baits = qafiya_val.get('total_baits', 0)
                                    quality_info += f"\nQafiya: {valid_baits}/{total_baits} baits correct"
                                    # Add detailed qafiya validation results
                                    if 'bait_results' in qafiya_val and qafiya_val['bait_results']:
                                        quality_info += f"\n  Qafiya details:"
                                        for k, bait_result in enumerate(qafiya_val['bait_results']):
                                            is_valid = bait_result.get('is_valid', False)
                                            error_details = bait_result.get('error_details', '')
                                            quality_info += f"\n    Bait {k+1}: {'✅' if is_valid else '❌'}"
                                            if not is_valid and error_details:
                                                quality_info += f" - {error_details}"
                                    if 'validation_summary' in qafiya_val:
                                        summary = qafiya_val['validation_summary']
                                        if summary:
                                            quality_info += f"\n  Summary: {summary}"
                                
                                # Add overall quality issues if available
                                if 'prosody_issues' in poem['quality'] and poem['quality']['prosody_issues']:
                                    issues = poem['quality']['prosody_issues']
                                    if isinstance(issues, list) and len(issues) > 0:
                                        quality_info += f"\nProsody issues:"
                                        for issue in issues[:3]:  # Show first 3 issues
                                            quality_info += f"\n  - {issue}"
                                
                                if 'qafiya_issues' in poem['quality'] and poem['quality']['qafiya_issues']:
                                    issues = poem['quality']['qafiya_issues']
                                    if isinstance(issues, list) and len(issues) > 0:
                                        quality_info += f"\nQafiya issues:"
                                        for issue in issues[:3]:  # Show first 3 issues
                                            quality_info += f"\n  - {issue}"
                            
                            return f"Prosody refined through {iterations} iteration(s){quality_info}\n{verses_text}"
                    
                    return f"Prosody refined through {iterations} iteration(s)"
                else:
                    return "No prosody refinement needed"
            return "Prosody refinement completed"
        
        elif component == 'qafiya_refiner':
            # Based on actual implementation: returns refined poem, refined flag, and iterations count
            if outputs and 'poem' in outputs:
                poem = outputs['poem']
                refined = outputs.get('refined', False)
                iterations = outputs.get('refinement_iterations', 0)
                
                if refined:
                    # Show the refined poem content with quality information
                    if hasattr(poem, 'verses'):
                        verses = poem.verses
                        if isinstance(verses, list) and len(verses) > 0:
                            verses_text = "\n".join([f"Refined Verse {i+1}: {verse}" for i, verse in enumerate(verses)])
                            quality_info = ""
                            if hasattr(poem, 'quality') and poem.quality:
                                if hasattr(poem.quality, 'overall_score'):
                                    quality_info = f"\nQuality score: {poem.quality.overall_score}"
                                    # Add detailed validation details if available
                                    if hasattr(poem.quality, 'prosody_validation') and poem.quality.prosody_validation:
                                        prosody_val = poem.quality.prosody_validation
                                        if hasattr(prosody_val, 'overall_valid'):
                                            valid_baits = getattr(prosody_val, 'valid_baits', 0)
                                            total_baits = getattr(prosody_val, 'total_baits', 0)
                                            quality_info += f"\nProsody: {valid_baits}/{total_baits} baits correct"
                                            # Add detailed prosody validation results
                                            if hasattr(prosody_val, 'bait_results') and prosody_val.bait_results:
                                                quality_info += f"\n  Prosody details:"
                                                for k, bait_result in enumerate(prosody_val.bait_results):
                                                    is_valid = getattr(bait_result, 'is_valid', False)
                                                    error_details = getattr(bait_result, 'error_details', '')
                                                    pattern = getattr(bait_result, 'pattern', '')
                                                    quality_info += f"\n    Bait {k+1}: {'✅' if is_valid else '❌'} {pattern}"
                                                    if not is_valid and error_details:
                                                        quality_info += f" - {error_details}"
                                            if hasattr(prosody_val, 'validation_summary'):
                                                summary = getattr(prosody_val, 'validation_summary', '')
                                                if summary:
                                                    quality_info += f"\n  Summary: {summary}"
                                    
                                    if hasattr(poem.quality, 'qafiya_validation') and poem.quality.qafiya_validation:
                                        qafiya_val = poem.quality.qafiya_validation
                                        if hasattr(qafiya_val, 'overall_valid'):
                                            valid_baits = getattr(qafiya_val, 'valid_baits', 0)
                                            total_baits = getattr(qafiya_val, 'total_baits', 0)
                                            quality_info += f"\nQafiya: {valid_baits}/{total_baits} baits correct"
                                            # Add detailed qafiya validation results
                                            if hasattr(qafiya_val, 'bait_results') and qafiya_val.bait_results:
                                                quality_info += f"\n  Qafiya details:"
                                                for k, bait_result in enumerate(qafiya_val.bait_results):
                                                    is_valid = getattr(bait_result, 'is_valid', False)
                                                    error_details = getattr(bait_result, 'error_details', '')
                                                    quality_info += f"\n    Bait {k+1}: {'✅' if is_valid else '❌'}"
                                                    if not is_valid and error_details:
                                                        quality_info += f" - {error_details}"
                                            if hasattr(qafiya_val, 'validation_summary'):
                                                summary = getattr(qafiya_val, 'validation_summary', '')
                                                if summary:
                                                    quality_info += f"\n  Summary: {summary}"
                                    
                                    # Add overall quality issues if available
                                    if hasattr(poem.quality, 'prosody_issues') and poem.quality.prosody_issues:
                                        issues = poem.quality.prosody_issues
                                        if isinstance(issues, list) and len(issues) > 0:
                                            quality_info += f"\nProsody issues:"
                                            for issue in issues[:3]:  # Show first 3 issues
                                                quality_info += f"\n  - {issue}"
                                    
                                    if hasattr(poem.quality, 'qafiya_issues') and poem.quality.qafiya_issues:
                                        issues = poem.quality.qafiya_issues
                                        if isinstance(issues, list) and len(issues) > 0:
                                            quality_info += f"\nQafiya issues:"
                                            for issue in issues[:3]:  # Show first 3 issues
                                                quality_info += f"\n  - {issue}"
                            
                            return f"Qafiya refined through {iterations} iteration(s){quality_info}\n{verses_text}"
                    elif isinstance(poem, dict) and 'verses' in poem:
                        verses = poem['verses']
                        if isinstance(verses, list) and len(verses) > 0:
                            verses_text = "\n".join([f"Refined Verse {i+1}: {verse}" for i, verse in enumerate(verses)])
                            quality_info = ""
                            if 'quality' in poem and poem['quality']:
                                quality_score = poem['quality'].get('overall_score', 'unknown')
                                quality_info = f"\nQuality score: {quality_score}"
                                # Add detailed validation details if available
                                if 'prosody_validation' in poem['quality'] and poem['quality']['prosody_validation']:
                                    prosody_val = poem['quality']['prosody_validation']
                                    valid_baits = prosody_val.get('valid_baits', 0)
                                    total_baits = prosody_val.get('total_baits', 0)
                                    quality_info += f"\nProsody: {valid_baits}/{total_baits} baits correct"
                                    # Add detailed prosody validation results
                                    if 'bait_results' in prosody_val and prosody_val['bait_results']:
                                        quality_info += f"\n  Prosody details:"
                                        for k, bait_result in enumerate(prosody_val['bait_results']):
                                            is_valid = bait_result.get('is_valid', False)
                                            error_details = bait_result.get('error_details', '')
                                            pattern = bait_result.get('pattern', '')
                                            quality_info += f"\n    Bait {k+1}: {'✅' if is_valid else '❌'} {pattern}"
                                            if not is_valid and error_details:
                                                quality_info += f" - {error_details}"
                                    if 'validation_summary' in prosody_val:
                                        summary = prosody_val['validation_summary']
                                        if summary:
                                            quality_info += f"\n  Summary: {summary}"
                                
                                if 'qafiya_validation' in poem['quality'] and poem['quality']['qafiya_validation']:
                                    qafiya_val = poem['quality']['qafiya_validation']
                                    valid_baits = qafiya_val.get('valid_baits', 0)
                                    total_baits = qafiya_val.get('total_baits', 0)
                                    quality_info += f"\nQafiya: {valid_baits}/{total_baits} baits correct"
                                    # Add detailed qafiya validation results
                                    if 'bait_results' in qafiya_val and qafiya_val['bait_results']:
                                        quality_info += f"\n  Qafiya details:"
                                        for k, bait_result in enumerate(qafiya_val['bait_results']):
                                            is_valid = bait_result.get('is_valid', False)
                                            error_details = bait_result.get('error_details', '')
                                            quality_info += f"\n    Bait {k+1}: {'✅' if is_valid else '❌'}"
                                            if not is_valid and error_details:
                                                quality_info += f" - {error_details}"
                                    if 'validation_summary' in qafiya_val:
                                        summary = qafiya_val['validation_summary']
                                        if summary:
                                            quality_info += f"\n  Summary: {summary}"
                                
                                # Add overall quality issues if available
                                if 'prosody_issues' in poem['quality'] and poem['quality']['prosody_issues']:
                                    issues = poem['quality']['prosody_issues']
                                    if isinstance(issues, list) and len(issues) > 0:
                                        quality_info += f"\nProsody issues:"
                                        for issue in issues[:3]:  # Show first 3 issues
                                            quality_info += f"\n  - {issue}"
                                
                                if 'qafiya_issues' in poem['quality'] and poem['quality']['qafiya_issues']:
                                    issues = poem['quality']['qafiya_issues']
                                    if isinstance(issues, list) and len(issues) > 0:
                                        quality_info += f"\nQafiya issues:"
                                        for issue in issues[:3]:  # Show first 3 issues
                                            quality_info += f"\n  - {issue}"
                            
                            return f"Qafiya refined through {iterations} iteration(s){quality_info}\n{verses_text}"
                    
                    return f"Qafiya refined through {iterations} iteration(s)"
                else:
                    return "No qafiya refinement needed"
            return "Qafiya refinement completed"
        
        elif component == 'RefinerChain':
            # Based on actual implementation: returns refined poem, refinement metadata
            if outputs:
                refined = outputs.get('refined', False)
                iterations = outputs.get('refinement_iterations', 0)
                refiners_used = outputs.get('refiners_used', [])
                refinement_history = outputs.get('refinement_history', [])
                
                if refined:
                    refiner_names = ', '.join(refiners_used) if isinstance(refiners_used, list) else 'unknown'
                    
                    # Show the final refined poem content with quality information
                    if 'poem' in outputs:
                        poem = outputs['poem']
                        quality_info = ""
                        
                        # Extract quality information
                        if hasattr(poem, 'quality') and poem.quality:
                            if hasattr(poem.quality, 'overall_score'):
                                quality_info = f"\nFinal quality score: {poem.quality.overall_score}"
                                # Add validation details if available
                                if hasattr(poem.quality, 'prosody_validation') and poem.quality.prosody_validation:
                                    prosody_val = poem.quality.prosody_validation
                                    if hasattr(prosody_val, 'overall_valid'):
                                        valid_baits = getattr(prosody_val, 'valid_baits', 0)
                                        total_baits = getattr(prosody_val, 'total_baits', 0)
                                        quality_info += f"\nFinal prosody: {valid_baits}/{total_baits} baits correct"
                                if hasattr(poem.quality, 'qafiya_validation') and poem.quality.qafiya_validation:
                                    qafiya_val = poem.quality.qafiya_validation
                                    if hasattr(qafiya_val, 'overall_valid'):
                                        valid_baits = getattr(qafiya_val, 'valid_baits', 0)
                                        total_baits = getattr(qafiya_val, 'total_baits', 0)
                                        quality_info += f"\nFinal qafiya: {valid_baits}/{total_baits} baits correct"
                        elif isinstance(poem, dict) and 'quality' in poem and poem['quality']:
                            quality_score = poem['quality'].get('overall_score', 'unknown')
                            quality_info = f"\nFinal quality score: {quality_score}"
                            # Add validation details if available
                            if 'prosody_validation' in poem['quality'] and poem['quality']['prosody_validation']:
                                prosody_val = poem['quality']['prosody_validation']
                                valid_baits = prosody_val.get('valid_baits', 0)
                                total_baits = prosody_val.get('total_baits', 0)
                                quality_info += f"\nFinal prosody: {valid_baits}/{total_baits} baits correct"
                            if 'qafiya_validation' in poem['quality'] and poem['quality']['qafiya_validation']:
                                qafiya_val = poem['quality']['qafiya_validation']
                                valid_baits = qafiya_val.get('valid_baits', 0)
                                total_baits = qafiya_val.get('total_baits', 0)
                                quality_info += f"\nFinal qafiya: {valid_baits}/{total_baits} baits correct"
                        
                        # Add detailed refinement iteration details with validation results
                        iteration_details = ""
                        if refinement_history and isinstance(refinement_history, list):
                            iteration_details = f"\nDetailed refinement iterations:"
                            for i, step in enumerate(refinement_history):
                                if hasattr(step, 'refiner_name'):
                                    refiner_name = getattr(step, 'refiner_name', 'unknown')
                                    quality_before = getattr(step, 'quality_before', 'unknown')
                                    quality_after = getattr(step, 'quality_after', 'unknown')
                                    details = getattr(step, 'details', '')
                                    
                                    iteration_details += f"\n  Iteration {i+1}: {refiner_name}"
                                    iteration_details += f"\n    Quality: {quality_before:.3f} → {quality_after:.3f}"
                                    iteration_details += f"\n    Details: {details}"
                                    
                                    # Add validation details from the 'after' poem if available
                                    after_poem = getattr(step, 'after', None)
                                    if after_poem and hasattr(after_poem, 'quality') and after_poem.quality:
                                        if hasattr(after_poem.quality, 'prosody_validation') and after_poem.quality.prosody_validation:
                                            prosody_val = after_poem.quality.prosody_validation
                                            if hasattr(prosody_val, 'overall_valid'):
                                                valid_baits = getattr(prosody_val, 'valid_baits', 0)
                                                total_baits = getattr(prosody_val, 'total_baits', 0)
                                                iteration_details += f"\n    Prosody validation: {valid_baits}/{total_baits} baits correct"
                                                # Add detailed prosody validation results
                                                if hasattr(prosody_val, 'bait_results') and prosody_val.bait_results:
                                                    iteration_details += f"\n      Prosody details:"
                                                    for k, bait_result in enumerate(prosody_val.bait_results):
                                                        is_valid = getattr(bait_result, 'is_valid', False)
                                                        error_details = getattr(bait_result, 'error_details', '')
                                                        pattern = getattr(bait_result, 'pattern', '')
                                                        iteration_details += f"\n        Bait {k+1}: {'✅' if is_valid else '❌'} {pattern}"
                                                        if not is_valid and error_details:
                                                            iteration_details += f" - {error_details}"
                                                if hasattr(prosody_val, 'validation_summary'):
                                                    summary = getattr(prosody_val, 'validation_summary', '')
                                                    if summary:
                                                        iteration_details += f"\n      Summary: {summary}"
                                        
                                        if hasattr(after_poem.quality, 'qafiya_validation') and after_poem.quality.qafiya_validation:
                                            qafiya_val = after_poem.quality.qafiya_validation
                                            if hasattr(qafiya_val, 'overall_valid'):
                                                valid_baits = getattr(qafiya_val, 'valid_baits', 0)
                                                total_baits = getattr(qafiya_val, 'total_baits', 0)
                                                iteration_details += f"\n    Qafiya validation: {valid_baits}/{total_baits} baits correct"
                                                # Add detailed qafiya validation results
                                                if hasattr(qafiya_val, 'bait_results') and qafiya_val.bait_results:
                                                    iteration_details += f"\n      Qafiya details:"
                                                    for k, bait_result in enumerate(qafiya_val.bait_results):
                                                        is_valid = getattr(bait_result, 'is_valid', False)
                                                        error_details = getattr(bait_result, 'error_details', '')
                                                        iteration_details += f"\n        Bait {k+1}: {'✅' if is_valid else '❌'}"
                                                        if not is_valid and error_details:
                                                            iteration_details += f" - {error_details}"
                                                if hasattr(qafiya_val, 'validation_summary'):
                                                    summary = getattr(qafiya_val, 'validation_summary', '')
                                                    if summary:
                                                        iteration_details += f"\n      Summary: {summary}"
                                        
                                        # Add overall quality issues if available
                                        if hasattr(after_poem.quality, 'prosody_issues') and after_poem.quality.prosody_issues:
                                            issues = after_poem.quality.prosody_issues
                                            if isinstance(issues, list) and len(issues) > 0:
                                                iteration_details += f"\n    Prosody issues:"
                                                for issue in issues[:3]:  # Show first 3 issues
                                                    iteration_details += f"\n      - {issue}"
                                        
                                        if hasattr(after_poem.quality, 'qafiya_issues') and after_poem.quality.qafiya_issues:
                                            issues = after_poem.quality.qafiya_issues
                                            if isinstance(issues, list) and len(issues) > 0:
                                                iteration_details += f"\n    Qafiya issues:"
                                                for issue in issues[:3]:  # Show first 3 issues
                                                    iteration_details += f"\n      - {issue}"
                                    
                                    # Add the actual verses that were refined
                                    if after_poem and hasattr(after_poem, 'verses'):
                                        verses = after_poem.verses
                                        if isinstance(verses, list) and len(verses) > 0:
                                            iteration_details += f"\n    Refined verses:"
                                            for j, verse in enumerate(verses):
                                                iteration_details += f"\n      Verse {j+1}: {verse}"
                                    
                                elif isinstance(step, dict):
                                    refiner_name = step.get('refiner_name', 'unknown')
                                    quality_before = step.get('quality_before', 'unknown')
                                    quality_after = step.get('quality_after', 'unknown')
                                    details = step.get('details', '')
                                    
                                    iteration_details += f"\n  Iteration {i+1}: {refiner_name}"
                                    iteration_details += f"\n    Quality: {quality_before:.3f} → {quality_after:.3f}"
                                    iteration_details += f"\n    Details: {details}"
                                    
                                    # Add validation details from the 'after' poem if available
                                    after_poem = step.get('after', {})
                                    if after_poem and 'quality' in after_poem and after_poem['quality']:
                                        if 'prosody_validation' in after_poem['quality'] and after_poem['quality']['prosody_validation']:
                                            prosody_val = after_poem['quality']['prosody_validation']
                                            valid_baits = prosody_val.get('valid_baits', 0)
                                            total_baits = prosody_val.get('total_baits', 0)
                                            iteration_details += f"\n    Prosody validation: {valid_baits}/{total_baits} baits correct"
                                            # Add detailed prosody validation results
                                            if 'bait_results' in prosody_val and prosody_val['bait_results']:
                                                iteration_details += f"\n      Prosody details:"
                                                for k, bait_result in enumerate(prosody_val['bait_results']):
                                                    is_valid = bait_result.get('is_valid', False)
                                                    error_details = bait_result.get('error_details', '')
                                                    pattern = bait_result.get('pattern', '')
                                                    iteration_details += f"\n        Bait {k+1}: {'✅' if is_valid else '❌'} {pattern}"
                                                    if not is_valid and error_details:
                                                        iteration_details += f" - {error_details}"
                                            if 'validation_summary' in prosody_val:
                                                summary = prosody_val['validation_summary']
                                                if summary:
                                                    iteration_details += f"\n      Summary: {summary}"
                                        
                                        if 'qafiya_validation' in after_poem['quality'] and after_poem['quality']['qafiya_validation']:
                                            qafiya_val = after_poem['quality']['qafiya_validation']
                                            valid_baits = qafiya_val.get('valid_baits', 0)
                                            total_baits = qafiya_val.get('total_baits', 0)
                                            iteration_details += f"\n    Qafiya validation: {valid_baits}/{total_baits} baits correct"
                                            # Add detailed qafiya validation results
                                            if 'bait_results' in qafiya_val and qafiya_val['bait_results']:
                                                iteration_details += f"\n      Qafiya details:"
                                                for k, bait_result in enumerate(qafiya_val['bait_results']):
                                                    is_valid = bait_result.get('is_valid', False)
                                                    error_details = bait_result.get('error_details', '')
                                                    iteration_details += f"\n        Bait {k+1}: {'✅' if is_valid else '❌'}"
                                                    if not is_valid and error_details:
                                                        iteration_details += f" - {error_details}"
                                            if 'validation_summary' in qafiya_val:
                                                summary = qafiya_val['validation_summary']
                                                if summary:
                                                    iteration_details += f"\n      Summary: {summary}"
                                        
                                        # Add overall quality issues if available
                                        if 'prosody_issues' in after_poem['quality'] and after_poem['quality']['prosody_issues']:
                                            issues = after_poem['quality']['prosody_issues']
                                            if isinstance(issues, list) and len(issues) > 0:
                                                iteration_details += f"\n    Prosody issues:"
                                                for issue in issues[:3]:  # Show first 3 issues
                                                    iteration_details += f"\n      - {issue}"
                                        
                                        if 'qafiya_issues' in after_poem['quality'] and after_poem['quality']['qafiya_issues']:
                                            issues = after_poem['quality']['qafiya_issues']
                                            if isinstance(issues, list) and len(issues) > 0:
                                                iteration_details += f"\n    Qafiya issues:"
                                                for issue in issues[:3]:  # Show first 3 issues
                                                    iteration_details += f"\n      - {issue}"
                                    
                                    # Add the actual verses that were refined
                                    if after_poem and 'verses' in after_poem:
                                        verses = after_poem['verses']
                                        if isinstance(verses, list) and len(verses) > 0:
                                            iteration_details += f"\n    Refined verses:"
                                            for j, verse in enumerate(verses):
                                                iteration_details += f"\n      Verse {j+1}: {verse}"
                        
                        if hasattr(poem, 'verses'):
                            verses = poem.verses
                            if isinstance(verses, list) and len(verses) > 0:
                                verses_text = "\n".join([f"Final Verse {i+1}: {verse}" for i, verse in enumerate(verses)])
                                return f"Completed {iterations} refinement iterations using {refiner_names}{quality_info}{iteration_details}\nFinal refined poem:\n{verses_text}"
                        elif isinstance(poem, dict) and 'verses' in poem:
                            verses = poem['verses']
                            if isinstance(verses, list) and len(verses) > 0:
                                verses_text = "\n".join([f"Final Verse {i+1}: {verse}" for i, verse in enumerate(verses)])
                                return f"Completed {iterations} refinement iterations using {refiner_names}{quality_info}{iteration_details}\nFinal refined poem:\n{verses_text}"
                    
                    return f"Completed {iterations} refinement iterations using {refiner_names}{quality_info}{iteration_details}"
                else:
                    return "No refinement needed"
            return "Refinement chain completed"
        
        elif component == 'KnowledgeRetriever':
            # Based on actual implementation: returns retrieval results with metadata
            if outputs:
                if 'corpus_results' in outputs:
                    results = outputs['corpus_results']
                    if isinstance(results, list):
                        return f"Retrieved {len(results)} corpus examples"
                elif 'web_results' in outputs:
                    results = outputs['web_results']
                    if isinstance(results, list):
                        return f"Retrieved {len(results)} web results"
                elif 'total_found' in outputs:
                    total = outputs['total_found']
                    return f"Retrieved {total} knowledge items"
            return "Knowledge retrieval completed"
        
        else:
            # Generic case - extract first meaningful output
            if outputs is None:
                return "Processing completed"
            for key, value in outputs.items():
                if key == 'constraints':
                    return "Updated constraints"
                elif key == 'poem':
                    return "Updated poem"
                elif key == 'evaluation':
                    return "Evaluation results"
                elif key == 'refined':
                    return "Refinement completed"
            return "Processing completed"
    
    def _serialize_output(self, output: Any, _depth: int = 0) -> Any:
        """Serialize output for JSON with recursion protection"""
        # Prevent infinite recursion
        if _depth > 10:
            return f"<max_depth_exceeded: {type(output).__name__}>"
        
        try:
            # Handle None values
            if output is None:
                return None
            
            # Handle common non-serializable types
            if hasattr(output, '__class__') and 'threading' in str(output.__class__.__module__):
                return f"<threading_object: {type(output).__name__}>"
            if hasattr(output, '__class__') and 'logging' in str(output.__class__.__module__):
                return f"<logging_object: {type(output).__name__}>"
            
            # Handle Constraints and other custom objects with to_dict method
            if hasattr(output, 'to_dict') and callable(output.to_dict):
                try:
                    result = output.to_dict()
                    return result
                except Exception as e:
                    return f"<to_dict_error: {str(e)}>"
            elif hasattr(output, '__dict__'):
                return {k: self._serialize_output(v, _depth + 1) for k, v in output.__dict__.items() if not k.startswith('_')}
            elif isinstance(output, (list, tuple)):
                return [self._serialize_output(item, _depth + 1) for item in output]
            elif isinstance(output, dict):
                return {k: self._serialize_output(v, _depth + 1) for k, v in output.items()}
            elif hasattr(output, 'value'):  # Handle Enum objects
                return output.value
            else:
                return str(output)  # Convert to string as fallback
        except Exception as e:
            print(f"❌ Serialization error in _serialize_output: {e}")
            return f"<serialization_error: {str(e)}>"
    
    def save_harmony_reasoning(self, reasoning: str, output_file: Path):
        """Save generated reasoning to file"""
        output_file.write_text(reasoning, encoding='utf-8')



