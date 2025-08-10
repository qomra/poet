#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Arabic Poetry Scorer using OpenAI Batch API

This script scores Arabic poems using OpenAI's Batch API for cost efficiency.
Total requests = len(items) * 4 (one for each model + human)

Usage:
    python score.py --input to_score.json --output scored.json --batch_size 50
"""

import argparse
import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import openai
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scoring.log'),
        logging.StreamHandler()
    ]
)

# API Keys - Set your keys here
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

# Batch API constants
MAX_BATCH_SIZE = 50000  # OpenAI batch API limit
MAX_BATCH_ANTHROPIC = 1000  # Anthropic batch API limit
BATCH_POLL_INTERVAL = 30  # seconds between status checks

# Scoring rubric - Single poem evaluation
SINGLE_SYSTEM_PROMPT = """أنت ناقد شعر عربي محترف قادر على تقييم القصائد الكلاسيكية والحديثة.

ستُعرَض عليك قصيدة واحدة لتقييمها حسب سبعة معايير، كل معيار على مقياس عشري من 0 إلى 10 (يمكن استخدام كسور مثل 7.5):

1- الوزن الشعري — مدى التزام الأبيات بالبحر المطلوب وعدم الحيود عنه في أي من الأبيات. يجب أن يكون الوزن موحداً تماماً في كل الأبيات. أي خطأ في الوزن في أي بيت يعتبر انتهاكاً للقاعدة الأساسية للشعر العربي ويستوجب خصم نقاط شديد. يجب أن تكون جميع الأبيات على نفس البحر الشعري دون أي تغيير أو انحراف.
2- القافية — اتساق حرف الروي والحركات (الفتحة والضمة والكسرة والسكون) في ختام الأبيات. يجب أن تكون القافية موحدة تماماً في كل الأبيات من حيث الحرف والحركة. أي اختلاف في الحرف أو الحركة يعتبر خطأ في القافية ويستوجب خصم نقاط شديد. إذا اختلفت القافية في أي بيت تأخذ القصيدة قيمة أقل من 5.
3- المعنى — وضوح المعاني ومدى تحقيقها لمتطلبات الموضوع.
4- الجمالية — جمال الصور والأسلوب والأساليب البلاغية.
5- الإبداع — مدى الابتكار والتجديد مقارنة بالشعر الموروث.
6- التناسق الأسلوبي — انسجام اللغة والمفردات وتماسك البناء.
7- سلامة المفردات — خلو الأبيات من ألفاظ ركيكة أو غير مفهومة في موقع القافية.

أنتج مخرَجًا واحدًا عبارة عن JSON خالص مطابق تمامًا للقالب التالي (دون أي شرح أو نص خارجي):
{
    "meter": 0-10,
    "rhyme": 0-10,
    "meaning": 0-10,
    "beauty": 0-10,
    "creativity": 0-10,
    "consistency": 0-10,
    "vocab": 0-10,
    "total": 0-10
}

يُحسب 'total' كمتوسط الدرجات الست الأخرى (وزن متساوٍ).
أي إخلال بالبنية أو إضافة نص خارج JSON يُعتبَر خطأ."""

# Scoring rubric - Comparative evaluation
COMPARATIVE_SYSTEM_PROMPT = """أنت ناقد شعر عربي محترف قادر على تقييم القصائد الكلاسيكية والحديثة.

ستُعرَض عليك قصيدتان. قيّم كلتا القصيدتين حسب سبعة معايير، كل معيار على مقياس عشري من 0 إلى 10 (يمكن استخدام كسور مثل 7.5):

1- الوزن الشعري — مدى التزام الأبيات بالبحر المطلوب وعدم الحيود عنه في أي من الأبيات. يجب أن يكون الوزن موحداً تماماً في كل الأبيات. أي خطأ في الوزن في أي بيت يعتبر انتهاكاً للقاعدة الأساسية للشعر العربي ويستوجب خصم نقاط شديد. يجب أن تكون جميع الأبيات على نفس البحر الشعري دون أي تغيير أو انحراف.
2- القافية — اتساق حرف الروي والحركات (الفتحة والضمة والكسرة والسكون) في ختام الأبيات. يجب أن تكون القافية موحدة تماماً في كل الأبيات من حيث الحرف والحركة. أي اختلاف في الحرف أو الحركة يعتبر خطأ في القافية ويستوجب خصم نقاط شديد. إذا اختلفت القافية في أي بيت تأخذ القصيدة قيمة أقل من 5.
3- المعنى — وضوح المعاني ومدى تحقيقها لمتطلبات الموضوع.
4- الجمالية — جمال الصور والأسلوب والأساليب البلاغية.
5- الإبداع — مدى الابتكار والتجديد مقارنة بالشعر الموروث.
6- التناسق الأسلوبي — انسجام اللغة والمفردات وتماسك البناء.
7- سلامة المفردات — خلو الأبيات من ألفاظ ركيكة أو غير مفهومة في موقع القافية.

أنتج مخرَجًا واحدًا عبارة عن JSON خالص مطابق تمامًا للقالب التالي (دون أي شرح أو نص خارجي):
{
    "meter": [درجة_القصيدة_الأولى, درجة_القصيدة_الثانية],
    "rhyme": [درجة_القصيدة_الأولى, درجة_القصيدة_الثانية],
    "meaning": [درجة_القصيدة_الأولى, درجة_القصيدة_الثانية],
    "beauty": [درجة_القصيدة_الأولى, درجة_القصيدة_الثانية],
    "creativity": [درجة_القصيدة_الأولى, درجة_القصيدة_الثانية],
    "consistency": [درجة_القصيدة_الأولى, درجة_القصيدة_الثانية],
    "vocab": [درجة_القصيدة_الأولى, درجة_القصيدة_الثانية],
    "total": [درجة_القصيدة_الأولى, درجة_القصيدة_الثانية]
}

يُحسب 'total' كمتوسط الدرجات الست الأخرى (وزن متساوٍ) لكل قصيدة.
أي إخلال بالبنية أو إضافة نص خارج JSON يُعتبَر خطأ."""

class BatchProvider:
    """Base class for batch providers."""
    
    async def submit_batch(self, requests: List[Dict[str, Any]]) -> str:
        """Submit batch and return batch ID."""
        raise NotImplementedError
    
    async def check_batch_status(self, batch_id: str) -> Tuple[str, Dict[str, Any]]:
        """Check batch status and return (status, results)."""
        raise NotImplementedError
    
    async def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: int = BATCH_POLL_INTERVAL,
        max_wait_seconds: Optional[int] = 3600,
    ) -> Dict[str, Any]:
        """Wait for batch completion and return results.
        If max_wait_seconds is None, wait indefinitely.
        """
        logging.info(f"⏱️  Waiting for batch {batch_id}")
        
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            if max_wait_seconds is not None and elapsed > max_wait_seconds:
                logging.warning(f"⏰ Batch {batch_id} exceeded max wait time ({max_wait_seconds} seconds)")
                # Try to get partial results if provider can return any
                status, results = await self.check_batch_status(batch_id)
                if results:
                    logging.info(f"🔄 Returning partial results ({len(results)} items)")
                    return results
                else:
                    raise RuntimeError(f"Batch {batch_id} timed out with no usable results (status={status})")
            
            status, results = await self.check_batch_status(batch_id)
            
            if status in ["completed", "failed", "expired", "cancelled"]:
                if status == "completed" or (status == "failed" and results):
                    if results:
                        logging.info(f"✅ Batch {batch_id} completed with {len(results)} results")
                        return results
                    else:
                        logging.warning(f"⚠️  Batch {batch_id} marked as {status} but no results available; returning empty results")
                        return {}
                else:
                    raise RuntimeError(f"Batch {batch_id} finished with status: {status}")
            
            await asyncio.sleep(poll_interval)

class OpenAIBatchProvider(BatchProvider):
    """OpenAI Batch API provider."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """Initialize the provider with OpenAI client."""
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_batch_size = MAX_BATCH_SIZE

    async def submit_batch(self, requests: List[Dict[str, Any]]) -> str:
        """Submit batch using OpenAI Batch API via JSONL file upload."""
        # Build JSONL content where each line is a request object
        lines: List[str] = []
        for req in requests:
            system_prompt = COMPARATIVE_SYSTEM_PROMPT if req.get("is_comparative") else SINGLE_SYSTEM_PROMPT
            body = {
                "model": self.model,
                "temperature": 1,  # follow current user-adjusted setting
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": req["prompt"]},
                ],
            }
            line_obj = {
                "custom_id": req["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            lines.append(json.dumps(line_obj, ensure_ascii=False))
        jsonl_content = ("\n").join(lines) + "\n"

        # Write to a temporary file for upload
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(jsonl_content.encode("utf-8"))
            tmp.flush()
        try:
            # Upload file with purpose "batch"
            with open(tmp_path, "rb") as f:  # type: ignore
                input_file = await self.client.files.create(file=f, purpose="batch")
            # Create the batch
            batch = await self.client.batches.create(
                input_file_id=input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            logging.info(f"🚀 Submitted OpenAI batch: {batch.id} ({len(requests)} requests)")
            return batch.id
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    async def check_batch_status(self, batch_id: str) -> Tuple[str, Dict[str, Any]]:
        """Check batch status and return (status, results_if_any)."""
        try:
            batch = await self.client.batches.retrieve(batch_id)
        except Exception as e:
            logging.error(f"Failed to retrieve OpenAI batch {batch_id}: {e}")
            return "failed", {}

        status = getattr(batch, "status", None) or getattr(batch, "state", None) or "unknown"
        # Normalize to our naming
        status_map = {
            "validating": "in_progress",
            "queued": "in_progress",
            "running": "in_progress",
            "processing": "in_progress",
            "finalizing": "in_progress",
            "completed": "completed",
            "failed": "failed",
            "expired": "expired",
            "canceled": "cancelled",
            "cancelled": "cancelled",
        }
        norm_status = status_map.get(status, status)

        results: Dict[str, Any] = {}
        if norm_status == "completed":
            output_file_id = getattr(batch, "output_file_id", None)
            if not output_file_id:
                # Some SDKs expose outputs as list
                output_files = getattr(batch, "output_files", None)
                if isinstance(output_files, list) and output_files:
                    output_file_id = getattr(output_files[0], "id", None) or output_files[0]
            if output_file_id:
                try:
                    # Support both coroutine and direct return values
                    raw = self.client.files.content(output_file_id)
                    content_obj = await raw if asyncio.iscoroutine(raw) else raw

                    # Normalize to bytes -> text
                    if isinstance(content_obj, (bytes, bytearray)):
                        text = content_obj.decode("utf-8", errors="ignore")
                    elif hasattr(content_obj, "aread") and callable(getattr(content_obj, "aread")):
                        data = await content_obj.aread()  # type: ignore
                        text = data.decode("utf-8", errors="ignore")
                    elif hasattr(content_obj, "read") and callable(getattr(content_obj, "read")):
                        maybe = content_obj.read()  # type: ignore
                        data = await maybe if asyncio.iscoroutine(maybe) else maybe
                        if isinstance(data, (bytes, bytearray)):
                            text = data.decode("utf-8", errors="ignore")
                        else:
                            text = str(data)
                    else:
                        # Fallback to retrieve_content if available
                        try:
                            raw2 = self.client.files.retrieve_content(output_file_id)
                            data2 = await raw2 if asyncio.iscoroutine(raw2) else raw2
                            if isinstance(data2, (bytes, bytearray)):
                                text = data2.decode("utf-8", errors="ignore")
                            else:
                                text = str(data2)
                        except Exception:
                            text = str(content_obj)

                    for line in text.splitlines():
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            custom_id = obj.get("custom_id")
                            resp = obj.get("response") or {}
                            body = resp.get("body") or {}
                            choices = body.get("choices") or []
                            if not custom_id or not choices:
                                continue
                            message = choices[0].get("message", {})
                            content_text = message.get("content", "{}")
                            scores = json.loads(content_text)
                            results[custom_id] = scores
                        except Exception as e:
                            logging.error(f"Error parsing batch output line: {e}")
                            continue
                except Exception as e:
                    logging.error(f"Failed to download/parse output file {output_file_id}: {e}")
            return norm_status, results

class AnthropicBatchProvider(BatchProvider):
    """Anthropic Message Batches API provider."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514", key: Optional[str] = None):
        """Initialize the provider with Anthropic client."""
        import anthropic
        self.client = anthropic.AsyncAnthropic(api_key=key or ANTHROPIC_API_KEY)
        self.model = model
        self.max_batch_size = MAX_BATCH_ANTHROPIC

    async def submit_batch(self, requests: List[Dict[str, Any]]) -> str:
        """Submit batch to Anthropic Message Batches API."""
        batch_requests = []
        
        for req in requests:
            # Choose system prompt based on evaluation type
            system_prompt = COMPARATIVE_SYSTEM_PROMPT if req.get("is_comparative") else SINGLE_SYSTEM_PROMPT
            
            batch_requests.append({
                "custom_id": req["custom_id"],
                "params": {
                    "model": self.model,
                    "max_tokens": 2048,
                    "temperature": 0.2,
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": req["prompt"]}
                    ]
                }
            })
        
        # Submit batch
        batch = await self.client.beta.messages.batches.create(
            requests=batch_requests
        )
        
        logging.info(f"🚀 Submitted Anthropic batch: {batch.id} ({len(requests)} requests)")
        return batch.id

    async def check_batch_status(self, batch_id: str) -> Tuple[str, Dict[str, Any]]:
        """Check Anthropic batch status and retrieve results if completed."""
        try:
            batch = await self.client.beta.messages.batches.retrieve(batch_id)
        except Exception as e:
            logging.error(f"Failed to retrieve batch {batch_id}: {e}")
            return "failed", {}
        
        # Enhanced status reporting with request counts
        if hasattr(batch, 'request_counts') and batch.request_counts:
            processing = batch.request_counts.processing
            succeeded = batch.request_counts.succeeded  
            errored = batch.request_counts.errored
            canceled = batch.request_counts.canceled
            expired = batch.request_counts.expired
            total = processing + succeeded + errored + canceled + expired
            
            if total > 0:
                completion_rate = succeeded / total
                logging.info(f"📊 Batch {batch_id}: {succeeded}/{total} succeeded ({completion_rate*100:.1f}%), {errored} errored")
                
                if errored > 0:
                    logging.warning(f"⚠️  Batch {batch_id}: {errored} failed requests out of {total} total")
        
        # Try to get results regardless of exact status if processing seems done
        results = {}
        if batch.processing_status in ["ended"] or (hasattr(batch, 'request_counts') and 
                                                   batch.request_counts and 
                                                   batch.request_counts.processing == 0):
            # Retrieve results
            try:
                results_response = await self.client.beta.messages.batches.results(batch_id)
                
                async for result in results_response:
                    custom_id = result.custom_id
                    
                    # Log the result type for debugging
                    logging.info(f"Result {custom_id}: type={result.result.type}")
                    
                    if result.result.type == "error" or result.result.type == "errored":
                        error_msg = getattr(result.result.error, 'message', str(result.result.error))
                        logging.warning(f"Request {custom_id} failed: {error_msg}")
                        # Log the full error object for debugging
                        logging.error(f"Full error for {custom_id}: {result.result.error}")
                        continue
                    
                    try:
                        # Check if it's a successful result with message content
                        if hasattr(result.result, 'message') and result.result.message.content:
                            content = result.result.message.content[0].text
                            # Handle potential markdown formatting
                            if content.startswith("```json"):
                                content = content.strip("```json").strip("```").strip()
                            results[custom_id] = json.loads(content)
                        else:
                            logging.warning(f"Request {custom_id} has no message content, result type: {result.result.type}")
                            # Log more details about the result structure
                            logging.info(f"Result structure for {custom_id}: {dir(result.result)}")
                            continue
                    except (json.JSONDecodeError, IndexError, AttributeError) as e:
                        logging.warning(f"Failed to parse result for {custom_id}: {e}")
                        continue
                
                # If we have good results and high success rate, consider it completed
                if results and hasattr(batch, 'request_counts') and batch.request_counts:
                    total_requests = (batch.request_counts.processing + batch.request_counts.succeeded + 
                                    batch.request_counts.errored + batch.request_counts.canceled + 
                                    batch.request_counts.expired)
                    if total_requests > 0:
                        success_rate = batch.request_counts.succeeded / total_requests
                        if success_rate >= 0.95:
                            logging.info(f"✅ Batch {batch_id}: High success rate ({success_rate*100:.1f}%), returning {len(results)} results")
                            return "completed", results
                
            except Exception as e:
                logging.error(f"Failed to retrieve results for batch {batch_id}: {e}")
                # Don't return failed status if we're just having trouble getting results
                # The batch itself might be fine
                pass
        
        # Map Anthropic status to our standard status
        status_map = {
            "in_progress": "in_progress",
            "canceling": "cancelling", 
            "canceled": "cancelled",
            "ended": "completed"
        }
        
        mapped_status = status_map.get(batch.processing_status, batch.processing_status)
        return mapped_status, results

class BatchPoetryScorer:
    def __init__(self, provider: BatchProvider):
        """Initialize the scorer with a batch provider."""
        self.provider = provider
        
    async def submit_batch(self, requests: List[Dict[str, Any]]) -> str:
        """Submit batch using the provider."""
        return await self.provider.submit_batch(requests)

    async def check_batch_status(self, batch_id: str) -> Tuple[str, Dict[str, Any]]:
        """Check batch status using the provider."""
        return await self.provider.check_batch_status(batch_id)

    async def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: int = BATCH_POLL_INTERVAL,
        max_wait_seconds: Optional[int] = 3600,
    ) -> Dict[str, Any]:
        """Wait for batch completion using the provider."""
        return await self.provider.wait_for_batch(batch_id, poll_interval, max_wait_seconds)

    def _validate_scores(self, scores: Dict[str, Any]) -> bool:
        """Validate that scores are within expected range."""
        required_fields = ["meter", "rhyme", "meaning", "beauty", "creativity", "consistency", "vocab", "total"]
        
        for field in required_fields:
            if field not in scores:
                return False
            try:
                # Handle both single scores and comparative scores (arrays)
                if isinstance(scores[field], list):
                    # Comparative format: [score1, score2]
                    if len(scores[field]) != 2:
                        return False
                    for score in scores[field]:
                        score_val = float(score)
                        if not (0 <= score_val <= 10):
                            return False
                else:
                    # Single format: single score
                    score = float(scores[field])
                    if not (0 <= score <= 10):
                        return False
            except (ValueError, TypeError):
                return False
        
        return True

    async def score_batch(
        self,
        items: List[Dict[str, Any]],
        batch_size: int = None,
        parallel_batches: int = 1,
        wait_forever: bool = False,
    ) -> List[Dict[str, Any]]:
        """Score a batch of items using the batch provider.
        Submits multiple provider batches and processes them as they complete.
        """
        if batch_size is None:
            batch_size = self.provider.max_batch_size
        
        all_results: List[Dict[str, Any]] = []
        
        # Prepare all requests
        all_requests: List[Dict[str, Any]] = []
        for item in items:
            poem_id = item["poem_id"]
            prompt = item["prompt"]
            
            # Create requests for each model and human poem
            sources = ["gemini-2.5-pro", "claude-sonnet-4-20250514", "gpt-5-2025-08-07", "human"]
            
            for source in sources:
                if source in item:
                    poem_text = item[source]
                    
                    # Single poem evaluation (for all sources)
                    single_user_prompt = f"""المطلب:
{prompt}

القصيدة:
{poem_text}

قيّم هذه القصيدة حسب المعايير المذكورة."""

                    custom_id_single = f"{poem_id}_{self._normalize_source(source)}_single".replace("-", "_").replace(".", "_")
                    all_requests.append({
                        "custom_id": custom_id_single,
                        "prompt": single_user_prompt,
                        "poem_id": poem_id,
                        "source": source,
                        "is_comparative": False
                    })
                    
                    # Comparative evaluation (only for AI sources, alongside human)
                    if source != "human" and "human" in item:
                        human_poem = item["human"]
                        
                        # Alternate which poem comes first to avoid bias
                        ai_first = hash(f"{poem_id}_{source}") % 2 == 0
                        
                        if ai_first:
                            poem1 = poem_text
                            poem2 = human_poem
                            poem1_label = "القصيدة الأولى"
                            poem2_label = "القصيدة الثانية"
                        else:
                            poem1 = human_poem
                            poem2 = poem_text
                            poem1_label = "القصيدة الأولى"
                            poem2_label = "القصيدة الثانية"
                        
                        comparative_user_prompt = f"""المطلب:
{prompt}

{poem1_label}:
{poem1}

{poem2_label}:
{poem2}

قيّم كلتا القصيدتين حسب المعايير المذكورة."""

                        custom_id_comparative = f"{poem_id}_{self._normalize_source(source)}_comparative".replace("-", "_").replace(".", "_")
                        all_requests.append({
                            "custom_id": custom_id_comparative,
                            "prompt": comparative_user_prompt,
                            "poem_id": poem_id,
                            "source": source,
                            "is_comparative": True,
                            "ai_first": ai_first
                        })
        
        logging.info(f"Prepared {len(all_requests)} requests for scoring")
        
        # Chunk requests
        chunks: List[List[Dict[str, Any]]] = [
            all_requests[i:i + batch_size] for i in range(0, len(all_requests), batch_size)
        ]
        total_batches = len(chunks)
        
        # Helper to process results for a chunk
        def process_results_for_chunk(
            batch_requests: List[Dict[str, Any]],
            results_dict: Dict[str, Any],
            batch_id: str,
        ) -> List[Dict[str, Any]]:
            batch_results: List[Dict[str, Any]] = []
            for req in batch_requests:
                custom_id = req["custom_id"]
                if custom_id not in results_dict:
                    logging.warning(f"No result for custom_id: {custom_id}")
                    continue
                try:
                    scores = results_dict[custom_id]
                    if not self._validate_scores(scores):
                        logging.warning(f"Invalid scores for {custom_id}: {scores}")
                        continue
                    result = {
                        "poem_id": req["poem_id"],
                        "source": req["source"],
                        "scores": scores,
                        "is_comparative": req.get("is_comparative", False),
                        "batch_id": batch_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    if req.get("is_comparative", False):
                        ai_first = req.get("ai_first", False)
                        ai_scores: Dict[str, Any] = {}
                        human_scores: Dict[str, Any] = {}
                        for field, values in scores.items():
                            if isinstance(values, list) and len(values) == 2:
                                if ai_first:
                                    ai_scores[field] = values[0]
                                    human_scores[field] = values[1]
                                else:
                                    ai_scores[field] = values[1]
                                    human_scores[field] = values[0]
                        ai_result = {
                            "poem_id": req["poem_id"],
                            "source": f"{req['source']}_vs_human",
                            "scores": ai_scores,
                            "is_comparative": False,
                            "batch_id": batch_id,
                            "timestamp": datetime.now().isoformat()
                        }
                        human_result = {
                            "poem_id": req["poem_id"],
                            "source": f"human_vs_{req['source']}",
                            "scores": human_scores,
                            "is_comparative": False,
                            "batch_id": batch_id,
                            "timestamp": datetime.now().isoformat()
                        }
                        batch_results.append(ai_result)
                        batch_results.append(human_result)
                    else:
                        batch_results.append(result)
                except Exception as e:
                    logging.error(f"Error processing result for {custom_id}: {e}")
                    continue
            return batch_results
        
        # Submit batches with limited parallelism
        semaphore = asyncio.Semaphore(parallel_batches if parallel_batches > 0 else 1)
        submitted: List[Tuple[str, int]] = []  # (batch_id, chunk_index)
        
        async def submit_one(i: int, chunk: List[Dict[str, Any]]) -> Tuple[str, int, List[Dict[str, Any]]]:
            async with semaphore:
                logging.info(f"Submitting batch {i+1}/{total_batches} ({len(chunk)} requests)")
                batch_id = await self.submit_batch(chunk)
                return batch_id, i, chunk
        
        submit_tasks = [submit_one(i, chunk) for i, chunk in enumerate(chunks)]
        submissions = await asyncio.gather(*submit_tasks)
        
        # Map for later processing
        batch_map: Dict[str, Tuple[int, List[Dict[str, Any]]]] = {bid: (i, chunk) for (bid, i, chunk) in submissions}
        
        # Wait for completion as they finish
        wait_tasks = []
        max_wait = None if wait_forever else 3600
        async def wait_with_id(bid: str):
            res = await self.wait_for_batch(bid, BATCH_POLL_INTERVAL, max_wait)
            return bid, res
        for batch_id in list(batch_map.keys()):
            wait_tasks.append(asyncio.create_task(wait_with_id(batch_id)))
        
        completed_count = 0
        for task in asyncio.as_completed(wait_tasks):
            try:
                finished_batch_id, results_dict = await task
            except Exception as e:
                logging.error(f"A batch failed while waiting: {e}")
                completed_count += 1
                continue
            # Map to original chunk via batch_id
            if finished_batch_id not in batch_map:
                logging.warning(f"Finished unknown batch_id {finished_batch_id}")
                continue
            idx, chunk = batch_map.pop(finished_batch_id)
            batch_results = process_results_for_chunk(chunk, results_dict, finished_batch_id)
            all_results.extend(batch_results)
            logging.info(f"Completed batch {idx+1}/{total_batches} with {len(batch_results)} valid results")
            if batch_results:
                self._save_results(all_results, "scored_intermediate.json")
            completed_count += 1
        
        return all_results
    
    def _save_results(self, results: List[Dict[str, Any]], filename: str):
        """Save results to JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved {len(results)} results to {filename}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")

    def _normalize_source(self, s: str) -> str:
        return s.replace("-", "_").replace(".", "_")

    def _denormalize_source(self, normalized: str) -> str:
        # Map back to a known source label by normalization match
        known = ["gemini-2.5-pro", "claude-sonnet-4-20250514", "gpt-5-2025-08-07", "human"]
        for k in known:
            if self._normalize_source(k) == normalized:
                return k
        return normalized  # fallback

    def _process_results_from_custom_ids(
        self,
        results_dict: Dict[str, Any],
        items: List[Dict[str, Any]],
        batch_id: str,
    ) -> List[Dict[str, Any]]:
        # Build quick item map by poem_id
        item_map: Dict[str, Dict[str, Any]] = {}
        for it in items:
            item_map[str(it.get("poem_id"))] = it
        
        collected: List[Dict[str, Any]] = []
        for custom_id, scores in results_dict.items():
            try:
                # custom_id format: {poem_id}_{source_norm}_{single|comparative}
                parts = custom_id.split("_")
                if len(parts) < 3:
                    logging.warning(f"Unexpected custom_id format: {custom_id}")
                    continue
                poem_id = parts[0]
                result_type = parts[-1]
                source_norm = "_".join(parts[1:-1])
                source = self._denormalize_source(source_norm)
                item = item_map.get(str(poem_id))
                if not item:
                    logging.warning(f"No item found for poem_id {poem_id} (custom_id {custom_id})")
                    continue
                
                if not self._validate_scores(scores):
                    logging.warning(f"Invalid scores for {custom_id}: {scores}")
                    continue
                
                if result_type == "comparative":
                    ai_first = (hash(f"{poem_id}_{source}") % 2 == 0)
                    ai_scores: Dict[str, Any] = {}
                    human_scores: Dict[str, Any] = {}
                    for field, values in scores.items():
                        if isinstance(values, list) and len(values) == 2:
                            if ai_first:
                                ai_scores[field] = values[0]
                                human_scores[field] = values[1]
                            else:
                                ai_scores[field] = values[1]
                                human_scores[field] = values[0]
                    ai_result = {
                        "poem_id": item["poem_id"],
                        "source": f"{source}_vs_human",
                        "scores": ai_scores,
                        "is_comparative": False,
                        "batch_id": batch_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    human_result = {
                        "poem_id": item["poem_id"],
                        "source": f"human_vs_{source}",
                        "scores": human_scores,
                        "is_comparative": False,
                        "batch_id": batch_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    collected.append(ai_result)
                    collected.append(human_result)
                else:
                    collected.append({
                        "poem_id": item["poem_id"],
                        "source": source,
                        "scores": scores,
                        "is_comparative": False,
                        "batch_id": batch_id,
                        "timestamp": datetime.now().isoformat()
                    })
            except Exception as e:
                logging.error(f"Error processing custom_id {custom_id}: {e}")
                continue
        return collected

    async def resume_polling(
        self,
        batch_ids: List[str],
        items: List[Dict[str, Any]],
        poll_interval: int = BATCH_POLL_INTERVAL,
        wait_forever: bool = True,
    ) -> List[Dict[str, Any]]:
        pending: Dict[str, None] = {bid: None for bid in batch_ids}
        all_results: List[Dict[str, Any]] = []
        start_time = time.time()
        while pending:
            completed_this_round: List[str] = []
            for bid in list(pending.keys()):
                try:
                    status, results_dict = await self.provider.check_batch_status(bid)
                except Exception as e:
                    logging.error(f"Failed to check status for {bid}: {e}")
                    continue
                if status == "completed" or (status == "failed" and results_dict):
                    batch_results = self._process_results_from_custom_ids(results_dict, items, bid)
                    all_results.extend(batch_results)
                    logging.info(f"Batch {bid} finished with {len(batch_results)} results; {len(pending)-1} remaining")
                    if batch_results:
                        self._save_results(all_results, "scored_intermediate.json")
                    completed_this_round.append(bid)
            for bid in completed_this_round:
                pending.pop(bid, None)
            if not pending:
                break
            if not wait_forever:
                # If not waiting forever, stop after one sweep without completions
                if not completed_this_round:
                    logging.info("No completed batches this cycle and not waiting forever; stopping resume loop")
                    break
            # Sleep before next poll
            await asyncio.sleep(poll_interval)
        elapsed = time.time() - start_time
        logging.info(f"Resume polling finished in {elapsed:.2f}s; total results: {len(all_results)}")
        return all_results

def load_data(input_file: str) -> List[Dict[str, Any]]:
    """Load data from JSON file."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading data from {input_file}: {e}")
        raise

def save_final_results(results: List[Dict[str, Any]], output_file: str):
    """Save final results to JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info(f"Final results saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving final results: {e}")
        raise

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Score Arabic poems using OpenAI Batch API")
    parser.add_argument("--input", required=True, help="Input JSON file with poems")
    parser.add_argument("--output", default="scored.json", help="Output JSON file for results")
    parser.add_argument("--batch_size", type=int, help="Batch size for API requests (default: provider max)")
    parser.add_argument("--model", help="Model to use (OpenAI or Anthropic)")
    parser.add_argument("--api_key", help="API key (optional, uses environment variable if not provided)")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="anthropic", help="Batch API provider to use")
    parser.add_argument("--parallel_batches", type=int, default=4, help="Number of provider batches to submit in parallel")
    parser.add_argument("--no_timeout", action="store_true", help="Wait indefinitely for batches to complete")
    parser.add_argument("--poll_interval", type=int, default=60, help="Polling interval (seconds) for resume mode")
    parser.add_argument("--resume_batch_file", help="Path to file containing batch IDs to resume, one per line")
    parser.add_argument("--resume_only", action="store_true", help="Only resume existing batches; do not submit new ones")
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        logging.error(f"Input file {args.input} does not exist")
        return
    
    # Load data
    logging.info(f"Loading data from {args.input}")
    items = load_data(args.input)
    logging.info(f"Loaded {len(items)} items")
    
    # Initialize scorer with appropriate provider
    if args.provider == "anthropic":
        model = args.model or "claude-sonnet-4-20250514"
        provider = AnthropicBatchProvider(model, args.api_key or ANTHROPIC_API_KEY)
    else:  # Default to OpenAI
        model = args.model or "gpt-5-2025-08-07"
        provider = OpenAIBatchProvider(args.api_key or OPENAI_API_KEY, model)
    
    scorer = BatchPoetryScorer(provider)

    # Resume mode
    if args.resume_batch_file:
        path = Path(args.resume_batch_file)
        if not path.exists():
            logging.error(f"Resume batch file not found: {path}")
            return
        raw_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        # Filter valid batch ids
        batch_ids = [bid for bid in raw_ids if bid.startswith("batch_")]
        invalid = [bid for bid in raw_ids if bid not in batch_ids]
        if invalid:
            logging.warning(f"Ignoring {len(invalid)} invalid batch id lines")
        logging.info(f"Resuming {len(batch_ids)} batches from {path}")
        start_time = time.time()
        results = await scorer.resume_polling(batch_ids, items, poll_interval=args.poll_interval, wait_forever=True)
        end_time = time.time()
        save_final_results(results, args.output)
        successful_scores = len(results)
        single_evaluations = sum(1 for r in results if not any(s in r["source"] for s in ["_vs_", "human_vs_"]))
        comparative_evaluations = sum(1 for r in results if any(s in r["source"] for s in ["_vs_", "human_vs_"]))
        logging.info(f"Scoring completed!")
        logging.info(f"Successful scores: {successful_scores}")
        logging.info(f"Single evaluations: {single_evaluations}")
        logging.info(f"Comparative evaluations: {comparative_evaluations}")
        logging.info(f"Time taken: {end_time - start_time:.2f} seconds")
        logging.info(f"Results saved to: {args.output}")
        if args.resume_only:
            return
    
    if args.resume_only:
        logging.info("--resume_only set but no --resume_batch_file provided; nothing to do.")
        return
    
    # Validate batch size based on provider
    max_batch_size = MAX_BATCH_ANTHROPIC if args.provider == "anthropic" else MAX_BATCH_SIZE
    if args.batch_size and args.batch_size > max_batch_size:
        logging.warning(f"Batch size {args.batch_size} exceeds maximum {max_batch_size} for {args.provider}, using {max_batch_size}")
        args.batch_size = max_batch_size
    
    # Calculate total requests
    # Each item has 4 sources, each AI source gets scored twice (single + comparative)
    # Human gets scored once (single only)
    total_requests = 0
    for item in items:
        sources = ["gemini-2.5-pro", "claude-sonnet-4-20250514", "gpt-5-2025-08-07", "human"]
        for source in sources:
            if source in item:
                if source == "human":
                    total_requests += 1  # Human gets single evaluation only
                else:
                    total_requests += 2  # AI sources get both single and comparative
    
    logging.info(f"Total scoring requests: {total_requests}")
    logging.info(f"Using provider: {args.provider}")
    logging.info(f"Using batch size: {args.batch_size or 'default'}")
    logging.info(f"Parallel provider batches: {args.parallel_batches}")
    logging.info(f"No-timeout mode: {'on' if args.no_timeout else 'off'}")
    
    # Score poems (fresh submission)
    start_time = time.time()
    results = await scorer.score_batch(
        items,
        args.batch_size,
        parallel_batches=args.parallel_batches,
        wait_forever=args.no_timeout,
    )
    end_time = time.time()
    
    # Save final results
    save_final_results(results, args.output)
    
    # Summary
    successful_scores = len(results)
    
    # Count by source type
    single_evaluations = sum(1 for r in results if not any(s in r["source"] for s in ["_vs_", "human_vs_"]))
    comparative_evaluations = sum(1 for r in results if any(s in r["source"] for s in ["_vs_", "human_vs_"]))
    
    logging.info(f"Scoring completed!")
    logging.info(f"Successful scores: {successful_scores}/{total_requests}")
    logging.info(f"Single evaluations: {single_evaluations}")
    logging.info(f"Comparative evaluations: {comparative_evaluations}")
    logging.info(f"Time taken: {end_time - start_time:.2f} seconds")
    logging.info(f"Results saved to: {args.output}")

if __name__ == "__main__":
    asyncio.run(main())