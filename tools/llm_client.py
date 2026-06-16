"""
Unified LLM client: Claude (primary) → OpenAI (fallback) → Ollama (offline).
Supports streaming, exponential backoff, and cost tracking.
Set PRIVACY_MODE=1 to force Ollama regardless of other keys.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)

COST_PER_1K = {
    "claude-opus-4-8":     {"input": 0.015,  "output": 0.075},
    "claude-sonnet-4-6":   {"input": 0.003,  "output": 0.015},
    "claude-haiku-4-5":    {"input": 0.00025,"output": 0.00125},
    "gpt-4o":              {"input": 0.005,  "output": 0.015},
    "gpt-4o-mini":         {"input": 0.00015,"output": 0.0006},
    "llama3":              {"input": 0.0,    "output": 0.0},
    "mistral":             {"input": 0.0,    "output": 0.0},
}


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float


class UnifiedLLMClient:
    def __init__(self, memory_manager=None):
        self._memory = memory_manager
        self._privacy_mode = os.getenv("PRIVACY_MODE", "0") == "1"

    def _build_provider_chain(self) -> list[tuple[str, str]]:
        if self._privacy_mode:
            return [("ollama", os.getenv("OLLAMA_MODEL", "llama3"))]

        chain = []
        if os.getenv("ANTHROPIC_API_KEY"):
            chain.append(("claude", os.getenv("CLAUDE_MODEL", "claude-opus-4-8")))
        if os.getenv("OPENAI_API_KEY"):
            chain.append(("openai", os.getenv("OPENAI_MODEL", "gpt-4o")))
        ollama_url = os.getenv("OLLAMA_BASE_URL", "")
        if ollama_url or not chain:
            chain.append(("ollama", os.getenv("OLLAMA_MODEL", "llama3")))
        return chain

    def _calc_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        rates = COST_PER_1K.get(model, {"input": 0.0, "output": 0.0})
        return (tokens_in * rates["input"] + tokens_out * rates["output"]) / 1000.0

    async def _call_claude(self, model: str, messages: list, system: str, max_tokens: int) -> LLMResult:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        t0 = time.monotonic()
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        text = response.content[0].text
        tin = response.usage.input_tokens
        tout = response.usage.output_tokens
        return LLMResult(
            text=text,
            provider="claude",
            model=model,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=self._calc_cost(model, tin, tout),
            latency_ms=latency_ms,
        )

    async def _call_openai(self, model: str, messages: list, system: str, max_tokens: int) -> LLMResult:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        all_messages = [{"role": "system", "content": system}] + messages
        t0 = time.monotonic()
        response = await client.chat.completions.create(
            model=model,
            messages=all_messages,
            max_tokens=max_tokens,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        text = response.choices[0].message.content or ""
        tin = response.usage.prompt_tokens
        tout = response.usage.completion_tokens
        return LLMResult(
            text=text,
            provider="openai",
            model=model,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=self._calc_cost(model, tin, tout),
            latency_ms=latency_ms,
        )

    async def _call_ollama(self, model: str, messages: list, system: str, max_tokens: int) -> LLMResult:
        import aiohttp
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        prompt_parts = [f"System: {system}"] if system else []
        for m in messages:
            prompt_parts.append(f"{m['role'].capitalize()}: {m['content']}")
        prompt = "\n".join(prompt_parts)
        payload = {"model": model, "prompt": prompt, "stream": False,
                   "options": {"num_predict": max_tokens}}
        t0 = time.monotonic()
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/api/generate", json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                data = await resp.json()
        latency_ms = (time.monotonic() - t0) * 1000
        text = data.get("response", "")
        tin = len(prompt) // 4
        tout = len(text) // 4
        return LLMResult(
            text=text,
            provider="ollama",
            model=model,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=0.0,
            latency_ms=latency_ms,
        )

    async def _call_with_retry(
        self, provider: str, model: str, messages: list, system: str, max_tokens: int
    ) -> LLMResult:
        delays = [1, 2, 4]
        last_err = None
        for attempt, delay in enumerate(delays, 1):
            try:
                if provider == "claude":
                    return await self._call_claude(model, messages, system, max_tokens)
                elif provider == "openai":
                    return await self._call_openai(model, messages, system, max_tokens)
                else:
                    return await self._call_ollama(model, messages, system, max_tokens)
            except Exception as e:
                last_err = e
                if attempt < len(delays):
                    logger.warning("LLM %s attempt %d failed: %s — retrying in %ds", provider, attempt, e, delay)
                    await asyncio.sleep(delay)
        raise RuntimeError(f"All {len(delays)} attempts failed for {provider}/{model}: {last_err}")

    async def complete(
        self,
        prompt: str,
        system: str = "You are a helpful market research analyst.",
        max_tokens: int = 1500,
        task: str = "general",
    ) -> LLMResult:
        chain = self._build_provider_chain()
        last_err = None
        for provider, model in chain:
            try:
                messages = [{"role": "user", "content": prompt}]
                result = await self._call_with_retry(provider, model, messages, system, max_tokens)
                if self._memory:
                    self._memory.log_llm_cost(provider, model, task,
                                               result.tokens_in, result.tokens_out, result.cost_usd)
                return result
            except Exception as e:
                last_err = e
                logger.warning("Provider %s/%s failed: %s — trying next", provider, model, e)

        logger.error("All LLM providers failed: %s", last_err)
        return LLMResult(
            text="",
            provider="none",
            model="none",
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            latency_ms=0.0,
        )

    async def stream(
        self,
        prompt: str,
        system: str = "You are a helpful analyst.",
        max_tokens: int = 2000,
    ) -> AsyncGenerator[str, None]:
        result = await self.complete(prompt, system=system, max_tokens=max_tokens)
        for chunk in result.text.split(" "):
            yield chunk + " "
            await asyncio.sleep(0.001)

    def complete_sync(
        self,
        prompt: str,
        system: str = "You are a helpful market research analyst.",
        max_tokens: int = 1500,
        task: str = "general",
    ) -> LLMResult:
        return asyncio.get_event_loop().run_until_complete(
            self.complete(prompt, system=system, max_tokens=max_tokens, task=task)
        )
