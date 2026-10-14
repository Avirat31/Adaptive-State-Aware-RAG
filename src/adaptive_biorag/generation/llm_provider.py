"""OpenAI-compatible chat-completions client (hosted APIs or local servers: Ollama, vLLM, llama.cpp, LM Studio).

Uses only the standard library. API key is read from an environment variable and never logged. External calls happen
ONLY when generation.provider=llm or decomposition.method=llm is configured.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..logging_utils import get_logger
from ..schemas import RetrievalResult
from .base import AnswerGenerator, GenerationOutput
from .citation_validation import validate_citations
from .prompts import GROUNDED_SYSTEM, build_user_prompt

log = get_logger("llm")


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMClientBase(ABC):
    model = "unknown"

    def __init__(self):
        self.calls: list[dict] = []
        self.in_tokens: int | None = 0
        self.out_tokens: int | None = 0
        self.cost: float | None = 0.0
        self.n_calls = 0

    @abstractmethod
    def _complete(self, system: str, user: str, json_mode: bool) -> LLMResponse: ...

    def complete(self, system: str, user: str, purpose: str = "", json_mode: bool = False) -> LLMResponse:
        resp = self._complete(system, user, json_mode)
        self.n_calls += 1
        self.in_tokens = None if (self.in_tokens is None or resp.input_tokens is None) else self.in_tokens + resp.input_tokens
        self.out_tokens = None if (self.out_tokens is None or resp.output_tokens is None) else self.out_tokens + resp.output_tokens
        self.calls.append({"purpose": purpose, "model": resp.model, "system": system, "user": user,
                           "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens})
        return resp

    def snapshot(self) -> dict:
        return {"calls": self.n_calls, "in": self.in_tokens, "out": self.out_tokens, "cost": self.cost}


class OpenAICompatibleClient(LLMClientBase):
    def __init__(self, cfg: dict):
        super().__init__()
        l = cfg["llm"]
        self.model, self.timeout, self.retries = l["model"], l["timeout_seconds"], l["max_retries"]
        self.temp, self.max_tokens = l["temperature"], l["max_output_tokens"]
        self.base_url = (l["base_url"] or os.environ.get("ADAPTIVE_BIORAG_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.key_env = l["api_key_env"]
        self.p_in, self.p_out = l["price_per_1k_input"], l["price_per_1k_output"]

    def _complete(self, system, user, json_mode) -> LLMResponse:
        key = os.environ.get(self.key_env, "")
        if not key and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            raise LLMError(f"Environment variable {self.key_env} is not set (see .env.example). No external call was made.")
        body = {"model": self.model, "temperature": self.temp, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.loads(r.read().decode())
                u = d.get("usage") or {}
                it, ot = u.get("prompt_tokens"), u.get("completion_tokens")
                if self.p_in is not None and self.p_out is not None and it is not None and ot is not None and self.cost is not None:
                    self.cost += it / 1000 * self.p_in + ot / 1000 * self.p_out
                elif self.cost is not None and (self.p_in is None or self.p_out is None or it is None):
                    self.cost = None                          # cannot be measured -> unavailable, never guessed
                return LLMResponse(d["choices"][0]["message"]["content"] or "", d.get("model", self.model), it, ot)
            except urllib.error.HTTPError as e:
                last = LLMError(f"HTTP {e.code} from LLM endpoint")
                if e.code not in (408, 429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last = LLMError(f"LLM request failed: {type(e).__name__}")
            time.sleep(min(2 ** attempt, 8))
        raise last or LLMError("LLM request failed")


def parse_json_object(text: str) -> dict:
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            raise ValueError("no JSON object in model output")
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("model output JSON is not an object")
    return obj


class LLMGenerator(AnswerGenerator):
    name = "llm"

    def __init__(self, client: LLMClientBase):
        self.client = client

    def generate(self, question, evidence: list[RetrievalResult], state, options=None) -> GenerationOutput:
        conflicts = state.state.conflicting_claims if state is not None else []
        user = build_user_prompt(question, evidence, state.summary() if state is not None else "", options, conflicts)
        try:
            raw = self.client.complete(GROUNDED_SYSTEM, user, purpose="answer", json_mode=True)
            d = parse_json_object(raw.text)
            out = GenerationOutput(
                answer=str(d.get("answer", "")),
                citations=[c for c in d.get("citations", []) if isinstance(c, dict)],
                limitations=[str(x) for x in d.get("limitations", [])], uncertainties=[str(x) for x in d.get("uncertainties", [])],
                insufficient_evidence=bool(d.get("insufficient_evidence", False)),
                label=(str(d["label"]).strip() if d.get("label") not in (None, "") else None), generator=f"llm:{raw.model}")
        except (LLMError, ValueError) as e:
            out = GenerationOutput(answer="Answer generation failed; no answer produced.", insufficient_evidence=True,
                                   generator="llm", error=str(e), limitations=["generation_failure"])
        return validate_citations(out, evidence)
