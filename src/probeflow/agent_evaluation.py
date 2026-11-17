"""LLM-judge harness compared with a small, explicitly synthetic label set."""
from __future__ import annotations

import json
import os
import time
from statistics import mean
from typing import Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field


class JudgeDecision(BaseModel):
    grounded: bool
    quality_score: int = Field(ge=1, le=5)
    rationale: str = Field(min_length=1, max_length=500)


class Judge(Protocol):
    name: str

    def evaluate(self, question: str, evidence: str, answer: str) -> JudgeDecision: ...


class OllamaJudge:
    """Local-only LLM judge with one schema-validation retry."""

    def __init__(self, model: str, url: str = "http://127.0.0.1:11434", timeout: float = 60):
        if not model.strip():
            raise ValueError("A judge model is required")
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("The portfolio judge accepts only a loopback Ollama endpoint")
        self.model = model.strip()
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.name = "ollama:" + self.model
        self.last_usage: dict[str, int | float] = {}

    def evaluate(self, question: str, evidence: str, answer: str) -> JudgeDecision:
        system = ("Judge a career-agent answer using only the supplied question and job evidence. "
                  "Grounded is true only if every factual job claim is supported by the evidence. "
                  "Quality is 1 (poor) to 5 (direct, accurate, useful, and complete). Do not reward "
                  "confidence or invented details. Return only the required JSON object.")
        validation_error = ""
        started = time.perf_counter()
        for attempt in range(2):
            user = {"question": question[:2000], "evidence": evidence[:8000], "answer": answer[:4000]}
            if validation_error:
                user["validation_error"] = validation_error
                user["instruction"] = "Retry with a valid object matching the JSON schema."
            payload = {"model": self.model, "stream": False, "format": JudgeDecision.model_json_schema(),
                       "options": {"temperature": 0, "num_predict": 220},
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]}
            try:
                with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                    response = client.post(self.url + "/api/chat", json=payload)
                    response.raise_for_status()
                    body = response.json()
                self.last_usage = {"prompt_tokens": int(body.get("prompt_eval_count", 0)),
                                   "completion_tokens": int(body.get("eval_count", 0)),
                                   "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
                return JudgeDecision.model_validate_json(body["message"]["content"])
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                validation_error = type(exc).__name__
        raise RuntimeError("LLM judge failed schema validation after one retry")


def configured_judge() -> OllamaJudge | None:
    model = os.getenv("PROBEFLOW_JUDGE_MODEL", "").strip()
    if not model:
        return None
    return OllamaJudge(model, os.getenv("PROBEFLOW_OLLAMA_URL", "http://127.0.0.1:11434"))


def evaluate_agent_answers(cases: list[dict], judge: Judge) -> dict:
    if not cases:
        raise ValueError("At least one labeled judge example is required")
    grounded_correct: list[float] = []
    quality_errors: list[float] = []
    quality_exact: list[float] = []
    detail_rows = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "judge_latency_ms": 0.0}
    for case in cases:
        result = judge.evaluate(case["question"], case["evidence"], case["answer"])
        expected_grounded = bool(case["label_grounded"])
        expected_quality = int(case["label_quality"])
        grounded_correct.append(float(result.grounded == expected_grounded))
        quality_errors.append(abs(result.quality_score - expected_quality))
        quality_exact.append(float(result.quality_score == expected_quality))
        detail_rows.append({"id": case["id"], "label_grounded": expected_grounded,
                            "judge_grounded": result.grounded, "label_quality": expected_quality,
                            "judge_quality": result.quality_score, "rationale": result.rationale})
        raw_usage = getattr(judge, "last_usage", {}) or {}
        for key in ("prompt_tokens", "completion_tokens"):
            usage[key] += int(raw_usage.get(key, 0))
        usage["judge_latency_ms"] = round(usage["judge_latency_ms"] + float(raw_usage.get("latency_ms", 0)), 2)
    return {"judge": judge.name, "examples": len(cases),
            "label_source": "author-curated synthetic examples, not user interaction data",
            "groundedness_accuracy": mean(grounded_correct),
            "quality_exact_agreement": mean(quality_exact),
            "quality_mean_absolute_error": mean(quality_errors),
            "usage": usage, "details": detail_rows}

