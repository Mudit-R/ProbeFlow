from __future__ import annotations

import pytest
from pydantic import ValidationError

from probeflow.resume_profile import (
    OllamaResumeExtractor,
    ResumeProfile,
    ResumeProfileError,
    configured_resume_extractor,
    extract_resume_profile,
)


VALID_PROFILE = {
    "summary": "Final-year engineering student with backend internship experience.",
    "target_roles": ["Backend Engineer Intern", "Software Engineer Intern"],
    "skills": ["Python", "FastAPI", "SQL"],
    "preferred_locations": ["India", "Remote"],
    "remote_ok": True,
    "experience": [
        {
            "role": "Software Engineering Intern",
            "organization": "Example Systems",
            "date_range": "Jun 2026 to Aug 2026",
            "summary": "Built and tested backend API integrations.",
            "skills": ["Java", "REST APIs"],
        }
    ],
}


class StubExtractor:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def extract(self, resume_text, *, validation_feedback=None):
        self.calls.append((resume_text, validation_feedback))
        return self.outputs.pop(0)


def test_valid_extraction_returns_candidate_profile_compatible_fields():
    extractor = StubExtractor([VALID_PROFILE])
    profile = extract_resume_profile("Resume text with private contact data.", extractor)

    assert profile.skills == ["Python", "FastAPI", "SQL"]
    assert profile.experience[0].organization == "Example Systems"
    assert profile.to_candidate_profile().model_dump() == {
        "summary": VALID_PROFILE["summary"],
        "target_roles": VALID_PROFILE["target_roles"],
        "skills": VALID_PROFILE["skills"],
        "preferred_locations": VALID_PROFILE["preferred_locations"],
        "remote_ok": True,
    }
    assert "Resume text" not in str(profile.model_dump())
    assert len(extractor.calls) == 1


def test_invalid_structured_output_retries_once_with_field_only_feedback():
    invalid = {**VALID_PROFILE, "unknown_field": "discard me"}
    extractor = StubExtractor([invalid, VALID_PROFILE])

    profile = extract_resume_profile("Resume text.", extractor)

    assert profile.skills == VALID_PROFILE["skills"]
    assert len(extractor.calls) == 2
    assert extractor.calls[0][1] is None
    assert "unknown_field" in extractor.calls[1][1]
    assert "discard me" not in extractor.calls[1][1]


def test_contact_email_is_rejected_and_retried_without_echoing_it():
    invalid = {**VALID_PROFILE, "summary": "Student. mudit@example.com"}
    extractor = StubExtractor([invalid, VALID_PROFILE])

    profile = extract_resume_profile("Resume text with contact details.", extractor)

    assert profile.summary == VALID_PROFILE["summary"]
    assert len(extractor.calls) == 2


def test_two_invalid_outputs_raise_sanitized_error_after_exactly_one_retry():
    extractor = StubExtractor(["not JSON", {**VALID_PROFILE, "skills": ["Python", 7]}])

    with pytest.raises(ResumeProfileError, match="after one retry") as error:
        extract_resume_profile("Resume text.", extractor)

    assert len(extractor.calls) == 2
    assert "Resume text" not in str(error.value)


def test_schema_is_strict_and_rejects_unknown_fields():
    schema = ResumeProfile.model_json_schema()

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    with pytest.raises(ValidationError):
        ResumeProfile.model_validate({**VALID_PROFILE, "extra": True}, strict=True)


def test_resume_input_is_bounded_and_never_sent_when_invalid():
    extractor = StubExtractor([VALID_PROFILE])
    with pytest.raises(ValueError, match="non-empty"):
        extract_resume_profile("   ", extractor)
    with pytest.raises(ValueError, match="character limit"):
        extract_resume_profile("x" * 30_001, extractor)
    assert extractor.calls == []


def test_ollama_adapter_rejects_non_loopback_hosts():
    with pytest.raises(ValueError, match="loopback"):
        OllamaResumeExtractor("qwen2.5:3b", "https://api.example.com")


def test_local_ollama_adapter_requires_explicit_model_configuration(monkeypatch):
    monkeypatch.delenv("PROBEFLOW_RESUME_MODEL", raising=False)
    monkeypatch.setenv("PROBEFLOW_OLLAMA_URL", "http://127.0.0.1:11434")
    assert configured_resume_extractor() is None

    monkeypatch.setenv("PROBEFLOW_RESUME_MODEL", "qwen2.5:3b")
    extractor = configured_resume_extractor()
    assert isinstance(extractor, OllamaResumeExtractor)
    assert extractor.url == "http://127.0.0.1:11434"

