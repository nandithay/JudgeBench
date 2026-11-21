"""Two small provider adapters, with bounded retries and no hidden model fallback."""
import os
import time

from .schemas import ModelSpec


class APIError(RuntimeError):
    pass


def require_key(provider):
    name = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}[provider]
    value = os.environ.get(name)
    if not value or value == "replace_locally":
        raise APIError(f"Set {name} locally before running live API calls.")
    return value


def _call(spec, system, user, schema):
    key = require_key(spec.provider)
    if spec.provider == "groq":
        from groq import Groq
        kwargs = {}
        if schema:
            # JSON mode works on more models than strict structured outputs.
            # Pydantic validation still runs locally; invalid output is retained.
            kwargs["response_format"] = {"type": "json_object"}
            system += "\nJSON schema:\n" + __import__("json").dumps(schema)
        with Groq(api_key=key, timeout=90, max_retries=0) as client:
            result = client.chat.completions.create(
                model=spec.model, temperature=spec.temperature,
                max_completion_tokens=spec.max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                **kwargs)
            raw = result.model_dump(mode="json")
            choice = result.choices[0]
            if choice.finish_reason != "stop":
                return {"text": choice.message.content or "", "raw": raw, "complete": False}
            return {"text": choice.message.content or "", "raw": raw, "complete": True}
    from google import genai
    from google.genai import types
    config = {"system_instruction": system, "temperature": spec.temperature,
              "max_output_tokens": spec.max_tokens}
    if schema:
        config.update(response_mime_type="application/json", response_json_schema=schema)
    with genai.Client(api_key=key, http_options=types.HttpOptions(timeout=90000)) as client:
        result = client.models.generate_content(model=spec.model, contents=user, config=config)
        raw = result.model_dump(mode="json")
        candidates = result.candidates or []
        finish = str(candidates[0].finish_reason) if candidates else ""
        return {"text": result.text or "", "raw": raw,
                "complete": finish in ("STOP", "FinishReason.STOP")}


def call_api(spec, system, user, schema=None, max_attempts=3):
    spec = ModelSpec.model_validate(spec)
    require_key(spec.provider)
    start = time.perf_counter()
    for attempt in range(max_attempts):
        try:
            result = _call(spec, system, user, schema)
            return {**result, "latency_seconds": time.perf_counter() - start,
                    "attempts": attempt + 1}
        except Exception as exc:
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            transient = status in (429, 500, 502, 503, 504) or isinstance(exc, (TimeoutError, ConnectionError))
            transient = transient or type(exc).__name__ in ("APITimeoutError", "APIConnectionError")
            if transient and attempt + 1 < max_attempts:
                headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
                try:
                    delay = float(headers.get("retry-after", 2 ** (attempt + 1)))
                except (ValueError, TypeError):
                    delay = 2 ** (attempt + 1)
                # A daily quota may ask for hours: exit and let the user resume later.
                if delay > 60:
                    raise APIError("Quota reset is more than 60 seconds away; resume this run later.") from None
                time.sleep(max(0, delay))
                continue
            # Provider exception messages/URLs can contain credentials. Never persist them.
            raise APIError(f"{spec.provider} call failed ({type(exc).__name__}, status={status}). "
                           "Check model access, quota, and credentials; saved results are safe to resume.") from None
