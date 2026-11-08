"""LLM provider adapters.

Every provider exposes one method:

    generate(system: str, messages: list[{"role": "user"|"assistant", "content": str}]) -> str

and raises ``LLMError`` (with a message that is safe to show to end users) on any failure.
"""
import time

import config


class LLMError(Exception):
    """An error whose ``user_message`` can be shown directly in the UI."""

    def __init__(self, user_message: str, status: int = 502, kind: str = "api_error"):
        super().__init__(user_message)
        self.user_message = user_message
        self.status = status
        self.kind = kind


def classify(exc: Exception) -> LLMError:
    """Turn any SDK / network exception into a friendly LLMError."""
    if isinstance(exc, LLMError):
        return exc
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)

    if "timeout" in name or "timed out" in text or isinstance(exc, TimeoutError):
        return LLMError("The AI service took too long to respond. Please try again.", 504, "timeout")
    if code == 429 or "ratelimit" in name or "rate limit" in text or "resource_exhausted" in text:
        return LLMError("Rate limit reached (free-tier quota). Wait a few seconds and try again.", 429, "rate_limit")
    if code in (401, 403) or "authentication" in name or "permission" in name or "api key" in text:
        return LLMError("The API key was rejected. Check the key in your .env file.", 502, "auth")
    if code == 404 or "notfound" in name:
        return LLMError("The configured model was not found. Check GEMINI_MODEL / OPENAI_MODEL in .env.", 502, "model")
    if "connection" in name or isinstance(exc, ConnectionError) or "connect" in text:
        return LLMError("Could not reach the AI service. Check your internet connection.", 503, "connection")
    return LLMError("The AI service returned an error. Please try again in a moment.", 502, "api_error")


class BaseProvider:
    name = "base"

    def generate(self, system: str, messages: list[dict]) -> str:
        """Call the model, retrying transient failures (rate limit / network) with backoff."""
        attempt = 0
        while True:
            try:
                return self._generate_once(system, messages)
            except Exception as exc:  # noqa: BLE001 - we normalise every failure
                err = classify(exc)
                transient = err.kind in ("rate_limit", "connection")
                if transient and attempt < config.MAX_RETRIES:
                    attempt += 1
                    time.sleep(2 ** attempt)  # 2s, 4s
                    continue
                raise err from exc

    def _generate_once(self, system: str, messages: list[dict]) -> str:
        raise NotImplementedError


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout: float):
        from google import genai
        from google.genai import types

        self._types = types
        self.model = model
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),  # milliseconds
        )

    def _generate_once(self, system, messages):
        t = self._types
        contents = [
            t.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[t.Part(text=m["content"])],
            )
            for m in messages
        ]
        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=t.GenerateContentConfig(
                system_instruction=system, temperature=0.7, max_output_tokens=2048
            ),
        )
        text = (resp.text or "").strip()
        if not text:
            raise LLMError(
                "The model returned an empty reply (possibly blocked by a safety filter). Try rephrasing.",
                502,
                "empty_response",
            )
        return text


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str, timeout: float):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)

    def _generate_once(self, system, messages):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            temperature=0.7,
            max_tokens=1024,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise LLMError("The model returned an empty reply. Try again.", 502, "empty_response")
        return text


class MockProvider(BaseProvider):
    """Offline provider for tests and demos - no API key needed.

    * Replies by echoing the last message and the number of messages it received.
    * Type "FAIL" in a message to simulate an API failure.
    * Type "SLOW" to simulate a timeout.
    * Ask "what did I say first" to see that earlier history is really being sent.
    """

    name = "mock"

    def __init__(self):
        self.calls = []  # (system, messages) for assertions in tests

    def _generate_once(self, system, messages):
        self.calls.append((system, [dict(m) for m in messages]))
        last = messages[-1]["content"]
        if "FAIL" in last:
            raise RuntimeError("simulated failure")
        if "SLOW" in last:
            raise TimeoutError("simulated timeout")
        if system.startswith("You compress conversations"):
            return f"Summary covering {last.count(chr(10)) + 1} lines of earlier chat."
        if "what did i say first" in last.lower():
            first_user = next(m["content"] for m in messages if m["role"] == "user")
            return f'Earlier you said: "{first_user}"'
        return f"[mock] You said: {last} (I can see {len(messages)} messages of context)"


def get_provider() -> BaseProvider:
    p = config.PROVIDER
    if p == "mock":
        return MockProvider()
    if p == "gemini":
        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
                "(free key: https://aistudio.google.com/apikey)."
            )
        return GeminiProvider(config.GEMINI_API_KEY, config.GEMINI_MODEL, config.REQUEST_TIMEOUT)
    if p == "openai":
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        return OpenAIProvider(config.OPENAI_API_KEY, config.OPENAI_MODEL, config.REQUEST_TIMEOUT)
    raise RuntimeError(f"Unknown LLM_PROVIDER '{p}'. Use gemini, openai or mock.")
