"""Single-agent research crew: DuckDuckGo search + Groq LLM (openai/gpt-oss-120b).

Groq is reached through its OpenAI-compatible endpoint using CrewAI's NATIVE OpenAI
integration (custom_openai=True). This avoids LiteLLM completely, which had install
problems with recent CrewAI versions.
"""

import functools
import inspect
import os
import re
import time

# Must be set BEFORE crewai is imported: no telemetry, no interactive tracing prompt.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")



# --------------------------------------------------------------------------
# GROQ SAFETY NET
# Some CrewAI 1.14+/1.15 code paths add a private key "cache_breakpoint" to
# chat messages (used for Anthropic prompt caching). Groq rejects it with:
#   "property 'cache_breakpoint' is unsupported"   (CrewAI issue #5886)
# CrewAI's native OpenAI-compatible path is supposed to remove it already;
# this small patch removes it once more at the last moment (inside the
# `openai` SDK call), so the request is clean whatever CrewAI version runs.
# --------------------------------------------------------------------------
def _strip_cache_breakpoint(messages):
    """Return a copy of `messages` without the 'cache_breakpoint' key."""
    if not isinstance(messages, list):
        return messages
    cleaned = []
    for message in messages:
        if isinstance(message, dict) and "cache_breakpoint" in message:
            message = {k: v for k, v in message.items() if k != "cache_breakpoint"}
        cleaned.append(message)
    return cleaned


def _install_groq_safety_net() -> None:
    try:
        from openai.resources.chat import completions as chat_completions
    except Exception:  # openai layout changed or not installed -> skip quietly
        return

    def wrap_sync(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if "messages" in kwargs:
                kwargs["messages"] = _strip_cache_breakpoint(kwargs["messages"])
            return fn(self, *args, **kwargs)
        return wrapper

    def wrap_async(fn):
        @functools.wraps(fn)
        async def wrapper(self, *args, **kwargs):
            if "messages" in kwargs:
                kwargs["messages"] = _strip_cache_breakpoint(kwargs["messages"])
            return await fn(self, *args, **kwargs)
        return wrapper

    for class_name in ("Completions", "AsyncCompletions"):
        cls = getattr(chat_completions, class_name, None)
        original = getattr(cls, "create", None) if cls else None
        if original is None or getattr(original, "_cache_breakpoint_fix", False):
            continue  # missing, or already patched (Streamlit re-runs the script)
        wrapper = wrap_async(original) if inspect.iscoroutinefunction(original) else wrap_sync(original)
        wrapper._cache_breakpoint_fix = True
        cls.create = wrapper


_install_groq_safety_net()

from crewai import Agent, Crew, LLM, Process, Task  # noqa: E402

from search_tool import build_search_tool  # noqa: E402

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL_ID = "openai/gpt-oss-120b"  # the model id exactly as Groq names it

# CrewAI strips ONE leading "openai/" routing prefix and sends the rest to Groq,
# so the first candidate reaches Groq as "openai/gpt-oss-120b". The second candidate
# is only tried automatically if Groq answers "model not found".
MODEL_CANDIDATES = [f"openai/{GROQ_MODEL_ID}", GROQ_MODEL_ID]

MAX_RATE_LIMIT_RETRIES = 3  # Groq free tier: ~8,000 tokens/minute for this model


def _build_llm(api_key: str, model: str) -> LLM:
    return LLM(
        model=model,
        custom_openai=True,
        base_url=GROQ_BASE_URL,
        api_key=api_key,
        temperature=0.3,
        max_completion_tokens=3000,  # upper limit for the report length
        reasoning_effort="low",      # "low" | "medium" | "high"; lower = fewer tokens used
    )


def _build_crew(api_key: str, max_results: int, model: str) -> Crew:
    researcher = Agent(
        role="Senior Research Analyst",
        goal="Find reliable, current information on a topic and turn it into a clear report.",
        backstory=(
            "You are a careful analyst. You search the web, compare what you find, "
            "and only write facts that are supported by your search results."
        ),
        llm=_build_llm(api_key, model),
        tools=[build_search_tool(max_results)],
        allow_delegation=False,
        max_iter=6,
        verbose=False,
    )

    task = Task(
        description=(
            "Research the topic: {topic}\n\n"
            "Use the web search tool at most 3 times, with different and specific queries. "
            "Then write a well-structured report using only information from the search results."
        ),
        expected_output=(
            "A report in Markdown, about 600-900 words, with these sections:\n"
            "# Title\n"
            "## Executive Summary\n"
            "## Key Findings (bullet points)\n"
            "## Detailed Analysis (2-4 short subsections)\n"
            "## Limitations and Open Questions\n"
            "## Conclusion\n"
            "## Sources (list only URLs that appeared in the search results)\n"
            "Never invent facts or URLs."
        ),
        agent=researcher,
    )

    return Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )


def _is_rate_limit(err: Exception) -> bool:
    text = str(err).lower()
    return "rate limit" in text or "ratelimit" in text or "429" in text


def _is_model_error(err: Exception) -> bool:
    text = str(err).lower()
    if _is_rate_limit(err) or "model" not in text:
        return False
    keywords = ("not found", "does not exist", "model_not_found", "decommissioned",
                "invalid model", "unknown model")
    return any(word in text for word in keywords)


def _wait_seconds(err: Exception) -> float:
    """Read Groq's 'try again in 7.5s' hint; fall back to 20 seconds."""
    match = re.search(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)s", str(err), re.IGNORECASE)
    if match:
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2))
        return min(minutes * 60 + seconds + 2, 65)
    return 20


def run_research(topic: str, max_results: int = 4, api_key: str | None = None) -> str:
    """Research `topic` and return the finished report as Markdown text."""
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to Streamlit secrets (see README).")

    model_index = 0
    rate_limit_waits = 0
    while True:  # ends by returning a report or raising (both counters are bounded)
        try:
            crew = _build_crew(api_key, max_results, MODEL_CANDIDATES[model_index])
            result = crew.kickoff(inputs={"topic": topic})
            return getattr(result, "raw", None) or str(result)
        except Exception as err:
            if _is_model_error(err) and model_index + 1 < len(MODEL_CANDIDATES):
                model_index += 1
                continue
            if _is_rate_limit(err) and rate_limit_waits < MAX_RATE_LIMIT_RETRIES:
                rate_limit_waits += 1
                time.sleep(_wait_seconds(err))
                continue
            raise
