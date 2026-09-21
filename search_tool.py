"""DuckDuckGo search tool for the CrewAI research agent.

Why a function and not a class?
CrewAI tools built as classes are Pydantic models, and Pydantic refuses to
set attributes (like self.max_results) that were not declared as fields.
That was the cause of the error
    ValueError: "DuckDuckGoResearchTool" object has no field "max_results"
Building the tool from a plain function with the @tool decorator avoids this
completely: max_results is just a normal Python variable captured by the function.
"""

from crewai.tools import tool
from ddgs import DDGS  # the package "duckduckgo_search" was renamed to "ddgs"

SNIPPET_CHARS = 300  # keep results short: Groq's free tier has a small tokens-per-minute limit


def build_search_tool(max_results: int = 4):
    """Return a CrewAI tool that searches DuckDuckGo and returns `max_results` results."""

    @tool("DuckDuckGo Web Search")
    def web_search(query: str) -> str:
        """Search the web with DuckDuckGo. Input must be a short search query string.
        Returns a numbered list of results, each with a title, a URL and a short snippet."""
        try:
            results = DDGS().text(query, max_results=max_results)
        except Exception as err:  # network problems, rate limits from DuckDuckGo, etc.
            return f"Search failed ({err}). Try a different or shorter query."

        if not results:
            return "No results found. Try a different or broader query."

        lines = []
        for number, item in enumerate(results, start=1):
            title = item.get("title", "").strip()
            url = item.get("href", "").strip()
            snippet = (item.get("body") or "").strip()[:SNIPPET_CHARS]
            lines.append(f"{number}. {title}\n   URL: {url}\n   {snippet}")
        return "\n\n".join(lines)

    return web_search
