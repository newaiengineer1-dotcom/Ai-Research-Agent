"""Streamlit app: type a topic, get a researched report."""

import os
import re

import streamlit as st

from research_agent import run_research

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="wide")


def get_groq_key() -> str:
    """Read the key from Streamlit secrets (cloud or .streamlit/secrets.toml), else from env."""
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key:
            return str(key)
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")


st.title("🔎 AI Research Agent")
st.caption("CrewAI  •  Groq (openai/gpt-oss-120b)  •  DuckDuckGo search")

with st.sidebar:
    st.header("Settings")
    max_results = st.slider(
        "Search results per search",
        min_value=2,
        max_value=8,
        value=4,
        help="More results = better coverage but more tokens. "
        "Keep it low on Groq's free tier.",
    )
    st.info(
        "Groq's free tier has a small tokens-per-minute limit. "
        "If you see a wait, the app retries automatically."
    )

topic = st.text_input(
    "Research topic",
    max_chars=200,
    placeholder="e.g. Impact of solar power on rural electricity access",
)

if st.button("Generate report", type="primary"):
    api_key = get_groq_key()
    if not topic.strip():
        st.warning("Please enter a topic first.")
    elif not api_key:
        st.error(
            "GROQ_API_KEY not found. On Streamlit Cloud add it under "
            "App settings -> Secrets. Locally, put it in .streamlit/secrets.toml."
        )
    else:
        with st.spinner("Researching and writing... this can take 30-90 seconds."):
            try:
                st.session_state["report"] = run_research(topic.strip(), max_results, api_key)
                st.session_state["topic"] = topic.strip()
            except Exception as err:
                st.session_state.pop("report", None)
                st.error(f"Something went wrong: {err}")

if "report" in st.session_state:
    st.divider()
    st.markdown(st.session_state["report"])
    file_stem = re.sub(r"[^a-zA-Z0-9]+", "_", st.session_state.get("topic", "report")).strip("_")
    st.download_button(
        "Download report (.md)",
        data=st.session_state["report"],
        file_name=f"{file_stem[:50] or 'report'}.md",
        mime="text/markdown",
    )
