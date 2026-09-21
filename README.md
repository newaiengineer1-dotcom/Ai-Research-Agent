# AI Research Agent

Type a topic, get a researched report. Built with **CrewAI** (single agent),
**Groq** (`openai/gpt-oss-120b`, via Groq's OpenAI-compatible API - no LiteLLM), **DuckDuckGo** search and **Streamlit**.

## Files
| File | What it does |
|------|--------------|
| `app.py` | Streamlit web page (input box, button, report, download) |
| `research_agent.py` | CrewAI agent + task + crew, Groq model, rate-limit retry |
| `search_tool.py` | DuckDuckGo search tool the agent can call |
| `requirements.txt` | Python packages (versions pinned) |
| `.streamlit/secrets.toml.example` | Template for your Groq key |

## Run locally (Python 3.12 recommended; CrewAI needs 3.10-3.13)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit the key
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository (do NOT upload `secrets.toml`).
2. share.streamlit.io -> Create app -> pick repo, branch `main`, file `app.py`.
3. Advanced settings -> Python version **3.12** -> Secrets:
   `GROQ_API_KEY = "gsk_..."`
4. Deploy.
