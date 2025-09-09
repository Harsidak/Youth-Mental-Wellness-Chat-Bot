# Youth Mental Wellness Chat Bot

Simple chat bot for youth mental wellness built with Python and LangChain-style RAG components. This repository contains the main app, service modules, templates and static assets.

## Quick start

1. Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate     # Windows

2. Install dependencies:
   pip install -r [requirements.txt](requirements.txt)

3. Configure credentials:
   - Edit [Services/Credentials.py](Services/Credentials.py) or supply environment variables as needed.

4. Run the app:
   python [main.py](main.py)
   - Alternatively run the example inside the subproject:
     python [Youth-Mental-Wellness-Chat-Bot/main.py](Youth-Mental-Wellness-Chat-Bot/main.py)

## Project structure

- [main.py](main.py) — root application entry point  
- [requirements.txt](requirements.txt) — Python dependencies for the root app  
- Services/  
  - [Services/Credentials.py](Services/Credentials.py) — credential/config helper  
  - [Services/RAG.py](Services/RAG.py) — retrieval-augmented generation logic  
  - [Services/Indexing.py](Services/Indexing.py) — indexing and document processing  
- [templates/chat.html](templates/chat.html) — main chat UI  
- [templates/landing.html](templates/landing.html) — landing page  
- [static/](static/) — static assets (CSS, JS, images)  
- [Youth-Mental-Wellness-Chat-Bot/main.py](Youth-Mental-Wellness-Chat-Bot/main.py) — packaged example app  
- [Youth-Mental-Wellness-Chat-Bot/requirements.txt](Youth-Mental-Wellness-Chat-Bot/requirements.txt) — subproject deps

## Configuration

- Ensure API keys and sensitive values are kept out of source control. Use [Services/Credentials.py](Services/Credentials.py) or environment variables.
- The app expects the document store/nlp models to be configured via the Services modules.

## Development

- Run the app locally and use the UI at the address printed to the console.
- Modify service code in [Services/](Services/) to change indexing, retrieval or response generation.

## Notes

- This project uses multiple third-party Python packages listed in [requirements.txt](requirements.txt) and the subproject requirements file.
- Sensitive files (API keys, tokens) should never be committed.

## License

Add a LICENSE file if you wish to open
