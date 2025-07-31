# Automated-Content-Creator

A CrewAI-based project that creates an interactive Q&A interface for researching and generating content about AI trends using TavilySearch, Hugging Face models, and Gradio.

---

## Features

- **Research Agent:** Searches for AI trends using TavilySearch.
- **Outliner Agent:** Creates a concise outline with 5 subheadings.
- **Writer Agent:** Generates a detailed response (up to 1000 words).
- **Prompt Suggestions:** Suggests three possible prompts as you type using GPT-2.
- **Gradio Interface:** Web-based UI for user queries and responses.

---

## Requirements

- **Python:** 3.12.4 (64-bit)
- **Hardware:** GPU/CPU with ~5GB VRAM (for EleutherAI/gpt-neo-2.7B)
- **API:** Tavily API key (free tier: 1,000 credits/month)

---

## Setup

### 1. Create Virtual Environment

```bash
python -m venv env
env\Scripts\activate  # Windows
# or
source env/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install crewai crewai[tools] langchain-huggingface langchain-community transformers torch tavily-python gradio chromadb sentence-transformers
```

### 3. Set Up .env File

- Create a `.env` file in your project directory:
    ```
    TAVILY_API_KEY=your-tavily-api-key
    ```
- Get your Tavily API key from [Tavily's website](https://app.tavily.com/).

### 4. (Windows Only) Install Microsoft Visual C++ Redistributable

- Download and install the Visual C++ Redistributable 2015-2022 (x64) from [Microsoft](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

### 5. Run the Application

- Save your main script as `app.py` (see provided code).
- Run:
    ```bash
    python app.py
    ```
- Access the Gradio interface in your browser (URL provided in terminal).

---

## Usage

1. Open the Gradio interface.
2. Type a question about AI trends (e.g., "What are AI trends in 2025?").
3. View three prompt suggestions as you type.
4. Click "Submit" to get a detailed response based on search results.

---

## Project Structure

- `app.py` — Main script with CrewAI agents, TavilySearch, and Gradio interface.
- `README.md` — Project documentation.

---


## License

MIT License
