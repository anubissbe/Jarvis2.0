# Jarvis 2.0

Jarvis 2.0 is a minimal FastAPI-based service that demonstrates a bilingual AI assistant. It uses a simple LLM implementation and includes an optional graph memory backend. A vector store helper is provided but not used in the default API.

## Setup

1. Install Python 3.11.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   ```bash
   docker compose up --build
   ```
   Once running, browse to `http://localhost:8080` to access OpenWebUI. The API
   itself remains available on `http://localhost:8000`.
   Environment variables such as `OLLAMA_BASE_URL` and `WEBUI_SECRET_KEY` are
   defined in `docker-compose.yml` and mirror the settings expected by
   `app/config.py`.

## Usage

Run the API locally:
```bash
uvicorn app.main:app --reload
```
Then POST to `/chat` with a JSON body containing `message`.

For more details on the overall architecture, see `architectdesign.md`.

## License

This project is licensed under the [MIT License](LICENSE).
