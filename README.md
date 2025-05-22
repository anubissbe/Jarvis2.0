# Jarvis 2.0

Jarvis 2.0 is a minimal FastAPI-based service that demonstrates a bilingual AI assistant. It uses a simple LLM implementation and includes an optional graph memory backend. A vector store helper is provided but not used in the default API.

## Setup

1. Install Python 3.11.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install dev dependencies for testing:
   ```bash
   pip install -r requirements-dev.txt
   ```

4. Build and start the Docker services:
   ```bash
   docker compose up --build
   ```
   Once running, browse to `http://localhost:8080` to access OpenWebUI. The API
   itself remains available on `http://localhost:8000`.

   Environment variables such as `OLLAMA_BASE_URL` and `WEBUI_SECRET_KEY` are
   defined in `docker-compose.yml` and mirror the settings expected by
   `app/config.py`.

## Configuration

Copy `.env.example` to `.env` and update the values to point at your own
services. At a minimum you should review the following variables:

- `OLLAMA_BASE_URL` - URL for the Ollama LLM service.
- `CHROMA_DB_URL` - URL for the ChromaDB vector store (if used).
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` - connection settings for the
  Neo4j graph database.
- `WEBUI_SECRET_KEY` - secret key for securing OpenWebUI sessions.

These environment variables must be configured before running the containers so
that Jarvis can connect to the required services.

## Usage

Run the API locally:
```bash
uvicorn app.main:app --reload
```
Then POST to `/chat` with a JSON body containing `message`.

### Using `jarvis.sh`

The `jarvis.sh` helper script bundles a few convenience steps. Run it from the
project root to build and launch the Docker services and monitor their logs:

```bash
./jarvis.sh
```

It first compiles the Python files, verifies that Docker and `docker-compose`
are installed, and then brings the containers up in the background. Any error
messages from the services will be collected in `error.log` while the script
streams the combined logs to your terminal.

For more details on the overall architecture, see `architectdesign.md`.

## License

This project is licensed under the [MIT License](LICENSE).

## Testing

Install the development requirements and run the test suite using `pytest`:

```bash
pip install -r requirements-dev.txt
pytest
```
