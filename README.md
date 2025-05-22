# Jarvis 2.0

Jarvis 2.0 is a minimal FastAPI-based service that demonstrates a bilingual AI assistant. It uses a simple LLM implementation and includes optional vector and graph memory backends.

## Setup

1. Install Python 3.11.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and update the values if needed.
   The required variables are documented in that file.
4. (Optional) Start the full stack with Docker Compose:
   ```bash
   docker compose up --build
   ```

## Usage

Run the API locally:
```bash
uvicorn app.main:app --reload
```
Then POST to `/chat` with a JSON body containing `message`.

For more details on the overall architecture, see `architectdesign.md`.

## License

This project is licensed under the [MIT License](LICENSE).
