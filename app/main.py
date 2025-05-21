
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent.llm import get_llm
from .memory.graph_memory import get_driver, save_interaction
from .memory.vector_memory import get_vector_store

app = FastAPI(title="Jarvis API")

vector_store = get_vector_store()
neo4j_driver = get_driver()

llm = get_llm()
conversation_history = []


class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        user_message = request.message
        conversation_history.append({"role": "user", "content": user_message})
        response = llm.generate(user_message)
        conversation_history.append({"role": "assistant", "content": response})
        save_interaction(neo4j_driver, user_message, response)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


