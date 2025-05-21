from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from .agent.llm import get_llm, prompt
from .memory.vector_memory import (
    get_vector_store,
    add_conversation_snippet,
    query_conversation_snippets,
)
from .memory.graph_memory import get_driver

app = FastAPI(title="Jarvis API")

memory = ConversationBufferMemory()
chain = ConversationChain(llm=get_llm(), memory=memory, prompt=prompt)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = chain.predict(input=request.message)
        add_conversation_snippet(request.message, {"role": "user"})
        add_conversation_snippet(response, {"role": "assistant"})
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
