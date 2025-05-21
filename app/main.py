
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from .agent.llm import get_llm, prompt

from .memory.graph_memory import get_driver, save_interaction
from .memory.vector_memory import get_vector_store

app = FastAPI(title="Jarvis API")

vector_store = get_vector_store()
neo4j_driver = get_driver()

memory = ConversationBufferMemory()
chain = ConversationChain(llm=get_llm(), memory=memory, prompt=prompt)


class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = chain.predict(input=request.message)
        save_interaction(neo4j_driver, request.message, response)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


