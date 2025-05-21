from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from .agent.llm import get_llm, PROMPT_TEMPLATE
from .memory.vector_memory import get_vector_store
from .memory.graph_memory import get_driver, save_interaction

app = FastAPI(title="Jarvis API")

neo4j_driver = get_driver()
vector_store = get_vector_store()

prompt = PROMPT_TEMPLATE

memory = ConversationBufferMemory()
chain = ConversationChain(llm=get_llm(), memory=memory, prompt=prompt)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = await chain.apredict(input=request.message)
        save_interaction(neo4j_driver, request.message, response)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
