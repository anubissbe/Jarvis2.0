from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent.llm import get_llm
from .memory.graph_memory import get_driver, save_interaction
from .memory.vector_memory import get_vector_store


class SimpleChain:
    """Minimal stand-in for a conversation chain."""

    def __init__(self, llm):
        self.llm = llm

    async def apredict(self, input: str) -> str:
        return self.llm.generate(input)


app = FastAPI(title="Jarvis API")

# Initialize dependencies
chain = SimpleChain(llm=get_llm())
neo4j_driver = get_driver()


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = await chain.apredict(input=request.message)
        if neo4j_driver is not None:
            await save_interaction(neo4j_driver, request.message, response)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Close the Neo4j driver if it was created."""
    if neo4j_driver is not None:
        await neo4j_driver.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
