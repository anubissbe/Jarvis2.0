from fastapi import FastAPI, HTTPException
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from .agent.llm import get_llm, prompt
from .memory.vector_memory import get_vector_store
from .memory.graph_memory import get_driver

app = FastAPI(title="Jarvis API")

vector_store = get_vector_store()
neo4j_driver = get_driver()

memory = ConversationBufferMemory()
chain = ConversationChain(llm=get_llm(), memory=memory, prompt=prompt)

@app.post("/chat")
async def chat(input: str):
    try:
        response = chain.predict(input=input)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

