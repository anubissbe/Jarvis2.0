from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from .agent.llm import get_llm, prompt

app = FastAPI(title="Jarvis API")

memory = ConversationBufferMemory()
chain = ConversationChain(llm=get_llm(), memory=memory, prompt=prompt)


class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = chain.predict(input=request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


