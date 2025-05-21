from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from .agent.llm import get_llm, prompt
from .memory.vector_memory import get_vector_store, add_memory
from .memory.graph_memory import get_driver, save_interaction

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
        user_text = request.message
        # Retrieve related memory from vector store
        docs = vector_store.similarity_search(user_text, k=2)
        context = "\n".join(doc.page_content for doc in docs)

        response = chain.predict(input=f"{context}\n{user_text}")

        add_memory(vector_store, user_text)
        add_memory(vector_store, response)
        save_interaction(neo4j_driver, user_text, response)

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

