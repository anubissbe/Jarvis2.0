from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities.tavily_search import TavilySearchAPIWrapper

from .agent.llm import get_llm, prompt
from .memory.vector_memory import get_vector_store, store_interaction
from .memory.graph_memory import get_driver, save_interaction

app = FastAPI(title="Jarvis API")

# Initialize external resources
vector_store = get_vector_store()
neo4j_driver = get_driver()
search = TavilySearchAPIWrapper()

memory = ConversationBufferMemory()
chain = ConversationChain(llm=get_llm(), memory=memory, prompt=prompt)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Retrieve context from vector store
        docs = vector_store.similarity_search(request.message, k=3)
        context = "\n".join(doc.page_content for doc in docs)

        # Perform internet search for additional context
        search_results = search.results(request.message, 3)
        search_context = "\n".join(r["content"] for r in search_results)

        full_context = f"{context}\n{search_context}".strip()

        response = chain.predict(input=request.message, context=full_context)

        # Persist interaction
        store_interaction(vector_store, request.message, response)
        save_interaction(neo4j_driver, request.message, response)

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
