
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
app = FastAPI()


from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

from trt_inference_api import CustomLLM 


token_path = "/home/adebolajo/Desktop/trt-llm/mistral-7b-int4-chat_1.2/mistral7b_hf_tokenizer"
engine_path = "/home/adebolajo/Desktop/trt-llm/mistral-7b-int4-chat_1.2/trt_engines"

llm = CustomLLM(n=5,tokenizer_dir = token_path, engine_dir = engine_path)

# Notice that "chat_history" is present in the prompt template
template = """You are a nice chatbot having a conversation with a human.

Previous conversation:
{chat_history}

New human question: {question}
Response:"""
prompt = PromptTemplate.from_template(template)
# Notice that we need to align the `memory_key`
memory = ConversationBufferMemory(memory_key="chat_history")
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

# Allow CORS for all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Allow OPTIONS method
    allow_headers=["*"],
)

class Message(BaseModel):
    content: str

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="/workspace/chatbot/static"), name="static")

# Create a Jinja2Templates instance for rendering HTML templates
templates = Jinja2Templates(directory="/workspace/chatbot/templates")

@app.get("/",response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat_completion(message: Message):
    # Your logic to generate a response based on the received message
    response = "This is a sample response."
    msg = message.content
    response = conversation({"question": msg})
    print( msg, response['text'])


    #return {"response": "Sure, I can help you with that. Here's a simple FastAPI code that you can use as a starting point:\n```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get(\"/\")\nasync def read_root():\n    return {\"Hello\": \"World\"}\n```\nThis code creates a FastAPI application and defines a route for the root endpoint. When you run this code, you can access the root endpoint by visiting `http://localhost:8000/` in your web browser.\nIs there anything else you need help with?"}
    return {"response": response['text']}


if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000)
