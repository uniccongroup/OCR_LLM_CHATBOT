

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from typing import List
import os
from typing import Optional
import asyncio

app = FastAPI()

from inference import chatbotInference

from ocr_processor import OCRProcessor

ocr_processor = OCRProcessor()


UPLOAD_DIRECTORY = "uploads"
# Create the upload directory if it doesn't exist
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)



# instance of chatbot
token_path = "trt-llm/mistral-7b-int4-chat_1.2/mistral7b_hf_tokenizer"
engine_path = "trt-llm/mistral-7b-int4-chat_1.2/trt_engines"
chatbot = chatbotInference(model_token_path=token_path,model_path=engine_path)


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
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# Create a Jinja2Templates instance for rendering HTML templates
templates = Jinja2Templates(directory="backend/templates")

@app.get("/",response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):


  file_path = os.path.join(UPLOAD_DIRECTORY, image.filename)
  # Get image filename (optional)

  # Read image content as bytes
  contents = await image.read()

  #Save the image locally (optional)
  with open(file_path, "wb") as f:
     f.write(contents)

  ocr_string = ocr_processor.process_image(file_path)

  chatbot.createRetriever(ocr_string)


  return {"message": f"Image '{image.filename}' uploaded successfully!"}


@app.post("/chat")
async def chat_completion(message: Message):
    # Your logic to generate a response based on the received message
    response = "This is a sample response." 
    msg = message.content

    response = chatbot.rag_chain.invoke(msg) #conversation({"question": msg})
    return {"response": response}


if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000)
