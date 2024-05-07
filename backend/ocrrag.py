

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

app = FastAPI()

from inference import chatbotInference

from ppaddleocr import OCRProcessor

ocr_processor = OCRProcessor()


UPLOAD_DIRECTORY = "uploads"
# Create the upload directory if it doesn't exist
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)



def save_upload_file(upload_file: UploadFile, destination: str):
    with open(destination, "wb") as buffer:
        buffer.write(upload_file.file.read())

# instance of chatbot
token_path = "path to your model"
engine_path = "path to your model engine"
chatbot = chatbotInference(token_path,engine_path)







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





@app.post("/upload/")
async def upload_images(file: UploadFile = File(...)):


    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    save_upload_file(file, file_path)


    # extract text from image 
    ocr_string = ocr_processor.process_image(file_path)




    chatbot.createRetriever(ocr_string)

    print(f"Uploaded file: {file.filename}")
    return {"message": "Files uploaded successfully"}


@app.post("/chat")
async def chat_completion(message: Message):
    # Your logic to generate a response based on the received message
    response = "This is a sample response." 
    msg = message.content

    response = chatbot.rag_chain.invoke(msg) #conversation({"question": msg})
    return {"response": response}


    #return {"response": "Sure, I can help you with that. Here's a simple FastAPI code that you can use as a starting point:\n```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get(\"/\")\nasync def read_root():\n    return {\"Hello\": \"World\"}\n```\nThis code creates a FastAPI application and defines a route for the root endpoint. When you run this code, you can access the root endpoint by visiting `http://localhost:8000/` in your web browser.\nIs there anything else you need help with?"}
    #return {"response": response['text']}


if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000)
