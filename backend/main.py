from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline
from PIL import Image
import pytesseract
import io

app = FastAPI()

# Initialize the summarization model
summarizer = pipeline("summarization")

# Endpoint for text extraction from image
@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Perform OCR using pytesseract
        extracted_text = pytesseract.image_to_string(image)

        if not extracted_text:
            return JSONResponse(content={"error": "No text found in the image"}, status_code=400)

        return {"extracted_text": extracted_text}
    except Exception as e:
        return JSONResponse(content={"error": f"Error extracting text: {str(e)}"}, status_code=500)

# Endpoint for text summarization
@app.post("/summarize/")
async def summarize_text(payload: dict):
    try:
        text = payload.get("text")
        
        if not text:
            return JSONResponse(content={"error": "No text provided for summarization"}, status_code=400)

        # Generate a summary with a maximum of 50 words
        summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
        summarized_text = summary[0]['summary_text']

        return {"summary": summarized_text}
    except Exception as e:
        return JSONResponse(content={"error": f"Error during summarization: {str(e)}"}, status_code=500)
