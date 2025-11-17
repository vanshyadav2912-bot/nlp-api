import os
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env

app = FastAPI()

# Pydantic model for input
class TextRequest(BaseModel):
    text: str

# Environment variables
HF_API_URL = os.environ.get("HF_API_URL")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_MODEL_NAME = os.environ.get("HF_MODEL_NAME")

# System prompt for the model
SYSTEM_PROMPT = (
    "You are a helpful and accurate data extraction assistant. "
    "Your task is to parse the user's input text and extract specific data fields. "
    "You MUST respond ONLY with a valid JSON object matching the following structure: "
    '{"name": "...", "email": null, "age": null, "gender": "..."} '
    "If a value is missing, use null. DO NOT include any text, markdown, or explanation outside of the JSON object."
)

# Ensure environment variables
def ensure_env_vars():
    if not HF_API_URL or not HF_API_TOKEN or not HF_MODEL_NAME:
        raise RuntimeError("HF_API_URL, HF_API_TOKEN, and HF_MODEL_NAME environment variables must be set.")

# Extract JSON safely from model output
def extract_json_text(text: str) -> str:
    """
    Convert model output to valid JSON string.
    Handles Python-style dicts with single quotes and None -> null.
    """
    if not text:
        raise ValueError("Empty text from model.")

    # Remove markdown code blocks
    text = text.replace("```json", "").replace("```", "").strip()

    # Convert Python-style dict to JSON
    text = text.replace("None", "null").replace("'", '"')

    # Find first balanced JSON object
    start = text.find("{")
    if start == -1:
        return text.strip()

    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                return text[start:i+1].strip()

    return text.strip()

# Home endpoint
@app.get("/")
def home():
    return {"status": "ok", "note": 'POST JSON like {"text": "..."} to /process'}

# Main POST endpoint
@app.post("/process")
def process(request: TextRequest):
    try:
        ensure_env_vars()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    user_input_text = request.text
    if not user_input_text:
        raise HTTPException(status_code=400, detail='Missing "text" field in request body.')

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": HF_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input_text}
        ],
        "max_tokens": 300
    }

    # Call Hugging Face Chat API
    try:
        hf_response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        hf_response.raise_for_status()
    except requests.exceptions.RequestException as re_err:
        raise HTTPException(status_code=502, detail=f"HuggingFace request failed: {str(re_err)}")

    # Parse response
    try:
        result = hf_response.json()
        raw_text = result["choices"][0]["message"]["content"].strip()
    except Exception:
        raise HTTPException(status_code=502, detail="Failed to parse response from HuggingFace Chat API.")

    # Extract JSON safely
    try:
        candidate = extract_json_text(raw_text)
        extracted_data = json.loads(candidate)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"Failed to parse JSON from model output. Raw output: {raw_text}")

    return JSONResponse(content=extracted_data)
