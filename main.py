import uvicorn
import webbrowser
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json

import tts_service
import scoring_service
from sample_service import get_random_word, get_random_sentence

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class TTSRequest(BaseModel):
    value: str

class ScoreRequest(BaseModel):
    title: str
    base64Audio: str
    language: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.post("/getExampleWord")
async def get_example_word():
    return get_random_word()

@app.post("/getExampleSentence")
async def get_example_sentence():
    return get_random_sentence()

@app.post("/getAudioFromText")
async def get_audio_from_text(payload: TTSRequest):
    event = {'body': payload.json()}
    return tts_service.lambda_handler(event, [])

@app.post("/GetAccuracyFromRecordedAudio")
async def get_accuracy_from_recorded_audio(payload: ScoreRequest):
    try:
        event = {'body': payload.json()}
        result = scoring_service.lambda_handler(event, [])
        return json.loads(result)
    except Exception as e:
        print(f'Error in GetAccuracyFromRecordedAudio: {e}')
        return {"error": str(e)}

if __name__ == "__main__":
    url = "http://127.0.0.1:8000"
    print(f"Starting server at {url}")
    webbrowser.open_new(url)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)