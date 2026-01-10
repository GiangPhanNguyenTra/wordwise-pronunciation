import uvicorn
import webbrowser
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json

from services import tts_service, scoring_service, sample_service
from core.models.rule_based import get_phonem_converter
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Word Wise AI Pronunciation Scoring")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class TTSRequest(BaseModel):
    value: str

class ScoreRequest(BaseModel):
    title: str
    base64Audio: str
    language: str

class IPARequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.post("/getExampleWord")
async def get_example_word():
    return sample_service.get_random_word()

@app.post("/getExampleSentence")
async def get_example_sentence():
    return sample_service.get_random_sentence()

class SampleRequest(BaseModel):
    count: int

@app.post("/getExampleWords/{count}")
async def get_example_words(count: int):
    return sample_service.get_random_words(count)

@app.post("/getExampleSentences/{count}")
async def get_example_sentences(count: int):
    return sample_service.get_random_sentences(count)


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
    
@app.post("/getIPA")
async def get_ipa(payload: IPARequest):
    try:
        converter = get_phonem_converter("en")
        ipa = converter.convertToPhonem(payload.text)
        return {"ipa": ipa}
    except Exception as e:
        print(f"Error in getIPA: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    url = "http://127.0.0.1:8001"
    print(f"Starting server at {url}")
    webbrowser.open_new(url)
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)