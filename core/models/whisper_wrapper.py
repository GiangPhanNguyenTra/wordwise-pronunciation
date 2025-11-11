import torch 
from transformers import pipeline
from .interfaces import IASRModel
from typing import Union
import numpy as np 

class WhisperASRModel(IASRModel):
    def __init__(self, model_name="openai/whisper-base"):
        self.asr = pipeline(
            "automatic-speech-recognition", 
            model=model_name
        )
        self._transcript = ""
        self._word_locations = []
        self.sample_rate = 16000

    def processAudio(self, audio: Union[np.ndarray, torch.Tensor]):
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        
        result = self.asr(
            audio[0], 
            return_timestamps="word",
            generate_kwargs={"language": "english", "task": "transcribe"}
        )

        self._transcript = result.get("text", "").strip()
        chunks = result.get("chunks", [])
        
        self._word_locations = [{
            "word": word_info["text"], 
            "start_ts": word_info["timestamp"][0] * self.sample_rate if word_info["timestamp"][0] is not None else None,
            "end_ts": (word_info["timestamp"][1] * self.sample_rate if word_info["timestamp"][1] is not None else (word_info["timestamp"][0] + 1) * self.sample_rate),
            "tag": "processed"
        } for word_info in chunks]

    def getTranscript(self) -> str:
        return self._transcript

    def getWordLocations(self) -> list:
        return self._word_locations