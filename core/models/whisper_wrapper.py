import torch 
from transformers import pipeline
from .interfaces import IASRModel
from typing import Union
import numpy as np 

class WhisperASRModel(IASRModel):
    def __init__(self, model_name="openai/whisper-base"):
        self.asr = pipeline(
            "automatic-speech-recognition", 
            model=model_name,
            chunk_length_s=30 
        )
        self._transcript = ""
        self._word_locations = []
        self.sample_rate = 16000

    def processAudio(self, audio: Union[np.ndarray, torch.Tensor]):
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        
        audio_input = audio[0] if audio.ndim > 1 else audio
        
        try:
            result = self.asr(
                audio_input, 
                return_timestamps="word",
                generate_kwargs={
                    "language": "english", 
                    "task": "transcribe",
                    "condition_on_prev_tokens": False, 
                    "temperature": 0.0
                }
            )

            self._transcript = result.get("text", "").strip()
            chunks = result.get("chunks", [])
            
            self._word_locations = []
            for word_info in chunks:
                raw_start = word_info.get("timestamp", [None, None])[0]
                raw_end = word_info.get("timestamp", [None, None])[1]
                
                if raw_start is not None:
                    start_sample = raw_start * self.sample_rate
                    if raw_end is not None:
                        end_sample = raw_end * self.sample_rate
                    else:
                        end_sample = start_sample + (0.3 * self.sample_rate)
                        
                    self._word_locations.append({
                        "word": word_info.get("text", "").strip(),
                        "start_ts": start_sample,
                        "end_ts": end_sample,
                        "tag": "processed"
                    })

        except Exception as e:
            print(f"ASR Error: {e}")
            self._transcript = ""
            self._word_locations = []

    def getTranscript(self) -> str:
        return self._transcript

    def getWordLocations(self) -> list:
        return self._word_locations