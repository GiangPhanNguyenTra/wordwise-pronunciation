import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from .interfaces import IASRModel

class Wav2VecPhonemeModel(IASRModel):
    def __init__(self, model_name="vitouphy/wav2vec2-xls-r-300m-phoneme"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self._transcript = ""
        self._phoneme_locations = []
        self.sample_rate = 16000

    def processAudio(self, audio):
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().numpy()
        if audio.ndim > 1:
            audio = audio[0]

        inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        output_ipa = self.processor.batch_decode(predicted_ids)[0]
        self._transcript = output_ipa
        
        # Tính toán timestamp từ logits (CTC frames)
        # 1 frame ~ 20ms cho model Wav2Vec2 tiêu chuẩn
        time_offset = self.sample_rate / logits.shape[1]
        
        self._phoneme_locations = []
        # Giải mã các token để lấy vị trí frame (bỏ qua pad token)
        for i, token_id in enumerate(predicted_ids[0]):
            if token_id != self.processor.tokenizer.pad_token_id:
                token = self.processor.tokenizer.decode(token_id)
                if token.strip():
                    self._phoneme_locations.append({
                        "phoneme": token,
                        "start_sample": i * time_offset,
                        "end_sample": (i + 1) * time_offset
                    })

    def getTranscript(self) -> str:
        return self._transcript

    def getPhonemeLocations(self) -> list:
        return self._phoneme_locations

    def getWordLocations(self) -> list:
        # Hàm này để tương thích interface cũ, Word locations sẽ được Trainer tính toán
        return []
