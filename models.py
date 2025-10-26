import torch
import torch.nn as nn
from ModelInterfaces import IASRModel

def getASRModel(language: str, use_whisper: bool = True) -> IASRModel:
    if use_whisper:
        from whisper_wrapper import WhisperASRModel
        return WhisperASRModel()
    else:
        # Fallback to Silero for English if whisper is false
        model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                               model='silero_stt',
                                               language='en',
                                               device=torch.device('cpu'),
                                               trust_repo=True)
        from AIModels import NeuralASR
        model.eval()
        return NeuralASR(model, decoder)

def getTTSModel(language: str) -> nn.Module:
    speaker = 'lj_16khz'
    # SỬA LỖI Ở ĐÂY: Bỏ ", _" vì hàm chỉ trả về 1 giá trị
    model = torch.hub.load(repo_or_dir='snakers4/silero-models',
                               model='silero_tts',
                               language='en',
                               speaker=speaker,
                               trust_repo=True)
    return model