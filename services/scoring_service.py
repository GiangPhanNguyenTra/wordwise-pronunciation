import torch
import json
import os
import base64
import time
import audioread
import numpy as np
from torchaudio.transforms import Resample
import tempfile

from core import pronunciation_trainer
from core.algorithms import word_matching as wm

trainer_SST_lambda = pronunciation_trainer.getTrainer("en")
transform = Resample(orig_freq=48000, new_freq=16000)

def lambda_handler(event, context):
    data = json.loads(event['body'])
    real_text = data['title']
    audio_string = data['base64Audio']

    if ',' in audio_string:
        header, encoded = audio_string.split(",", 1)
        file_bytes = base64.b64decode(encoded.encode('utf-8'))
    else:
        file_bytes = base64.b64decode(audio_string.encode('utf-8'))

    if not real_text:
        return json.dumps({})

    tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
    tmp_name = tmp.name
    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()
        signal, fs = audioread_load(tmp_name)
    finally:
        os.remove(tmp_name)

    signal = transform(torch.Tensor(signal)).unsqueeze(0)
    result = trainer_SST_lambda.processAudioForGivenText(signal, real_text)

    # Lấy các cặp từ đã được so khớp bởi hàm chính
    real_and_transcribed_words = result['real_and_transcribed_words']

    # --- LOGIC CŨ ĐƯỢC PHỤC HỒI VÀ SỬA LỖI ---
    # Logic này không hiệu quả nhưng nó tương thích với các thuật toán cũ
    is_letter_correct_all_words = ''
    for real_word, transcribed_word in real_and_transcribed_words:
        if transcribed_word == '-':
             is_letter_correct = [0] * len(real_word)
        else:
            # So sánh từng ký tự theo logic cũ
            is_letter_correct = wm.getWhichLettersWereTranscribedCorrectly(
                real_word, transcribed_word)
        
        is_letter_correct_all_words += ''.join(map(str, is_letter_correct)) + ' '
    # --- KẾT THÚC PHẦN PHỤC HỒI ---

    res = {
        'real_transcript': result['recording_transcript'],
        'ipa_transcript': result['recording_ipa'],
        'pronunciation_accuracy': str(int(result['pronunciation_accuracy'])),
        'real_transcripts': ' '.join([word[0] for word in result['real_and_transcribed_words']]),
        'matched_transcripts': ' '.join([word[1] for word in result['real_and_transcribed_words']]),
        'real_transcripts_ipa': ' '.join([word[0] for word in result['real_and_transcribed_words_ipa']]),
        'matched_transcripts_ipa': ' '.join([word[1] for word in result['real_and_transcribed_words_ipa']]),
        'pair_accuracy_category': ' '.join(map(str, result['pronunciation_categories'])),
        'start_time': result['start_time'],
        'end_time': result['end_time'],
        'is_letter_correct_all_words': is_letter_correct_all_words.strip()
    }

    return json.dumps(res)


def audioread_load(path, offset=0.0, duration=None, dtype=np.float32):
    y = []
    with audioread.audio_open(path) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels
        s_start = int(np.round(sr_native * offset)) * n_channels
        if duration is None: s_end = np.inf
        else: s_end = s_start + (int(np.round(sr_native * duration)) * n_channels)
        n = 0
        for frame in input_file:
            frame = buf_to_float(frame, dtype=dtype)
            n_prev, n = n, n + len(frame)
            if n < s_start: continue
            if s_end < n_prev: break
            if s_end < n: frame = frame[: s_end - n_prev]
            if n_prev <= s_start <= n: frame = frame[(s_start - n_prev):]
            y.append(frame)
    if y:
        y = np.concatenate(y)
        if n_channels > 1: y = y.reshape((-1, n_channels)).T
    else: y = np.empty(0, dtype=dtype)
    return y, sr_native


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))
    fmt = f"<i{n_bytes}"
    return scale * np.frombuffer(x, fmt).astype(dtype)