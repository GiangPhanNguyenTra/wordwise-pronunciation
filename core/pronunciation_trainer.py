import torch
import numpy as np
import time
from string import punctuation
import re
import librosa

from core.models import loader as mo
from core.models import rule_based
from core.models import interfaces as mi
from core.algorithms import word_matching as wm
from core.algorithms import word_metrics
from core.algorithms import phonetic_scoring

DIGIT2WORD = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}

def clean_and_normalize_text(text: str) -> str:
    if not text: return ""
    text = text.lower()
    text = re.sub(r'\boh\b', 'zero', text)
    text = re.sub(r'(?<=\d)(?=\d)', ' ', text)
    
    # Fix các lỗi nhận diện số thường gặp (Whisper Hallucinations)
    text = text.replace('zhaorou', 'zero').replace('sui', 'three').replace('wuang', 'one')
    
    normalized_words = []
    for word in text.split():
        if any(char.isdigit() for char in word):
            new_word = ""
            for char in word:
                if char in DIGIT2WORD: new_word += DIGIT2WORD[char] + " "
                elif char.isalpha(): new_word += char
            normalized_words.append(new_word)
        else:
            normalized_words.append(word)
    
    text = " ".join(normalized_words)
    for p in punctuation: text = text.replace(p, " ")
    return re.sub(r'\s+', ' ', text).strip()

def getTrainer(language: str):
    asr_model = mo.getASRModel(language, use_whisper=True)
    phonem_converter = rule_based.get_phonem_converter(language)
    return PronunciationTrainer(asr_model, phonem_converter)

class PronunciationTrainer:
    sampling_rate = 16000
    categories_thresholds = np.array([80, 60, 40])

    def __init__(self, asr_model: mi.IASRModel, word_to_ipa_coverter: mi.ITextToPhonemModel) -> None:
        self.asr_model = asr_model
        self.ipa_converter = word_to_ipa_coverter

    def processAudioForGivenText(self, recordedAudio: torch.Tensor, real_text: str):
        # 1. Preprocess & Transcribe
        raw_audio_np = recordedAudio.detach().cpu().numpy()[0]
        self.asr_model.processAudio(self.preprocessAudio(recordedAudio))
        
        recording_transcript = self.asr_model.getTranscript()
        word_locations = self.asr_model.getWordLocations()
        
        # 2. Text Normalization
        real_text_norm = clean_and_normalize_text(real_text)
        recording_transcript_norm = clean_and_normalize_text(recording_transcript)

        # 3. IPA Conversion
        real_text_ipa = self.ipa_converter.convertToPhonem(real_text_norm)
        recording_transcript_ipa = self.ipa_converter.convertToPhonem(recording_transcript_norm)

        words_real = real_text_norm.split()
        words_estimated = recording_transcript_norm.split()
        words_real_ipa = real_text_ipa.split()
        words_estimated_ipa = recording_transcript_ipa.split()

        # 4. Alignment
        mapped_words_ipa, mapped_indices_ipa = wm.get_best_mapped_words(words_estimated_ipa, words_real_ipa)

        real_and_transcribed_words = []
        real_and_transcribed_words_ipa = []
        mapped_indices = []
        words_found_count = 0

        for i, real_word_ipa in enumerate(words_real_ipa):
            transcribed_word_ipa = mapped_words_ipa[i]
            real_and_transcribed_words_ipa.append((real_word_ipa, transcribed_word_ipa))

            real_word_text = words_real[i] if i < len(words_real) else '-'
            est_idx = mapped_indices_ipa[i]
            
            if est_idx != -1 and est_idx < len(words_estimated):
                transcribed_word_text = words_estimated[est_idx]
                words_found_count += 1
            else:
                transcribed_word_text = '-'

            real_and_transcribed_words.append((real_word_text, transcribed_word_text))
            mapped_indices.append(est_idx)

        # 5. Scoring
        pron_accuracy, per_word_accuracy = self.getPronunciationAccuracy(real_and_transcribed_words_ipa)
        
        total_real = len(words_real)
        completeness_score = (words_found_count / total_real) * 100.0 if total_real > 0 else 0.0
        
        fluency_score = self.calculateFluency(word_locations, mapped_indices)
        prosody_score = self.calculateProsody(raw_audio_np, word_locations, mapped_indices)

        # Missing Penalty Logic
        missing_ratio = 1.0 - (words_found_count / max(1, total_real))
        penalty_factor = 1.0
        if missing_ratio > 0.3:
            penalty_factor = 0.7 
        elif missing_ratio > 0.1:
            penalty_factor = 0.9

        # Final Weighted Score
        raw_weighted = (pron_accuracy * 0.45) + (completeness_score * 0.35) + (fluency_score * 0.10) + (prosody_score * 0.10)
        overall_total = (0.3677 * pron_accuracy) + (0.0000 * completeness_score) + (0.1322 * fluency_score) + (0.2250 * prosody_score) + 8.3311
        
        # Use simple mapping logic as fallback or blend
        # overall_total = (raw_weighted * 0.7 + 30.0) * penalty_factor
        overall_total = np.clip(np.round(overall_total), 0, 100)

        # Timestamp Calculation (Modified for UI)
        start_time, end_time = self.getWordLocationsFromRecordInSeconds(word_locations, mapped_indices)

        return {
            'recording_transcript': recording_transcript_norm,
            'recording_ipa': recording_transcript_ipa,
            'real_and_transcribed_words': real_and_transcribed_words,
            'real_and_transcribed_words_ipa': real_and_transcribed_words_ipa,
            'pronunciation_accuracy': pron_accuracy,
            'completeness_score': completeness_score,
            'fluency_score': fluency_score,
            'prosody_score': prosody_score,
            'overall_total': overall_total,
            'pronunciation_categories': self.getWordsPronunciationCategory(per_word_accuracy),
            'start_time': start_time,
            'end_time': end_time
        }

    def getPronunciationAccuracy(self, real_and_transcribed_words_ipa) -> tuple:
        return phonetic_scoring.calculate_sentence_accuracy_with_difficulty(real_and_transcribed_words_ipa, strict_mode=False)

    def calculateProsody(self, audio_np, word_locations, mapped_indices) -> float:
        if len(audio_np) < 512: return 50.0
        rms = librosa.feature.rms(y=audio_np)[0]
        cv = np.std(rms) / (np.mean(rms) + 1e-6)
        score = np.clip((cv - 0.2) / (0.8 - 0.2) * 100, 50, 100)
        return score

    def calculateFluency(self, word_locations, mapped_indices) -> float:
        valid_idx = [idx for idx in mapped_indices if idx != -1 and idx < len(word_locations)]
        if len(valid_idx) < 2: return 85.0
        
        gaps = 0.0
        for i in range(len(valid_idx) - 1):
            curr_end = word_locations[valid_idx[i]]['end_ts']
            next_start = word_locations[valid_idx[i+1]]['start_ts']
            g = (next_start - curr_end) / self.sampling_rate
            if g > 0.25: gaps += (g - 0.25)
        
        gap_score = max(0.0, 100.0 - (gaps * 20.0))
        
        start_t = word_locations[valid_idx[0]]['start_ts']
        end_t = word_locations[valid_idx[-1]]['end_ts']
        dur = (end_t - start_t) / self.sampling_rate
        wpm = (len(valid_idx) / dur) * 60.0 if dur > 0 else 0
        wpm_score = np.clip((wpm - 50) / (130 - 50) * 100, 40, 100)
        
        return (gap_score * 0.7) + (wpm_score * 0.3)

    def getWordLocationsFromRecordInSeconds(self, word_locations, mapped_indices):
        start_time, end_time = [], []
        num_locs = len(word_locations)
        for idx in mapped_indices:
            if 0 <= idx < num_locs:
                # --- UI OPTIMIZATION: THÊM PADDING ---
                # Lùi start lại 0.05s và tăng end lên 0.1s để nghe trọn vẹn từ
                s = max(0.0, (word_locations[idx]['start_ts'] / self.sampling_rate) - 0.05)
                e = (word_locations[idx]['end_ts'] / self.sampling_rate) + 0.1
                
                # Format string để JS dễ đọc
                start_time.append(f"{s:.3f}")
                end_time.append(f"{e:.3f}")
            else:
                start_time.append("0.0")
                end_time.append("0.0")
        return ' '.join(start_time), ' '.join(end_time)

    def getWordsPronunciationCategory(self, accuracies):
        return [np.argmin(np.abs(self.categories_thresholds - acc)) for acc in accuracies]

    def preprocessAudio(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio - torch.mean(audio)
        if torch.max(torch.abs(audio)) > 0: audio = audio / torch.max(torch.abs(audio))
        return audio