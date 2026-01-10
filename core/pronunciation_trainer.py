import torch
import numpy as np
import time
from string import punctuation
import re

from core.models import loader as mo
from core.models import rule_based
from core.models import interfaces as mi
from core.algorithms import word_matching as wm
from core.algorithms import word_metrics
from core.algorithms import phonetic_scoring

DIGIT2WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
}

NUMBER_TOKEN_REGEX = re.compile(r"\d+")


def normalize_numbers_in_text(text: str) -> str:
    if not text:
        return text

    def repl(m):
        num = m.group(0)
        return " " + " ".join(DIGIT2WORD.get(c, c) for c in num) + " "

    text = NUMBER_TOKEN_REGEX.sub(repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def getTrainer(language: str):
    asr_model = mo.getASRModel(language, use_whisper=True)
    phonem_converter = rule_based.get_phonem_converter(language)
    trainer = PronunciationTrainer(asr_model, phonem_converter)
    return trainer


class PronunciationTrainer:
    sampling_rate = 16000
    categories_thresholds = np.array([80, 60, 40])

    def __init__(self, asr_model: mi.IASRModel, word_to_ipa_coverter: mi.ITextToPhonemModel, use_enhanced_scoring: bool = True) -> None:
        self.asr_model = asr_model
        self.ipa_converter = word_to_ipa_coverter
        self.use_enhanced_scoring = use_enhanced_scoring

    def processAudioForGivenText(self, recordedAudio: torch.Tensor, real_text: str):
        start_time_asr = time.time()
        self.asr_model.processAudio(self.preprocessAudio(recordedAudio))
        recording_transcript = self.asr_model.getTranscript()
        word_locations = self.asr_model.getWordLocations()
        print(f'Time for ASR: {time.time() - start_time_asr:.4f}s')

        real_text_norm = normalize_numbers_in_text(real_text)
        recording_transcript_norm = normalize_numbers_in_text(recording_transcript)

        real_text_ipa = self.ipa_converter.convertToPhonem(real_text_norm)
        recording_transcript_ipa = self.ipa_converter.convertToPhonem(recording_transcript_norm)

        words_real = real_text_norm.split()
        words_estimated = recording_transcript_norm.split()
        words_real_ipa = real_text_ipa.split()
        words_estimated_ipa = recording_transcript_ipa.split()

        start_time_match = time.time()
        mapped_words_ipa, mapped_indices_ipa = wm.get_best_mapped_words(words_estimated_ipa, words_real_ipa)
        print(f'Time for IPA matching: {time.time() - start_time_match:.4f}s')

        real_and_transcribed_words = []
        real_and_transcribed_words_ipa = []
        mapped_indices = []

        for i, real_word_ipa in enumerate(words_real_ipa):
            transcribed_word_ipa = mapped_words_ipa[i]
            real_and_transcribed_words_ipa.append((real_word_ipa, transcribed_word_ipa))

            real_word_text = words_real[i] if i < len(words_real) else '-'

            est_idx = mapped_indices_ipa[i]
            transcribed_word_text = words_estimated[est_idx] if 0 <= est_idx < len(words_estimated) else '-'

            real_and_transcribed_words.append((real_word_text, transcribed_word_text))
            mapped_indices.append(est_idx)

        pronunciation_accuracy, per_word_accuracy = self.getPronunciationAccuracy(real_and_transcribed_words_ipa)
        pronunciation_categories = self.getWordsPronunciationCategory(per_word_accuracy)
        start_time, end_time = self.getWordLocationsFromRecordInSeconds(word_locations, mapped_indices)

        result = {
            'recording_transcript': recording_transcript_norm,
            'recording_ipa': recording_transcript_ipa,
            'real_and_transcribed_words': real_and_transcribed_words,
            'real_and_transcribed_words_ipa': real_and_transcribed_words_ipa,
            'pronunciation_accuracy': pronunciation_accuracy,
            'pronunciation_categories': pronunciation_categories,
            'start_time': start_time,
            'end_time': end_time
        }
        return result

    def getPronunciationAccuracy(self, real_and_transcribed_words_ipa) -> tuple:
        if self.use_enhanced_scoring:
            # Use enhanced phonetic-based scoring with strict mode and difficulty weighting
            overall_accuracy, per_word_accuracy = phonetic_scoring.calculate_accuracy_with_penalty(
                real_and_transcribed_words_ipa,
                missing_word_penalty=0.7,  # Increased penalty for missing words
                strict_mode=True,  # Enable strict mode for better human score correlation
                use_difficulty_weighting=True  # Weight harder words more
            )
            return overall_accuracy, per_word_accuracy
        else:
            # Use original edit distance based scoring
            per_word_accuracy = []
            total_similarity = 0.0
            total_phonemes = 0.0

            for real_ipa, transcribed_ipa in real_and_transcribed_words_ipa:
                real_ipa = self.removePunctuation(real_ipa)
                transcribed_ipa = self.removePunctuation(transcribed_ipa)

                len_real = len(real_ipa)
                if len_real == 0:
                    word_acc = 100.0 if not transcribed_ipa else 0.0
                else:
                    distance = word_metrics.edit_distance_python(real_ipa, transcribed_ipa)
                    denom = max(len_real, len(transcribed_ipa), 1)
                    similarity = (denom - distance) / denom
                    word_acc = max(0.0, similarity * 100.0)

                per_word_accuracy.append(word_acc)
                total_similarity += word_acc * len_real
                total_phonemes += len_real

            overall_accuracy = (total_similarity / total_phonemes) if total_phonemes > 0 else 0.0
            return np.round(overall_accuracy), per_word_accuracy

    def getWordLocationsFromRecordInSeconds(self, word_locations, mapped_indices):
        start_time, end_time = [], []
        num_word_locations = len(word_locations)
        for idx in mapped_indices:
            if 0 <= idx < num_word_locations and word_locations[idx]:
                start = word_locations[idx].get('start_ts')
                end = word_locations[idx].get('end_ts')
                start_time.append(float(start) / self.sampling_rate if start is not None else 0.0)
                end_time.append(float(end) / self.sampling_rate if end is not None else 0.0)
            else:
                start_time.append(0.0)
                end_time.append(0.0)
        return ' '.join(map(str, start_time)), ' '.join(map(str, end_time))

    def getWordsPronunciationCategory(self, accuracies):
        return [np.argmin(np.abs(self.categories_thresholds - acc)) for acc in accuracies]

    def removePunctuation(self, text: str) -> str:
        return ''.join(c for c in text if c not in punctuation)

    def preprocessAudio(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio - torch.mean(audio)
        audio_max = torch.max(torch.abs(audio))
        if audio_max > 0:
            audio = audio / audio_max
        return audio