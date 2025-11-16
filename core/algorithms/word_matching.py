import numpy as np
from string import punctuation
from dtwalign import dtw_from_distance_matrix
import time
from typing import List, Tuple
from . import word_metrics

def get_word_distance_matrix(words_estimated: list, words_real: list) -> np.ndarray:
    number_of_real_words = len(words_real)
    number_of_estimated_words = len(words_estimated)

    word_distance_matrix = np.zeros((number_of_estimated_words + 1, number_of_real_words))
    for idx_estimated in range(number_of_estimated_words):
        for idx_real in range(number_of_real_words):
            word_distance_matrix[idx_estimated, idx_real] = word_metrics.edit_distance_python(
                words_estimated[idx_estimated], words_real[idx_real])

    for idx_real in range(number_of_real_words):
        word_distance_matrix[number_of_estimated_words, idx_real] = len(words_real[idx_real])
    return word_distance_matrix

def get_resulting_string(mapped_indices: np.ndarray, words_estimated: list, words_real: list) -> Tuple[List,List]:
    mapped_words = []
    mapped_words_indices = []
    WORD_NOT_FOUND_TOKEN = '-'
    number_of_real_words = len(words_real)
    for word_idx in range(number_of_real_words):
        position_of_real_word_indices = np.where(mapped_indices == word_idx)[0].astype(int)

        if len(position_of_real_word_indices) == 0:
            mapped_words.append(WORD_NOT_FOUND_TOKEN)
            mapped_words_indices.append(-1)
            continue

        if len(position_of_real_word_indices) == 1:
            est_idx = position_of_real_word_indices[0]
            if est_idx < len(words_estimated):
                mapped_words.append(words_estimated[est_idx])
                mapped_words_indices.append(est_idx)
            else:
                mapped_words.append(WORD_NOT_FOUND_TOKEN)
                mapped_words_indices.append(-1)
            continue

        if len(position_of_real_word_indices) > 1:
            error = 99999
            best_possible_combination = ''
            best_possible_idx = -1
            for single_word_idx in position_of_real_word_indices:
                if single_word_idx >= len(words_estimated):
                    continue
                error_word = word_metrics.edit_distance_python(
                    words_estimated[single_word_idx], words_real[word_idx])
                if error_word < error:
                    error = error_word
                    best_possible_combination = words_estimated[single_word_idx]
                    best_possible_idx = single_word_idx
            
            mapped_words.append(best_possible_combination)
            mapped_words_indices.append(best_possible_idx)
            continue

    return mapped_words, mapped_words_indices

def get_best_mapped_words(words_estimated: list, words_real: list, use_dtw:bool = True) -> list:
    if not words_estimated or not words_real:
        return ['-'] * len(words_real), [-1] * len(words_real)
        
    word_distance_matrix = get_word_distance_matrix(words_estimated, words_real)
    
    alignment = dtw_from_distance_matrix(word_distance_matrix.T)
    mapped_indices = alignment.get_warping_path()

    mapped_words, mapped_words_indices = get_resulting_string(
        mapped_indices, words_estimated, words_real)

    return mapped_words, mapped_words_indices

def getWhichLettersWereTranscribedCorrectly(real_word, transcribed_word):
    real = real_word or ""
    trans = transcribed_word or ""
    n = len(real)
    m = len(trans)

    if n == 0:
        return []

    # DP edit-distance alignment
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        dp[i, 0] = i
    for j in range(1, m + 1):
        dp[0, j] = j

    for i in range(1, n + 1):
        rch = real[i - 1].lower()
        for j in range(1, m + 1):
            tch = trans[j - 1].lower()
            cost_sub = 0 if rch == tch else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,          # delete
                dp[i, j - 1] + 1,          # insert
                dp[i - 1, j - 1] + cost_sub  # substitute / match
            )

    # backtrack để biết mỗi chữ real[i] align với trans[j] nào
    aligned_j_for_i = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            rch = real[i - 1].lower()
            tch = trans[j - 1].lower()
            cost_sub = 0 if rch == tch else 1
            if dp[i, j] == dp[i - 1, j - 1] + cost_sub:
                aligned_j_for_i[i - 1] = j - 1  # i-1 của real map với j-1 của trans (có thể đúng hoặc sai)
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            j -= 1
        else:
            if i > 0:
                i -= 1
            elif j > 0:
                j -= 1

    is_letter_correct = [0] * n
    for idx, ch in enumerate(real):
        ch_low = ch.lower()
        if ch in punctuation:
            is_letter_correct[idx] = 1
            continue
        mapped_j = aligned_j_for_i[idx]
        if mapped_j is not None and 0 <= mapped_j < m:
            if ch_low == trans[mapped_j].lower():
                is_letter_correct[idx] = 1
            else:
                is_letter_correct[idx] = 0
        else:
            is_letter_correct[idx] = 0

    return is_letter_correct

    # Sửa lỗi immutable string: chuyển transcribed_word thành list để có thể sửa đổi
    transcribed_list = list(transcribed_word)
    is_letter_correct = [None] * len(real_word)    
    for idx, letter in enumerate(real_word):   
        letter = letter.lower()
        if idx < len(transcribed_list):
            transcribed_char = transcribed_list[idx].lower()
            if letter == transcribed_char or letter in punctuation:
                is_letter_correct[idx] = 1
            else:
                is_letter_correct[idx] = 0
        else: # Trường hợp từ được transcribe ngắn hơn từ thật
            is_letter_correct[idx] = 0
            
    return is_letter_correct