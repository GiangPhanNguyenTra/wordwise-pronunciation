"""
Enhanced phonetic scoring module with feature-based similarity.
This module provides improved pronunciation accuracy calculation by considering
phonetic features and similarity between sounds.
"""

import numpy as np
from typing import List, Tuple


# IPA Phoneme definitions and features
VOWELS = set([
    'i', 'ɪ', 'e', 'ɛ', 'æ', 'ɑ', 'ɔ', 'o', 'ʊ', 'u', 'ʌ', 'ə', 'ɜ', 'ɐ',
    'aɪ', 'aʊ', 'eɪ', 'oʊ', 'ɔɪ', 'ɪə', 'ɛə', 'ʊə'
])

CONSONANTS = set([
    'p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ',
    'tʃ', 'dʒ', 'h', 'm', 'n', 'ŋ', 'l', 'r', 'w', 'j'
])

# Voiced-voiceless pairs (switching these is less severe)
VOICED_PAIRS = {
    'p': 'b', 'b': 'p',
    't': 'd', 'd': 't',
    'k': 'g', 'g': 'k',
    'f': 'v', 'v': 'f',
    'θ': 'ð', 'ð': 'θ',
    's': 'z', 'z': 's',
    'ʃ': 'ʒ', 'ʒ': 'ʃ',
    'tʃ': 'dʒ', 'dʒ': 'tʃ'
}

# Similar vowel groups (confusing these is common)
# Each group has a similarity level
VERY_SIMILAR_VOWELS = [
    {'i', 'ɪ'},           # beat/bit - high front
    {'e', 'ɛ'},           # bait/bet - mid front  
    {'u', 'ʊ'},           # boot/book - high back
    {'ə', 'ʌ'},           # schwa/wedge - mid central
    {'o', 'oʊ'},          # close-mid back
]

SOMEWHAT_SIMILAR_VOWELS = [
    {'ɑ', 'ɔ'},           # father/thought - back
    {'ɑ', 'ʌ'},           # father/but
    {'æ', 'ɛ'},           # cat/bet - front
    {'aɪ', 'aʊ'},         # bite/bout - diphthongs
    {'eɪ', 'e'},          # bait/bet
    {'oʊ', 'ɔ'},          # boat/bought
]

RELATED_VOWELS = [
    {'i', 'e', 'ɛ'},      # front vowels
    {'u', 'o', 'ɔ'},      # back rounded
    {'ɑ', 'æ', 'ʌ'},      # open/mid
]

# Similar consonant groups with different similarity levels
VERY_SIMILAR_CONSONANTS = [
    {'m', 'n'},           # nasals
    {'f', 'θ'},           # voiceless fricatives (often confused)
    {'v', 'ð'},           # voiced fricatives
    {'s', 'ʃ'},           # sibilants
    {'z', 'ʒ'},           # voiced sibilants
]

SOMEWHAT_SIMILAR_CONSONANTS = [
    {'l', 'r'},           # liquids (very commonly confused)
    {'p', 't', 'k'},      # voiceless stops
    {'b', 'd', 'g'},      # voiced stops
    {'tʃ', 'ʃ'},          # post-alveolar
    {'dʒ', 'ʒ'},          # voiced post-alveolar
    {'n', 'ŋ'},           # nasals (different place)
]


def tokenize_ipa(ipa_string: str) -> List[str]:
    """
    Tokenize IPA string into phoneme units.
    Handles multi-character phonemes correctly.
    
    Args:
        ipa_string: IPA representation of a word
        
    Returns:
        List of phoneme tokens
    """
    # Multi-character phonemes that should be treated as single units
    multi_char_phonemes = ['tʃ', 'dʒ', 'aɪ', 'aʊ', 'eɪ', 'oʊ', 'ɔɪ', 'ɪə', 'ɛə', 'ʊə', 'ɜr', 'ər']
    
    tokens = []
    i = 0
    ipa_clean = ipa_string.replace('ˈ', '').replace('ˌ', '')  # Remove stress marks
    
    while i < len(ipa_clean):
        # Try to match multi-char phonemes first
        matched = False
        for phoneme in multi_char_phonemes:
            if ipa_clean[i:i+len(phoneme)] == phoneme:
                tokens.append(phoneme)
                i += len(phoneme)
                matched = True
                break
        
        if not matched:
            # Single character phoneme
            if ipa_clean[i] not in [' ', '.', ',', '!', '?', ';', ':']:
                tokens.append(ipa_clean[i])
            i += 1
    
    return tokens


def get_phoneme_similarity(ph1: str, ph2: str) -> float:
    """
    Calculate similarity between two phonemes based on phonetic features.
    Returns value between 0.0 (completely different) and 1.0 (identical).
    
    Uses graduated similarity levels for more accurate pronunciation scoring.
    
    Args:
        ph1: First phoneme
        ph2: Second phoneme
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if ph1 == ph2:
        return 1.0
    
    # Both are vowels
    if ph1 in VOWELS and ph2 in VOWELS:
        # Check very similar vowels (minimal difference)
        for group in VERY_SIMILAR_VOWELS:
            if ph1 in group and ph2 in group:
                return 0.75  # Very similar vowels - minor error
        
        # Check somewhat similar vowels (noticeable but related)
        for group in SOMEWHAT_SIMILAR_VOWELS:
            if ph1 in group and ph2 in group:
                return 0.50  # Somewhat similar vowels
        
        # Check related vowels (same general area)
        for group in RELATED_VOWELS:
            if ph1 in group and ph2 in group:
                return 0.30  # Related vowels
        
        return 0.15  # Completely different vowels
    
    # Both are consonants
    if ph1 in CONSONANTS and ph2 in CONSONANTS:
        # Voiced-voiceless pair (very similar - common learner error)
        if VOICED_PAIRS.get(ph1) == ph2:
            return 0.85  # Very minor difference
        
        # Very similar consonants
        for group in VERY_SIMILAR_CONSONANTS:
            if ph1 in group and ph2 in group:
                return 0.65  # Similar consonants
        
        # Somewhat similar consonants
        for group in SOMEWHAT_SIMILAR_CONSONANTS:
            if ph1 in group and ph2 in group:
                return 0.45  # Somewhat similar
        
        return 0.20  # Different consonants
    
    # One is vowel, one is consonant - very different (major error)
    return 0.0


def weighted_edit_distance(seq1: List[str], seq2: List[str]) -> float:
    """
    Calculate weighted edit distance between two phoneme sequences.
    Uses phonetic similarity to weight substitution costs.
    
    Args:
        seq1: First sequence of phonemes
        seq2: Second sequence of phonemes
        
    Returns:
        Weighted edit distance
    """
    n = len(seq1)
    m = len(seq2)
    
    # Create DP table
    dp = np.zeros((n + 1, m + 1))
    
    # Initialize base cases
    for i in range(n + 1):
        dp[i, 0] = i  # Cost of deleting all characters from seq1
    for j in range(m + 1):
        dp[0, j] = j  # Cost of inserting all characters to match seq2
    
    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i-1] == seq2[j-1]:
                # Perfect match - no cost
                substitution_cost = 0
            else:
                # Weighted substitution based on phonetic similarity
                similarity = get_phoneme_similarity(seq1[i-1], seq2[j-1])
                substitution_cost = 1.0 - similarity
            
            dp[i, j] = min(
                dp[i-1, j] + 1.0,                    # Deletion
                dp[i, j-1] + 1.0,                    # Insertion
                dp[i-1, j-1] + substitution_cost     # Substitution
            )
    
    return dp[n, m]


def calculate_word_difficulty(phonemes: List[str]) -> float:
    """
    Estimate pronunciation difficulty of a word based on phoneme complexity.
    More difficult words should be weighted higher in overall scoring.
    
    Args:
        phonemes: List of phonemes in the word
        
    Returns:
        Difficulty multiplier (1.0 = normal, higher = more difficult)
    """
    if not phonemes:
        return 1.0
    
    difficulty = 1.0
    
    # Longer words are harder
    if len(phonemes) > 6:
        difficulty += 0.3
    elif len(phonemes) > 4:
        difficulty += 0.15
    
    # Consonant clusters are harder
    consonant_cluster = 0
    for i in range(len(phonemes) - 1):
        if phonemes[i] in CONSONANTS and phonemes[i+1] in CONSONANTS:
            consonant_cluster += 1
    difficulty += consonant_cluster * 0.2
    
    # Complex vowels (diphthongs) are harder
    complex_vowels = sum(1 for p in phonemes if len(p) > 1 and p in VOWELS)
    difficulty += complex_vowels * 0.15
    
    # Difficult consonants
    difficult_consonants = {'θ', 'ð', 'ʒ', 'ŋ', 'r'}
    difficult_count = sum(1 for p in phonemes if p in difficult_consonants)
    difficulty += difficult_count * 0.2
    
    return min(difficulty, 2.0)  # Cap at 2.0


def calculate_word_accuracy(real_ipa: str, transcribed_ipa: str, strict_mode: bool = False) -> float:
    """
    Calculate pronunciation accuracy for a single word using enhanced phonetic scoring.
    
    Args:
        real_ipa: Expected IPA pronunciation
        transcribed_ipa: Actual transcribed IPA pronunciation
        strict_mode: If True, apply stricter scoring (harder to get 100%)
        
    Returns:
        Accuracy score (0.0 to 100.0)
    """
    # Tokenize into phonemes
    real_phonemes = tokenize_ipa(real_ipa)
    trans_phonemes = tokenize_ipa(transcribed_ipa)
    
    len_real = len(real_phonemes)
    len_trans = len(trans_phonemes)
    
    if len_real == 0:
        return 100.0 if len_trans == 0 else 0.0
    
    # Calculate weighted edit distance
    distance = weighted_edit_distance(real_phonemes, trans_phonemes)
    
    # Use max length for normalization (more forgiving for length differences)
    max_len = max(len_real, len_trans, 1)
    
    # Calculate similarity
    similarity = max(0.0, (max_len - distance) / max_len)
    
    if strict_mode:
        # Apply non-linear scaling to make 100% harder to achieve
        # Use power function to make the curve more strict
        # similarity^1.5 means: 0.9 -> 0.85, 0.95 -> 0.93, 1.0 -> 1.0
        similarity = similarity ** 1.2
    
    # Convert to percentage
    accuracy = similarity * 100.0
    
    return accuracy


def calculate_sentence_accuracy(
    real_and_transcribed_words_ipa: List[Tuple[str, str]],
    use_phoneme_weighting: bool = True,
    strict_mode: bool = False
) -> Tuple[float, List[float]]:
    """
    Calculate overall pronunciation accuracy for a sentence.
    
    Args:
        real_and_transcribed_words_ipa: List of (real_ipa, transcribed_ipa) tuples
        use_phoneme_weighting: If True, weight by phoneme count; if False, equal word weights
        strict_mode: If True, apply stricter scoring
        
    Returns:
        Tuple of (overall_accuracy, per_word_accuracies)
    """
    per_word_accuracy = []
    
    for real_ipa, transcribed_ipa in real_and_transcribed_words_ipa:
        # Remove punctuation
        real_ipa_clean = ''.join(c for c in real_ipa if c not in '.,!?;:')
        trans_ipa_clean = ''.join(c for c in transcribed_ipa if c not in '.,!?;:')
        
        word_acc = calculate_word_accuracy(real_ipa_clean, trans_ipa_clean, strict_mode=strict_mode)
        per_word_accuracy.append(word_acc)
    
    if not per_word_accuracy:
        return 0.0, []
    
    if use_phoneme_weighting:
        # Weight by phoneme count
        total_weighted = 0.0
        total_phonemes = 0
        
        for (real_ipa, _), acc in zip(real_and_transcribed_words_ipa, per_word_accuracy):
            real_clean = ''.join(c for c in real_ipa if c not in '.,!?;:')
            phoneme_count = len(tokenize_ipa(real_clean))
            total_weighted += acc * phoneme_count
            total_phonemes += phoneme_count
        
        overall = total_weighted / max(total_phonemes, 1)
    else:
        # Simple average
        overall = np.mean(per_word_accuracy)
    
    return np.round(overall), per_word_accuracy


def calculate_sentence_accuracy_with_difficulty(
    real_and_transcribed_words_ipa: List[Tuple[str, str]],
    strict_mode: bool = True
) -> Tuple[float, List[float]]:
    """
    Calculate sentence accuracy with word difficulty weighting.
    Harder words contribute more to the overall score.
    
    Args:
        real_and_transcribed_words_ipa: List of (real_ipa, transcribed_ipa) tuples
        strict_mode: If True, apply stricter scoring
        
    Returns:
        Tuple of (overall_accuracy, per_word_accuracies)
    """
    per_word_accuracy = []
    per_word_difficulty = []
    
    for real_ipa, transcribed_ipa in real_and_transcribed_words_ipa:
        # Remove punctuation
        real_ipa_clean = ''.join(c for c in real_ipa if c not in '.,!?;:')
        trans_ipa_clean = ''.join(c for c in transcribed_ipa if c not in '.,!?;:')
        
        # Calculate accuracy
        word_acc = calculate_word_accuracy(real_ipa_clean, trans_ipa_clean, strict_mode=strict_mode)
        per_word_accuracy.append(word_acc)
        
        # Calculate difficulty
        real_phonemes = tokenize_ipa(real_ipa_clean)
        difficulty = calculate_word_difficulty(real_phonemes)
        per_word_difficulty.append(difficulty)
    
    if not per_word_accuracy:
        return 0.0, []
    
    # Weight by both phoneme count AND difficulty
    total_weighted = 0.0
    total_weight = 0.0
    
    for (real_ipa, _), acc, diff in zip(real_and_transcribed_words_ipa, per_word_accuracy, per_word_difficulty):
        real_clean = ''.join(c for c in real_ipa if c not in '.,!?;:')
        phoneme_count = len(tokenize_ipa(real_clean))
        
        # Combined weight: phoneme count × difficulty
        weight = phoneme_count * diff
        total_weighted += acc * weight
        total_weight += weight
    
    overall = total_weighted / max(total_weight, 1)
    
    return np.round(overall), per_word_accuracy


def calculate_accuracy_with_penalty(
    real_and_transcribed_words_ipa: List[Tuple[str, str]],
    missing_word_penalty: float = 0.7,
    strict_mode: bool = True,
    use_difficulty_weighting: bool = True
) -> Tuple[float, List[float]]:
    """
    Calculate accuracy with additional penalty for missing or extra words.
    
    Args:
        real_and_transcribed_words_ipa: List of (real_ipa, transcribed_ipa) tuples
        missing_word_penalty: Penalty multiplier for missing words (0.0-1.0)
        strict_mode: If True, apply stricter scoring (recommended for better human correlation)
        use_difficulty_weighting: If True, weight by word difficulty
        
    Returns:
        Tuple of (overall_accuracy, per_word_accuracies)
    """
    if use_difficulty_weighting:
        overall, per_word = calculate_sentence_accuracy_with_difficulty(
            real_and_transcribed_words_ipa,
            strict_mode=strict_mode
        )
    else:
        overall, per_word = calculate_sentence_accuracy(
            real_and_transcribed_words_ipa, 
            use_phoneme_weighting=True,
            strict_mode=strict_mode
        )
    
    # Count missing words (marked as '-')
    missing_count = sum(1 for _, trans in real_and_transcribed_words_ipa if trans == '-')
    total_words = len(real_and_transcribed_words_ipa)
    
    if total_words > 0 and missing_count > 0:
        missing_ratio = missing_count / total_words
        # Apply stronger penalty for missing words
        penalty_factor = 1.0 - (missing_ratio * missing_word_penalty)
        overall = overall * penalty_factor
    
    return np.round(overall), per_word
