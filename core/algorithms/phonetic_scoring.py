import numpy as np
from typing import List, Tuple

VOWELS = set(['i', 'ɪ', 'e', 'ɛ', 'æ', 'ɑ', 'ɔ', 'o', 'ʊ', 'u', 'ʌ', 'ə', 'ɜ', 'ɐ', 'aɪ', 'aʊ', 'eɪ', 'oʊ', 'ɔɪ', 'ɪə', 'ɛə', 'ʊə'])
CONSONANTS = set(['p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'tʃ', 'dʒ', 'h', 'm', 'n', 'ŋ', 'l', 'r', 'w', 'j'])

SIMILARITY_MAP = {
    # Nhóm cơ bản
    ('θ', 't'): 0.85, ('ð', 'd'): 0.85, 
    ('s', 'z'): 0.85, ('p', 'b'): 0.85, ('k', 'g'): 0.85, ('t', 'd'): 0.85, ('f', 'v'): 0.85,
    ('i', 'ɪ'): 0.90, ('u', 'ʊ'): 0.90, ('ə', 'ʌ'): 0.90, ('æ', 'ɛ'): 0.80, ('ʊ', 'oʊ'): 0.75,
    ('m', 'n'): 0.85, ('ŋ', 'n'): 0.80, ('l', 'r'): 0.60, ('w', 'v'): 0.75,
    
    # --- CẶP ĐẶC TRỊ TỪ DEBUG & JSON ANALYSIS ---
    ('t', 'z'): 0.50, ('d', 'z'): 0.50,
    ('v', 'ŋ'): 0.45, ('ŋ', 'z'): 0.45,
    ('m', 'k'): 0.40, 
    ('n', 'm'): 0.80, # washroom: on -> home
    ('i', 's'): 0.20, # Annie -> s (giữ 0.2 thay vì 0)
    
    # Missing Handling (Map với -) -> Tăng similarity để giảm penalty khi mất âm
    ('k', '-'): 0.30, ('t', '-'): 0.30, ('d', '-'): 0.30, ('s', '-'): 0.30,
    ('b', '-'): 0.30, ('p', '-'): 0.30, # job -> -
    
    # Fix danh tính
    ('i', 'i'): 1.0, ('u', 'u'): 1.0
}

def get_phoneme_similarity(ph1: str, ph2: str) -> float:
    if ph1 == ph2: return 1.0
    if (ph1, ph2) in SIMILARITY_MAP: return SIMILARITY_MAP[(ph1, ph2)]
    if (ph2, ph1) in SIMILARITY_MAP: return SIMILARITY_MAP[(ph2, ph1)]
    if (ph1 in VOWELS and ph2 in VOWELS): return 0.50
    if (ph1 in CONSONANTS and ph2 in CONSONANTS): return 0.45
    return 0.15 # Tăng điểm sàn similarity

def weighted_edit_distance(seq1: List[str], seq2: List[str]) -> float:
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1))
    
    # Giảm penalty cho Deletion (Mất từ) để kéo điểm sàn lên
    for i in range(n + 1): dp[i, 0] = i * 0.8  # Giảm từ 1.0 -> 0.8
    for j in range(m + 1): dp[0, j] = j * 0.3 
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 1.0 - get_phoneme_similarity(seq1[i-1], seq2[j-1])
            dp[i, j] = min(
                dp[i-1, j] + 0.8,    # Deletion cost giảm
                dp[i, j-1] + 0.3,    # Insertion
                dp[i-1, j-1] + cost
            )
    return dp[n, m]

def calculate_word_accuracy(real_ipa: str, transcribed_ipa: str) -> float:
    if transcribed_ipa == '-' or not transcribed_ipa or transcribed_ipa.strip() == '':
        return 40.0
        
    real_ph = tokenize_ipa(real_ipa)
    trans_ph = tokenize_ipa(transcribed_ipa)
    
    clean_trans = []
    if trans_ph:
        clean_trans.append(trans_ph[0])
        for p in trans_ph[1:]:
            if p != clean_trans[-1]: clean_trans.append(p)
            
    if not real_ph: return 0.0
    
    dist = weighted_edit_distance(real_ph, clean_trans)
    denom = len(real_ph)
    
    score = max(0.0, (denom - dist) / denom) * 100.0
    
    if score > 0:
        score = (score / 100.0) ** 0.5 * 100.0 
    
    return min(100.0, score)

def tokenize_ipa(ipa_string: str) -> List[str]:
    multi = ['tʃ', 'dʒ', 'aɪ', 'aʊ', 'eɪ', 'oʊ', 'ɔɪ', 'ɪə', 'ɛə', 'ʊə']
    tokens = []
    i = 0
    s = ipa_string.replace('ˈ', '').replace('ˌ', '').replace(' ', '')
    while i < len(s):
        found = False
        for p in multi:
            if s[i:i+len(p)] == p:
                tokens.append(p); i += len(p); found = True; break
        if not found:
            tokens.append(s[i]); i += 1
    return tokens

def calculate_sentence_accuracy_with_difficulty(real_and_trans_ipa, strict_mode=False):
    word_accs = []
    for r, t in real_and_trans_ipa:
        word_accs.append(calculate_word_accuracy(r, t))
    if not word_accs: return 0.0, []
    return np.round(np.mean(word_accs)), word_accs