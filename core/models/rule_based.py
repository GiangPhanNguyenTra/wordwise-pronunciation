import core.models.interfaces as interfaces
import nltk
from nltk.corpus import cmudict
import re 

class EngPhonemConverter(interfaces.ITextToPhonemModel):
    def __init__(self) -> None:
        super().__init__()
        
        try:
            self.cmu_dict = cmudict.dict()
        except LookupError:
            self.cmu_dict = None
        
        self.ipa_map = {
            'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ', 
            'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'ər', 
            'EY': 'eɪ', 'F': 'f', 'G': 'g', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 
            'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 
            'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'ʃ', 
            'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'ə', 'W': 'w', 
            'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
            'AX': 'ə' 
        }

    def _map_phone(self, phone_with_stress: str) -> str:
        base_phone = re.sub(r'[012]', '', phone_with_stress)
        return self.ipa_map.get(base_phone, base_phone.lower())

    def convertToPhonem(self, sentence: str) -> str:
        if not self.cmu_dict:
            return sentence 

        ipa_parts = []
        
        for word in sentence.lower().split():
            clean_word = word.strip('.,?!')
            punctuation = word[len(clean_word):]

            if clean_word in self.cmu_dict:
                # Lấy phát âm đầu tiên
                arpabet_pron = self.cmu_dict[clean_word][0] 
                
                ipa_word = ""
                for phone in arpabet_pron:
                    ipa_word += self._map_phone(phone)
                
                ipa_parts.append(ipa_word + punctuation)
            else:
                ipa_parts.append(word)

        return " ".join(ipa_parts)

def get_phonem_converter(language: str):
    return EngPhonemConverter()