import core.models.interfaces as interfaces
import nltk
from nltk.corpus import cmudict
import re 
from g2p_en import G2p

try:
    g2p_model = G2p() 
except Exception as e:
    g2p_model = None
    print("Warning: g2p_en initialization failed.")


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
        self.g2p = g2p_model

    def _map_phone(self, phone_with_stress: str) -> str:
        base_phone = re.sub(r'[012]', '', phone_with_stress)
        return self.ipa_map.get(base_phone, base_phone.lower())

    def _map_g2p_output(self, g2p_pron_list: list) -> str:
        ipa_word = ""
        for phone in g2p_pron_list:
            ipa_word += self._map_phone(phone.upper()) 
            
        return ipa_word

    def convertToPhonem(self, sentence: str) -> str:
        if not self.cmu_dict:
            if self.g2p:
                return self._map_g2p_output(self.g2p(sentence))
            return sentence 

        ipa_parts = []
        
        for word in sentence.lower().split():
            clean_word = word.strip('.,?!')
            punctuation = word[len(clean_word):]

            if clean_word in self.cmu_dict:
                arpabet_pron = self.cmu_dict[clean_word][0] 
                ipa_word = "".join(self._map_phone(phone) for phone in arpabet_pron)
                ipa_parts.append(ipa_word + punctuation)
            
            else:
                if self.g2p:
                    g2p_phones = self.g2p(clean_word)
                    ipa_word = self._map_g2p_output(g2p_phones)
                    ipa_parts.append(ipa_word + punctuation)
                else:
                    ipa_parts.append(word) 

        return " ".join(ipa_parts)
    
def get_phonem_converter(language: str):
    return EngPhonemConverter()