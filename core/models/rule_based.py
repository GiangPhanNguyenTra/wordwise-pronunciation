import core.models.interfaces as interfaces
import nltk
from nltk.corpus import cmudict
import re
from g2p_en import G2p
from functools import lru_cache

nltk.download("averaged_perceptron_tagger", quiet=True)


TOKEN_REGEX = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;:]"
)

def smart_join(tokens):
    result = ""
    for t in tokens:
        if t in ".,!?;:":
            result = result.rstrip() + t
        else:
            result += " " + t
    return result.strip()


try:
    g2p_model = G2p()
except Exception:
    g2p_model = None


class EngPhonemConverter(interfaces.ITextToPhonemModel):

    def __init__(self) -> None:
        super().__init__()

        try:
            self.cmu_dict = cmudict.dict()
        except LookupError:
            self.cmu_dict = None

        self.ipa_map = {
            'AA': 'ɑ', 'AE': 'æ', 'AO': 'ɔ',
            'AW': 'aʊ', 'AY': 'aɪ',
            'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð',
            'EH': 'ɛ',
            'EY': 'eɪ',
            'F': 'f', 'G': 'g', 'HH': 'h',
            'IH': 'ɪ', 'IY': 'i',
            'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm',
            'N': 'n', 'NG': 'ŋ',
            'OW': 'oʊ', 'OY': 'ɔɪ',
            'P': 'p', 'R': 'r', 'S': 's', 'SH': 'ʃ',
            'T': 't', 'TH': 'θ',
            'UH': 'ʊ', 'UW': 'u',
            'V': 'v',
            'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
            'AX': 'ə'
        }

        if g2p_model:
            self.g2p = lru_cache(maxsize=10000)(g2p_model.__call__)
        else:
            self.g2p = None


    def _tokenize(self, sentence: str):
        return TOKEN_REGEX.findall(sentence)


    def _map_phone(self, phone: str) -> str:
        m = re.match(r"^([A-Z]+)([0-2])?$", phone)
        if not m:
            return phone.lower()

        base, stress = m.group(1), m.group(2)

        if base == "ER":
            return "ɜr" if stress in ("1", "2") else "ər"
        if base == "AH":
            return "ə" if stress == "0" else "ʌ"

        return self.ipa_map.get(base, base.lower())


    def _map_g2p_output(self, phones: list) -> str:
        ipa = ""
        for p in phones:
            if re.match(r"^[A-Z]+[0-2]?$", p):
                ipa += self._map_phone(p)
        return ipa


    def _select_pron_by_pos(self, word, pos, prons):
        if word == "record":
            if pos.startswith("NN"):  # noun
                for p in prons:
                    if any(phone.startswith("EH1") for phone in p):
                        return p
            else:  
                for p in prons:
                    if any(phone.startswith("AO1") for phone in p):
                        return p

        return prons[0]  # fallback


    def convertToPhonem(self, sentence: str) -> str:
        tokens = self._tokenize(sentence)

        # POS-tag only alphabetic tokens
        words_for_pos = [t for t in tokens if t[0].isalpha()]
        pos_tags = nltk.pos_tag(words_for_pos)

        # Map token -> POS
        pos_map = {}
        idx = 0
        for t in tokens:
            if t[0].isalpha():
                pos_map[t.lower()] = pos_tags[idx][1]
                idx += 1

        ipa_parts = []

        for token in tokens:

            if token in ".,!?;:":
                ipa_parts.append(token)
                continue

            w = token.lower()
            pos = pos_map.get(w, "")

            # CMU
            if self.cmu_dict and w in self.cmu_dict:
                prons = self.cmu_dict[w]

                selected = self._select_pron_by_pos(w, pos, prons)
                ipa_word = "".join(self._map_phone(p) for p in selected)
                ipa_parts.append(ipa_word)
                continue

            # G2P fallback
            if self.g2p:
                g2p_res = self.g2p(w)
                ipa_word = self._map_g2p_output(g2p_res)
                ipa_parts.append(ipa_word)
                continue

            ipa_parts.append(token)

        return smart_join(ipa_parts)



_converter_instance = None

def get_phonem_converter(language: str):
    global _converter_instance
    if _converter_instance is None:
        _converter_instance = EngPhonemConverter()
    return _converter_instance
