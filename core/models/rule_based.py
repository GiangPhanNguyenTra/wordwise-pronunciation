import core.models.interfaces as interfaces
import nltk
from nltk.corpus import cmudict
import re
from g2p_en import G2p
from functools import lru_cache


nltk.download("averaged_perceptron_tagger", quiet=True)


TOKEN_REGEX = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;:]")


def smart_join(tokens):
    s = ""
    for t in tokens:
        if t in ".,!?;:":
            s = s.rstrip() + t
        else:
            s += " " + t
    return s.strip()


HETERONYM_RULES = {
    "record":  {"NN": "EH1", "VB": "AO1", "VBP": "AO1", "VBZ": "AO1", "VBD": "AO1", "VBG": "AO1"},
    "object":  {"NN": "EH1", "VB": "AE2"},
    "subject": {"NN": "AH1", "VB": "EH2"},
    "project": {"NN": "AA1", "VB": "EH2"},
    "produce": {"NN": "OW1", "VB": "UW1"},
    "permit":  {"NN": "ER1", "VB": "IH2"},
    "contract":{"NN": "AA1", "VB": "AE2"},
    "contest": {"NN": "AA1", "VB": "EH2"},
    "survey":  {"NN": "ER1", "VB": "EY2"},
    "increase":{"NN": "IH1", "VB": "IY2"},
    "decrease":{"NN": "IY1", "VB": "IY2"}
}


try:
    g2p_model = G2p()
except:
    g2p_model = None




class EngPhonemConverter(interfaces.ITextToPhonemModel):


    def __init__(self):
        super().__init__()
        try:
            self.cmu = cmudict.dict()
        except:
            self.cmu = None


        self.map = {
            'AA':'ɑ','AE':'æ','AO':'ɔ','AW':'aʊ','AY':'aɪ','B':'b','CH':'tʃ',
            'D':'d','DH':'ð','EH':'ɛ','EY':'eɪ','F':'f','G':'g','HH':'h','IH':'ɪ',
            'IY':'i','JH':'dʒ','K':'k','L':'l','M':'m','N':'n','NG':'ŋ','OW':'oʊ',
            'OY':'ɔɪ','P':'p','R':'r','S':'s','SH':'ʃ','T':'t','TH':'θ','UH':'ʊ',
            'UW':'u','V':'v','W':'w','Y':'j','Z':'z','ZH':'ʒ','AX':'ə'
        }


        if g2p_model:
            self.g2p = lru_cache(maxsize=10000)(g2p_model.__call__)
        else:
            self.g2p = None


    def _tokenize(self, s):
        return TOKEN_REGEX.findall(s)


    def _map_phone(self, ph):
        m = re.match(r"^([A-Z]+)([0-2])?$", ph)
        if not m:
            return ph.lower()
        base, stress = m.group(1), m.group(2)
        mark = "ˈ" if stress == "1" else ""
        if base == "ER":
            ipa = "ɜr" if stress in ("1","2") else "ər"
            return mark + ipa
        if base == "AH":
            ipa = "ə" if stress == "0" else "ʌ"
            return mark + ipa
        ipa = self.map.get(base, base.lower())
        return mark + ipa


    def _map_g2p(self, phones):
        out = ""
        for p in phones:
            if re.match(r"^[A-Z]+[0-2]?$", p):
                out += self._map_phone(p)
        return out


    def _check_rule(self, pron, code):
        for ph in pron:
            if code in ph:
                return True
        return False


    def _choose_pron(self, word, pos, prons):
        if word in HETERONYM_RULES:
            for tag, code in HETERONYM_RULES[word].items():
                if pos.startswith(tag):
                    for p in prons:
                        if self._check_rule(p, code):
                            return p
        return prons[0]


    def convertToPhonem(self, s):
        tokens = self._tokenize(s)


        alpha_tokens = [t for t in tokens if t[0].isalpha()]
        pos_tags = nltk.pos_tag(alpha_tokens)


        pos_by_index = {}
        j = 0
        for i, tk in enumerate(tokens):
            if tk[0].isalpha():
                pos_by_index[i] = pos_tags[j][1]
                j += 1


        out = []
        for i, tk in enumerate(tokens):
            if tk in ".,!?;:":
                out.append(tk)
                continue


            w = tk.lower()
            pos = pos_by_index.get(i, "")


            if self.cmu and w in self.cmu:
                pr = self._choose_pron(w, pos, self.cmu[w])
                ipa = "".join(self._map_phone(p) for p in pr)
                ipa = ipa.replace("ˈ","").replace("ˌ","")
                out.append(ipa)
                continue


            if self.g2p:
                ipa = self._map_g2p(self.g2p(w))
                ipa = ipa.replace("ˈ","").replace("ˌ","")
                out.append(ipa)
                continue


            out.append(tk)


        return smart_join(out)




_converter = None
def get_phonem_converter(language: str):
    global _converter
    if _converter is None:
        _converter = EngPhonemConverter()
    return _converter
