import core.models.interfaces as interfaces
import eng_to_ipa

def get_phonem_converter(language: str):
    return EngPhonemConverter()

class EngPhonemConverter(interfaces.ITextToPhonemModel):
    def __init__(self,) -> None:
        super().__init__()

    def convertToPhonem(self, sentence: str) -> str:
        phonem_representation = eng_to_ipa.convert(sentence)
        phonem_representation = phonem_representation.replace('*', '')
        return phonem_representation