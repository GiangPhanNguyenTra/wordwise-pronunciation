import pandas as pd
import json
import core.models.rule_based as rule_based
import random

class TextDataset():
    def __init__(self, table):
        self.table_dataframe = table
        self.number_of_samples = len(table)

    def __getitem__(self, idx):
        line = [self.table_dataframe['sentence'].iloc[idx]]
        return line

    def __len__(self):
        return self.number_of_samples

sample_folder = "./databases/"
db_words = TextDataset(pd.read_csv(sample_folder + 'data_en_words.csv', delimiter='\t'))
db_sentences = TextDataset(pd.read_csv(sample_folder + 'data_en_sentences.csv', delimiter='\t'))
ipa_converter = rule_based.get_phonem_converter('en')

def get_random_word():
    sample_idx = random.randint(0, len(db_words) - 1)
    transcript = db_words[sample_idx]
    ipa = ipa_converter.convertToPhonem(transcript[0])

    return {
        'real_transcript': transcript,
        'ipa_transcript': ipa,
        'transcript_translation': ""
    }

def get_random_sentence():
    sample_idx = random.randint(0, len(db_sentences) - 1)
    transcript = db_sentences[sample_idx]
    ipa = ipa_converter.convertToPhonem(transcript[0])

    return {
        'real_transcript': transcript,
        'ipa_transcript': ipa,
        'transcript_translation': ""
    }


def get_random_words(n: int):
    n = min(n, len(db_words))  
    sample_indices = random.sample(range(len(db_words)), n)

    results = []
    for idx in sample_indices:
        transcript = db_words[idx]
        ipa = ipa_converter.convertToPhonem(transcript[0])

        results.append({
            'real_transcript': transcript,
            'ipa_transcript': ipa,
            'transcript_translation': ""
        })

    return results


def get_random_sentences(n: int):
    n = min(n, len(db_sentences))
    sample_indices = random.sample(range(len(db_sentences)), n)

    results = []
    for idx in sample_indices:
        transcript = db_sentences[idx]
        ipa = ipa_converter.convertToPhonem(transcript[0])

        results.append({
            'real_transcript': transcript,
            'ipa_transcript': ipa,
            'transcript_translation': ""
        })

    return results