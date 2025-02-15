import numpy as np
import pandas as pd
from transformers import pipeline
import torch
import json

from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from deep_translator import MyMemoryTranslator
import ollama


class Characterizer:
    """
    Given a French Text, characterize its content into word lists of different types.
    clean_text: remove invalid words from the lost.
    extract_word_groups: extract lists of nouns, verbs and adjectives from the test.
    get_infinitive_pylefff: Convert a list of verbs from their conjugated form into infinitive form.
    translate_mymemory: Translate the list of words into English.
    add_gender: create a dictionary of {noun : gender} for the list of nouns with their 
    derived grammatical gender. Requires ollama and will be rather slow.
    """
    
    def __init__(self):
        self.device = torch.device("cpu")
        self.pos_tagger = pipeline(
            "token-classification", 
            model="waboucay/french-camembert-postag-model-finetuned-perceo", 
            device=self.device
        )
        self.lemmatizer = FrenchLefffLemmatizer()
        self.translator = MyMemoryTranslator(source="fr-FR", target="en-US")

    def clean_text(self, wordlist):
        """
        Cleans the text by removing tokens with a leading '▁' and short words.
        """
        return [n[1:] if n.startswith('▁') else n for n in wordlist if len(n) > 2]

    def extract_word_groups(self, text):
        """
        Given a text, find all nouns, verbs and adjectives.
        """
        # Get the POS tags
        pos_tags = self.pos_tagger(text)

        # Filter out nouns (you may need to adjust based on your specific model's tagset)
        nouns = self.clean_text(
            [word['word'] for word in pos_tags if word['entity'] == 'NOM']
        )
        verbs = self.clean_text(
            [word['word'] for word in pos_tags if word['entity'].startswith('VER')]
        )
        adjs = self.clean_text(
            [word['word'] for word in pos_tags if word['entity'].startswith('ADJ')]
        )

        return nouns, verbs, adjs

    def get_infinitive_pylefff(self, verbs):
        """
        Given a list of verbs, convert to their infinitives.
        """
        infinitives = [self.lemmatizer.lemmatize(v, 'v') for v in self.clean_text(verbs)]
        return infinitives

    def translate_mymemory(self, words):
        """
        Given a list of French words, translate them into English
        """
        translations = [self.translator.translate(w) for w in words]
        return translations

    def get_gender(self, text_input,  mod = "aya-expanse:32B", options={"temperature":0.8}):
        """
        A function to categorize the grammatical gender of French nouns.
        :param: text_input - a list of nouns to categorize
        """
    
        system_message = """In this task I want you to, given a list of French nouns, 
        determine the grammatical gender of each noun and return the results in another list. 
        The gender (masculine or feminine) will be abbreviated to 'm' for masculine 
        nouns and 'f' for feminine nouns. If the word does not appear to be a noun or it
        has no definite gender then please mark it as 'u' for unknown. Examples of words where
        this may apply are numbers, e.g. '2022', verbs like 'changer',adjectives like 'rouge'
        or prepostitions like 'sur'. Please assign a gender to each word of the input list.
        
        For example if passed the list ['chat', 'timbre', '20', 'robe'] please return the list
        ['m', 'm', 'u','f']. 
        
        Please only return the list of genders. Make sure
        Here is the list of nouns to determine their genders:
        """
        
        system_message = system_message + ' ' + str(text_input)
        
        
        self.device = torch.device("cpu")
    
        response = ollama.chat(model = mod, 
                               messages = [{"role":"user", "content" : system_message}],
                               options = options)

        return response['message']['content']
