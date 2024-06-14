# This task calculates the average similarity within a sentence.
# To do this, it first calculates the similarity between the two words inside a sentence.
# For Turkish, it uses tr_core_web_lg from https://github.com/turkish-nlp-suite/turkish-spacy-models

#@inproceedings{altinok-2023-diverse,
#    title = "A Diverse Set of Freely Available Linguistic Resources for {T}urkish",
#    author = "Altinok, Duygu",
#    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
#    month = jul,
#    year = "2023",
#    address = "Toronto, Canada",
#    publisher = "Association for Computational Linguistics",
#    url = "https://aclanthology.org/2023.acl-long.768",
#    pages = "13739--13750",
#    abstract = "This study presents a diverse set of freely available linguistic resources for Turkish natural language processing, including corpora, pretrained models and education material. Although Turkish is spoken by a sizeable population of over 80 million people, Turkish linguistic resources for natural language processing remain scarce. In this study, we provide corpora to allow practitioners to build their own applications and pretrained models that would assist industry researchers in creating quick prototypes. The provided corpora include named entity recognition datasets of diverse genres, including Wikipedia articles and supplement products customer reviews. In addition, crawling e-commerce and movie reviews websites, we compiled several sentiment analysis datasets of different genres. Our linguistic resources for Turkish also include pretrained spaCy language models. To the best of our knowledge, our models are the first spaCy models trained for the Turkish language. Finally, we provide various types of education material, such as video tutorials and code examples, that can support the interested audience on practicing Turkish NLP. The advantages of our linguistic resources are three-fold: they are freely available, they are first of their kind, and they are easy to use in a broad range of implementations. Along with a thorough description of the resource creation process, we also explain the position of our resources in the Turkish NLP world.",
#}

# Also, it uses cc.tr.300.bin from fasttext

import spacy
import fasttext
import json
from nltk.util import ngrams
import numpy as np
import pandas as pd

nlp = spacy.load("tr_core_news_lg")
ft = fasttext.load_model('cc.tr.300.bin')

with open('corpus.json', 'r') as json_file:
    corpus = json.load(json_file)["sentences"]

def generate_pos_ngrams(pos_tagged_sentence, n) -> list:
    """
    Creates the n-grams from the POS-tagged sentence
    """
    return list(ngrams(pos_tagged_sentence, n))

def word_similarity(word1, word2) -> float:
    """
    Calculates the similarity between two words
    """
    vec1 = ft.get_word_vector(word1)
    vec2 = ft.get_word_vector(word2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def average_similarity(sentence) -> float:
    """
    Calculates the average similarity within a sentence
    """
    words = sentence.split()
    similarity_scores = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            similarity_scores.append(word_similarity(words[i], words[j]))
    if similarity_scores:
        return sum(similarity_scores) / len(similarity_scores)
    else:
        return 0.0  # if no word pairs are found
    

if __name__ == "__main__":

    sentence_similarity = []

    for sentence in corpus:
        print(f"{sentence}: {average_similarity(sentence)}")
        sentence_similarity.append(average_similarity(sentence))

    similarities = pd.DataFrame({"sentence_similarity":sentence_similarity})
    
    #similarities.to_excel("word_similarity.xlsx") # Save the results to an Excel file
