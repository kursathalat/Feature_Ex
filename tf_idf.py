#This script extracts the tf-idf features from the text data and saves it in a csv file.
#The tf-idf features are extracted from the text data using the TfidfVectorizer from the scikit-learn library.
#You should have a 'corpus.json' file in the working directory which contains the text data.


from collections import defaultdict
import json
from sklearn.feature_extraction.text import TfidfVectorizer


with open('corpus.json', 'r') as json_file:
    corpus = json.load(json_file)["sentences"]

# words for which we want to extract tf-idf scores
words_to_tfidf = ["rock"]


def vectorize(corpus):
    """
    Vectorize the text data using the TfidfVectorizer
    """
    vectorizer = TfidfVectorizer()

    tfidf_vectors = vectorizer.fit_transform(corpus)

    terms = vectorizer.get_feature_names_out()

    return tfidf_vectors, terms

tf_idf_vectors, terms = vectorize(corpus)


def get_tfidf_scores(corpus, tf_idf_vectors, terms):
    """
    Extract the tf-idf scores for the words we are interested in
    """
    # save the tf-idf scores in a dictionary


    tf_idf_scores = defaultdict(list)

    for index, text in enumerate(corpus):

        for term_i in range(len(terms)):
            score = tf_idf_vectors[index, term_i]

            if terms[term_i] in words_to_tfidf:
                tf_idf_scores[text].append(score)

    return tf_idf_scores

def save_tfidf_scores(tf_idf_scores,words_to_tfidf):
    """
    Save the tf-idf scores in a csv file
    """
    with open('tf_idf_scores.csv', 'w') as file:
        file.write(f"text,tf_idf_score{words_to_tfidf}\n")
        for text, score in tf_idf_scores.items():
            print(text, score)
            file.write(f"{text},{score}\n")
        

tf_idf_scores = get_tfidf_scores(corpus, tf_idf_vectors, terms)

save_tfidf_scores(tf_idf_scores, words_to_tfidf)
