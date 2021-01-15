import numpy as np
import nltk
#nltk.download()
import pymorphy2
morph = pymorphy2.MorphAnalyzer()


def tokenizator(sentence):
    return nltk.word_tokenize(sentence, language="russian")


def normal_word(word):
    return morph.parse(word)[0].normal_form


def bag_of_words(tokenized_sentence, words):
    # приводим слово к начальной форме
    split_sentence = [normal_word(word) for word in tokenized_sentence]
    # создаем "сумку слов" из нулей
    bag = np.zeros(len(words), dtype=np.float32)
    for id, w in enumerate(words):
        if w in split_sentence:
            bag[id] = 1

    return bag

