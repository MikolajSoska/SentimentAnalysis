from typing import List
from bs4 import BeautifulSoup
from unidecode import unidecode
from string import punctuation
from num2words import num2words
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


class Normalize:
    def __init__(self):
        self.tag_dict = {"J": wordnet.ADJ,
                         "N": wordnet.NOUN,
                         "V": wordnet.VERB,
                         "R": wordnet.ADV}

    def normalize(self, text: str) -> str:
        text = self.__remove_html_tags(text)
        text = self.__to_lower_case(text)
        text = self.__remove_non_ascii(text)
        text = self.__remove_punctuation(text)

        words: List[str] = self.__tokenize_words(text)
        words = self.__numbers_to_words(words)
        words = self.__remove_stop_words(words)
        words = self.__lemmatize_words(words)

        return " ".join(words)

    def __remove_html_tags(self, text: str) -> str:
        html = BeautifulSoup(text, "html.parser")
        return html.get_text()

    def __to_lower_case(self, text: str) -> str:
        return text.lower()

    def __remove_non_ascii(self, text: str) -> str:
        return unidecode(text)

    def __remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans(punctuation, ' ' * len(punctuation)))

    def __tokenize_words(self, text: str) -> List[str]:
        return word_tokenize(text)

    def __numbers_to_words(self, words: List[str]) -> List[str]:
        new_words = list()

        for word in words:
            if word.isdigit():
                new_words.append(num2words(word))
            else:
                new_words.append(word)

        return new_words

    def __remove_stop_words(self, words: List[str]) -> List[str]:
        stop_words = stopwords.words('english')
        new_words = list()

        for word in words:
            if word not in stop_words:
                new_words.append(word)

        return new_words

    def __lemmatize_words(self, words: List[str]) -> List[str]:
        lemmatizer = WordNetLemmatizer()
        new_words = list()
        tagged_words = pos_tag(words)

        for word, tag in tagged_words:
            new_words.append(lemmatizer.lemmatize(word, pos=self.__get_wordnet_tag(tag)))

        return new_words

    def __get_wordnet_tag(self, tag: str) -> str:
        letter = tag[0].upper()

        return self.tag_dict.get(letter, wordnet.NOUN)
