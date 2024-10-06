from typing import List, Union, Tuple
from collections import defaultdict
from .processing import preprocess_text, email_tokenizer
from .tools import process_predicted_text

import pickle
import math
import os

from . import settings


class PrefixTreeNode:
    def __init__(self):
        # словарь с буквами, которые могут идти после данной вершины
        self.children: dict[str, PrefixTreeNode] = {}
        self.is_end_of_word = False

    def print(self, level=0):
        for letter, child in self.children.items():
            print(
                "\t" * level + letter + ":" + ("final" if child.is_end_of_word else "")
            )
            child.print(level + 1)


class PrefixTree:
    def __init__(self, vocabulary: List[str]):
        """
        vocabulary: список всех уникальных токенов в корпусе
        """
        self.root = PrefixTreeNode()

        for word in vocabulary:
            current_node = self.root
            for i in range(len(word)):
                if word[i] not in current_node.children:
                    current_node.children[word[i]] = PrefixTreeNode()
                current_node = current_node.children[word[i]]
            current_node.is_end_of_word = True

    def search_prefix(self, prefix) -> List[str]:
        """
        Возвращает все слова, начинающиеся на prefix
        prefix: str – префикс слова
        """

        current_node = self.root
        for letter in prefix:
            current_node = current_node.children.get(letter)
            if current_node is None:
                return []

        found_words = []

        def recursive_walk(node, word=prefix):
            for letter, child in node.children.items():
                recursive_walk(child, word + letter)
            if node.is_end_of_word:
                found_words.append(word)

        recursive_walk(current_node)

        return found_words

    def print(self):
        self.root.print()


class WordCompletor:
    def __init__(self, corpus, minimal_occurance=0, max_vocab_size=50000):
        """
        corpus: list – корпус текстов
        """
        self.counts = defaultdict(int)
        for text in corpus:
            for word in text:
                self.counts[word] += 1
        if self.counts:
            min_threshold = max(
                minimal_occurance,
                sorted(self.counts.values())[
                    min(
                        max(0, len(self.counts) - max_vocab_size - 1),
                        len(self.counts) - 1,
                    )
                ],
            )
            self.counts = {
                word: count
                for word, count in self.counts.items()
                if count >= min_threshold
            }
        self.number_words_total = sum(self.counts.values())
        self.prefix_tree = PrefixTree(self.counts.keys())

    def most_often(self, k):
        min_threshold = sorted(self.counts.values())[
            min(len(self.counts) - k - 1, len(self.counts) - 1)
        ]
        return {
            word: count for word, count in self.counts.items() if count >= min_threshold
        }

    def most_rare(self, k):
        max_threshold = sorted(self.counts.values())[min(k - 1, len(self.counts) - 1)]
        return {
            word: count for word, count in self.counts.items() if count <= max_threshold
        }

    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.counts, f)

    def load(self, file_path):
        with open(file_path, "rb") as f:
            self.counts = pickle.load(f)
        self.number_words_total = sum(self.counts.values())
        self.prefix_tree = PrefixTree(self.counts.keys())

    def get_words_and_probs(self, prefix: str) -> (List[str], List[float]):
        """
        Возвращает список слов, начинающихся на prefix,
        с их вероятностями (нормировать ничего не нужно)
        """
        words = self.prefix_tree.search_prefix(prefix)
        probs = [self.counts.get(word, 0) for word in words]
        return words, [prob / self.number_words_total for prob in probs]

    def get_top_k_words_and_probs(
        self, prefix: str, k: int
    ) -> (List[str], List[float]):
        """
        Возвращает список слов, начинающихся на prefix,
        с их вероятностями (нормировать ничего не нужно)
        """
        words = self.prefix_tree.search_prefix(prefix)
        probs = [self.counts.get(word, 0) for word in words]

        sorded_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        words = [words[i] for i in sorded_indices[:k]]
        probs = [
            math.log2(probs[i] / self.number_words_total) for i in sorded_indices[:k]
        ]
        return words, probs


class NGramLanguageModel:
    SEPARATOR = "$"

    def __init__(self, corpus, n, checkpoint_every=None, load_checkpoint_id=None):
        self.n = n
        self.n_grams = {}
        if load_checkpoint_id:
            self.load(
                f"{settings.PATH_TO_WORKDIR}/ngram_checkpoint_{load_checkpoint_id}.pkl"
            )
        for text_id, text in enumerate(
            corpus,
            start=load_checkpoint_id or 0,
        ):
            if checkpoint_every and (text_id + 1) % checkpoint_every == 0:
                print(f"Current iteration: {text_id}")
                self.checkpoint(
                    f"{settings.PATH_TO_WORKDIR}/ngram_checkpoint_{text_id+1}.pkl"
                )
                if (
                    text_id > 3
                    and f"ngram_checkpoint_{text_id+1-3*checkpoint_every}.pkl"
                    in os.listdir(settings.PATH_TO_WORKDIR)
                ):
                    os.remove(
                        f"{settings.PATH_TO_WORKDIR}/ngram_checkpoint_{text_id + 1 - 3 * checkpoint_every}.pkl"
                    )
            for i in range(len(text)):
                ngram_before_last = self.SEPARATOR.join(text[max(0, i - n) : i])

                next_word = text[i]
                if ngram_before_last not in self.n_grams:
                    self.n_grams[ngram_before_last] = (
                        {}
                    )  # Initialize inner dictionary, defaultdict said fuck you when pickling
                # we'll also store number of n-grams like this
                self.n_grams[ngram_before_last][self.SEPARATOR] = (
                    self.n_grams[ngram_before_last].get("$", 0) + 1
                )
                if next_word != settings.UNK_TOKEN:
                    self.n_grams[ngram_before_last][next_word] = (
                        self.n_grams[ngram_before_last].get(next_word, 0) + 1
                    )

    def get_next_words_and_probs(self, prefix: list) -> (List[str], List[float]):
        """
        Возвращает список слов, которые могут идти после prefix,
        а так же список вероятностей этих слов
        """

        next_words, probs = [], []
        ngram_prefix = self.SEPARATOR.join(prefix[-self.n :])
        next_words = list(self.n_grams[ngram_prefix].keys())
        found_ngrams = sum(self.n_grams[ngram_prefix].values())
        probs = [v / found_ngrams for v in self.n_grams[ngram_prefix].values()]
        return next_words, probs

    def checkpoint(self, file_path):
        print(f"---MAKING A CHECKPOINT AT: {file_path}---")
        self.save(file_path)

    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump({"n": self.n, "n_grams": self.n_grams}, f)

    def load(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            self.n_grams = data["n_grams"]
            self.n = data["n"]

    def get_top_k_next_words_and_probs(
        self, prefix: list, k: int, alpha: float = 0, vocab_size: int = 0
    ) -> (List[str], List[float]):
        """
        Возвращает список слов, которые могут идти после prefix,
        а так же список вероятностей этих слов (top k)
        считаются они с Laplace smoothing и отлогарифмированные
        """

        next_words, probs = [], []
        # for maximizing the probaility of finding some known prefix:
        for left_i in range(-max(len(prefix), -self.n), 0):
            ngram_prefix = self.SEPARATOR.join(prefix[left_i:])
            if ngram_prefix in self.n_grams:
                next_words = list(self.n_grams[ngram_prefix].keys())
                break

        if not next_words:
            print(
                f"No prefix found out of: {[self.SEPARATOR.join(prefix[left_i:]) for left_i in range(-max(len(prefix), -self.n), 0)]}"
            )
            return next_words, probs

        try:
            # we'll also predict seperator which is obviously unnecessary
            next_words.remove(self.SEPARATOR)
        except:
            pass

        try:
            next_words.remove(settings.UNK_TOKEN)
        except:
            pass

        sorded_indices = sorted(
            range(len(next_words)),
            key=lambda i: self.n_grams[ngram_prefix][next_words[i]],
            reverse=True,
        )[:k]
        next_words = [next_words[i] for i in sorded_indices]

        prev_number_ngrams = self.n_grams[ngram_prefix][self.SEPARATOR]
        probs = [
            math.log2(
                (self.n_grams[ngram_prefix][next_word] + alpha) / prev_number_ngrams
                + alpha * vocab_size
            )
            for next_word in next_words
        ]
        return next_words, probs


class TextSuggestion:
    def __init__(self, word_completor, n_gram_model, dictionary=None) -> None:
        self.word_completor = word_completor
        self.n_gram_model = n_gram_model
        self.dictionary = dictionary or self.word_completor.counts

    def suggest_beam_text(
        self, text: Union[str, list], beam_k=3, n_words=3, n_texts=5, alpha=0.5
    ) -> List[Tuple[List[str], float]]:
        """
        Возвращает возможные варианты продолжения текста (по умолчанию только один)

        text: строка или список слов – написанный пользователем текст
        n_words: число слов, которые дописывает n-граммная модель
        n_texts: число возвращаемых продолжений (пока что только одно)

        return: list[tuple[list[srt], float]] – список из n_texts списков слов, по 1 + n_words слов в каждом + логарифм вероятности (типа)
        Первое слово – это то, которое WordCompletor дополнил до целого.
        """

        # I suppose we should at least try to prepare enough suggestions via beam if there're too many desired texts
        if int(math.log(n_texts, n_words) + 1) > beam_k:
            print(
                f"Warning! Suggestions are run with beam parameters: {beam_k} not enough to cover desired number texts: {n_texts}"
            )

        suggestions = []

        if isinstance(text, str):
            text = text.strip().split()
        else:
            text = text[:-1]

        # first make word continuation suggestions
        last_word = ""
        if len(text) > 1:
            last_word = text[-1]
            text = text[:-1]
        text = [settings.BOS_TOKEN] + [
            token if token in self.dictionary else settings.UNK_TOKEN
            for token in text[1:]
        ]

        pred_words, pred_probs = self.word_completor.get_top_k_words_and_probs(
            last_word, k=beam_k
        )

        if not pred_words:
            print(f"Couldn't complete word: {last_word}")
            return suggestions  # we can't even understand the word so predicting future's impossible

        suggestions.extend([([p_w], p_p) for p_w, p_p in zip(pred_words, pred_probs)])
        final_suggestions = []

        i = 0
        # now let's make n-gram word suggestions
        while i < len(suggestions):
            old_words, old_prob = suggestions[i]
            if len(old_words) >= n_words:
                break

            if len(old_words) >= self.n_gram_model.n:
                n_gram_prefix = old_words[min(-1, -self.n_gram_model.n) :]
            else:
                n_gram_prefix = (
                    text[min(-1, -self.n_gram_model.n + len(old_words)) :]
                    + old_words[min(-1, -self.n_gram_model.n) :]
                )
            new_words, new_probs = self.n_gram_model.get_top_k_next_words_and_probs(
                n_gram_prefix, k=beam_k, alpha=alpha, vocab_size=settings.MAX_VOCAB_SIZE
            )
            for new_word, new_prob in zip(new_words, new_probs):
                new_pair = (old_words.copy() + [new_word], old_prob + new_prob)
                if new_word == settings.EOS_TOKEN or len(new_pair[0]) == n_words:
                    final_suggestions.append(new_pair)
                    print(f"New pair added to final suggestions: {new_pair}")
                elif new_word != settings.BOS_TOKEN:
                    suggestions.append(new_pair)
                    print(f"New pair added to suggestions: {new_pair}")
            i += 1

        # sort long enough suggestions by probability
        final_suggestions.extend(
            suggestions[
                min(max(len(suggestions) + len(final_suggestions) - n_texts, 0), i) :
            ]
        )
        final_suggestions = sorted(final_suggestions, key=lambda pair: pair[1])
        return final_suggestions[-n_texts:]


class PredictorModel:
    def __init__(self, word_completor_checkpoint_path, n_gram_checkpoint_path, **args):
        self.word_completor = WordCompletor(
            corpus=[],
            minimal_occurance=settings.MINIMAL_OCCURANCE,
            max_vocab_size=settings.MAX_VOCAB_SIZE,
        )
        self.word_completor.load(word_completor_checkpoint_path)

        self.n_gram_model = NGramLanguageModel(corpus=[], n=settings.N_GRAMS)
        self.n_gram_model.load(n_gram_checkpoint_path)

        self.suggestor = TextSuggestion(
            word_completor=self.word_completor,
            n_gram_model=self.n_gram_model,
        )

        self.tokenizer = lambda text: email_tokenizer(
            text,
            leave_punctuation=settings.LEAVE_PUNCTUATION,
            add_sentence_tokens=True,
            # dictionary=self.word_completor.counts, # unfortunately when we use both n_gram and word completer we can't filter straight away
            preprocess_func=preprocess_text,  # so we don't attempt to cut prefix like in emails
        )

    def predict(self, text, alpha=0.5, n_words=3, n_texts=3, beam=3):
        text = [t for t in self.tokenizer(text)]
        preds_probs = self.suggestor.suggest_beam_text(
            text=text, beam_k=beam, n_words=n_words, n_texts=n_texts
        )
        return [(process_predicted_text(pred), prob) for (pred, prob) in preds_probs]
