import settings
from .models import NGramLanguageModel, WordCompletor, TextSuggestion
from .processing import preprocess_text, email_tokenizer
from .tools import process_predicted_texts


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
            word_completor=self.word_completor, n_gram_model=self.n_gram_model
        )

        self.tokenizer = lambda text: email_tokenizer(
            text,
            leave_punctuation=settings.LEAVE_PUNCTUATION,
            add_sentence_tokens=True,
            dictionary=self.word_completor.counts,
            preprocess_func=preprocess_text,  # so we don't attempt to cut prefix like in emails
        )

    def predict(self, text, alpha=0.5, n_words=3, n_texts=3, beam=3):
        text = [t for t in self.tokenizer(text)]
        preds, probs = self.suggestor.suggest_beam_text(
            text=text, beam_k=beam, n_words=n_words, n_texts=n_texts
        )
        return process_predicted_texts(preds), probs


def main():

    n_gram_checkpoint_path = f"{settings.PATH_TO_WORKDIR}/ngram_checkpoint_76000.pkl"
    word_completor_checkpoint_path = (
        "{settings.PATH_TO_WORKDIR}/word_completor_dictionary.pkl"
    )
    predictor = PredictorModel(
        n_gram_checkpoint_path=n_gram_checkpoint_path,
        word_completor_checkpoint_path=word_completor_checkpoint_path,
    )
