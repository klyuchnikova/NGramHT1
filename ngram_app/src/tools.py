from . import settings


def process_predicted_text(sentence: list[str]) -> str:
    sentence_string = ""
    for token in sentence:
        if token in settings.PUNCTUATION_SIGNS:
            sentence_string += token + " "
        elif token == settings.UNK_TOKEN:
            sentence_string += " ... "
        elif token == settings.BOS_TOKEN:
            pass
        elif token == settings.EOS_TOKEN:
            break
        else:
            sentence_string += " " + token + " "
    return sentence_string
