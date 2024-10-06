import re
from . import settings
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize


def remove_html(data):
    beauti = BeautifulSoup(data, "html.parser")
    return beauti.get_text()


def cut_prefix_metadata(email):
    # cuts off both metadata and forwards or re/cites/anything coming before 'main body'
    latest_start = -1
    for prefex_template in ["X-FileName:[^\n]*\n", "Subject:[^\n]*\n"]:
        for m in re.finditer(prefex_template, email):
            latest_start = max(latest_start, m.end(0))
    return email[latest_start:]


def preprocess_text(
    text, replace_with_unknown=True, leave_punctuation=settings.LEAVE_PUNCTUATION
):
    string_to_replace_with = settings.UNK_TOKEN + " " if replace_with_unknown else " "
    text = (
        text.lower()
    )  # I decided to ignore all names and stuff that should be capitalized
    text = re.sub(r"https?:\/\/.*[\r\n]*", string_to_replace_with, text)  # remove links
    text = re.sub(r"\S*@\S*\s", string_to_replace_with, text)  # remove mail adresses
    text = re.sub(r"\S*\.\S*\s", string_to_replace_with, text)  # remove file notions
    text = remove_html(text)  # remove <a>, <bs> and such
    # text = re.sub('\(.*?\)','',text) # remove round brackets
    text = re.sub(
        "\S\S*\d+\S*\s", string_to_replace_with, text
    )  # remove everything with numbers
    non_english_re = (
        f"[^a-zA-Z{settings.PUNCTUATION_SIGNS}]+" if leave_punctuation else "[^a-zA-Z]+"
    )
    text = re.sub(
        non_english_re, " ", text
    )  # remove literally everything except numbers, letters, basic punctuation
    text = " ".join(text.split())  # remove unnecessary \n and whitespaces
    return text


def preprocess_email(
    email,
    replace_with_unknown=True,
    leave_punctuation=settings.LEAVE_PUNCTUATION,
):
    email = cut_prefix_metadata(email)
    return preprocess_text(
        email,
        replace_with_unknown=replace_with_unknown,
        leave_punctuation=leave_punctuation,
    )


def email_tokenizer(
    email,
    leave_punctuation=settings.LEAVE_PUNCTUATION,
    add_sentence_tokens=True,
    dictionary=None,
    preprocess_func=preprocess_email,
):  # THIS IS A GENERATOR!
    # we will also add <EOS> and <BOS> around sentences.
    # N-grams are generally not for multiple logically connected sentences generation so it's fine
    tokens = word_tokenize(
        preprocess_func(
            email,
            leave_punctuation=leave_punctuation,
        )
    )
    is_start_of_sentence = True
    previous_token_punkt = False  # we don't want multiple punkts in a row
    previous_token_unk = False  # same here, there are way too many unknown tokens
    for token in tokens:
        if add_sentence_tokens and is_start_of_sentence:
            # We don't want sentences to start with punctual signs
            if token in settings.PUNCTUATION_SIGNS:
                continue
            yield settings.BOS_TOKEN
            is_start_of_sentence = False

        # checking if punkt
        if token in settings.PUNCTUATION_SIGNS:
            if previous_token_punkt or not leave_punctuation:
                continue
            previous_token_punkt = True
        else:
            previous_token_punkt = False

        if dictionary is not None and token not in dictionary:
            token = settings.UNK_TOKEN

        # checking if unk
        if token == settings.UNK_TOKEN:
            if previous_token_unk:
                continue
            previous_token_unk = True
        else:
            previous_token_unk = False

        yield token
        if add_sentence_tokens and token in ".?!":
            is_start_of_sentence = True
            yield settings.EOS_TOKEN
    if add_sentence_tokens and not is_start_of_sentence:
        yield settings.EOS_TOKEN
