# yeah, I know this is better realised via config file but I don't want to fully rewrite my goddamn code from collab
global PATH_TO_WORKDIR
global PUNCTUATION_SIGNS
global LEAVE_PUNCTUATION
global UNK_TOKEN
global BOS_TOKEN
global EOS_TOKEN
global N_GRAMS
global MINIMAL_OCCURANCE
global MAX_VOCAB_SIZE

PATH_TO_WORKDIR = "n_gram_keyboard\\data"
PUNCTUATION_SIGNS = ",.?!:;'"
LEAVE_PUNCTUATION = True
UNK_TOKEN = "UNK"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
N_GRAMS = 3
MINIMAL_OCCURANCE = 20
MAX_VOCAB_SIZE = 10000
