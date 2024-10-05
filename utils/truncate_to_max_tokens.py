import nltk
from nltk.data import find
from nltk.data import load


def ensure_punkt_downloaded():
    try:
        # Try to find the punkt tokenizer in NLTK's data
        find('tokenizers/punkt')
        print("Punkt tokenizer is already downloaded.")
    except LookupError:
        # If not found, download the punkt tokenizer
        print("Punkt tokenizer not found. Downloading...")
        nltk.download('punkt')
        print("Punkt tokenizer downloaded.")


def truncate_to_max_tokens(text, max_tokens=512):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Truncate the tokens to the maximum allowed
    truncated_tokens = tokens[:max_tokens]
    # Join the tokens back into a string
    truncated_text = ' '.join(truncated_tokens)
    return truncated_text


if __name__ == "__main__":
    # Call the function to ensure punkt is downloaded
    ensure_punkt_downloaded()