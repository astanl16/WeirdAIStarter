import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from weird_ai.config import SAMPLE_LYRICS_FILE, PROCESSED_DATA_DIR, TOKENS_FILE
from weird_ai.tokenizer import SimpleCharacterTokenizer

"""
Convert text into tokens
"""
def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    text = SAMPLE_LYRICS_FILE.read_text(encoding="utf-8")

    tokenizer = SimpleCharacterTokenizer(text)
    tokens = tokenizer.encode(text)

    TOKENS_FILE.write_text(
        " ".join(str(token) for token in tokens),
        encoding="utf-8"
    )

    print(f"Characters in corpus: {len(text)}")
    print(f"Vocabulary size: {len(tokenizer.chars)}")
    print(f"Total tokens: {len(tokens)}")
    print(f"Tokens written to: {TOKENS_FILE}")

    decoded_preview = tokenizer.decode(tokens[:500])

    print("\nDecoded preview:")
    print(decoded_preview)


if __name__ == "__main__":
    main()