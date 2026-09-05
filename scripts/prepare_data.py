from datasets import load_dataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from weird_ai.config import RAW_DATA_DIR, SAMPLE_LYRICS_FILE, PROCESSED_DATA_DIR

"""
Read the downloaded dataset (parquet files) and create lyrics_sample.txt.
"""
def main():
    lyrics_column = "lyrics"
    limit = 5000
    selected_lyrics = []

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "parquet",
        data_files=str(RAW_DATA_DIR / "*.parquet"),
        split="train"
    )

    print(dataset)
    print(dataset.column_names)

 
    for row in dataset:
        lyrics = row.get(lyrics_column)

        if lyrics is None:
            continue

        lyrics = lyrics.strip()

        if len(lyrics) < 100:
            continue

        selected_lyrics.append(lyrics)

        if len(selected_lyrics) >= limit:
            break

    output_text = "\n\n<|song|>\n\n".join(selected_lyrics)

    SAMPLE_LYRICS_FILE.write_text(output_text, encoding="utf-8")

    print(f"Songs written: {len(selected_lyrics)}")
    print(f"Output file: {SAMPLE_LYRICS_FILE}")


if __name__ == "__main__":
    main()
