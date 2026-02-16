# Language Detection Script

This script detects languages in a dataset using the `langdetect` library.

## Requirements

- Python 3.x
- Install dependencies: `pip install -r requirements.txt`

## Usage

1. Prepare your dataset as a CSV file (update the filename in the script, e.g., `df = pd.read_csv('your_dataset.csv')`).
2. Ensure the CSV has a column named `text` containing the text data.
3. Run the script: `python language_detection.py`
4. The script will show progress during language detection.
5. Processed data will be saved to `output.csv` with an added `language` column.
6. A bar chart of the language distribution will be saved as `language_distribution.png`.
7. The second most frequent language (excluding English) will be printed.

## Features

- Handles exceptions for short texts or non-text content by marking as 'unknown'.
- Categorizes languages and visualizes the distribution (language codes on chart; full names in console output).
- Shows progress during processing for large datasets.
- Prints the second most frequent language with full name (excluding English).
- Translates up to 3 sample texts from the second most language to English.
- Saves the processed data to a new file.

## Notes for Large Datasets

- For datasets with 100,000+ rows, processing may take 10-30 minutes depending on your hardware.
- The script includes a progress bar using `tqdm`.
- If needed, uncomment the subset line in the script to test on fewer rows first.