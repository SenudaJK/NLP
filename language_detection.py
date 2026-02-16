import pandas as pd
from langdetect import detect, LangDetectException
import matplotlib.pyplot as plt
from tqdm import tqdm
import pycountry
from deep_translator import GoogleTranslator

# Load the dataset (assuming CSV with a 'text' column)
df = pd.read_csv('multilingual_dataset_train_final.csv')

# Optional: Process only a subset for testing (e.g., first 1000 rows)
# Uncomment the next line to limit processing for large datasets
df = df.head(1000)
# df = df.head(100)  # Limit to the first 100 entries for testing purposes

# List to store detected languages
languages = []

# Detect language for each text entry
for text in tqdm(df['text'], desc="Detecting languages"):
    try:
        # Attempt to detect the language
        lang = detect(text)
        languages.append(lang)
    except LangDetectException:
        # Handle exceptions: if detection fails (e.g., text too short, only numbers, etc.)
        languages.append('unknown')

# Add the language column to the dataframe
df['language'] = languages

# Categorize and count the languages
lang_counts = df['language'].value_counts()

# Find the second most frequent language excluding English
lang_counts_no_en = lang_counts.drop('en', errors='ignore')
if len(lang_counts_no_en) >= 2:
    second_most_lang = lang_counts_no_en.index[1]
    second_most_count = lang_counts_no_en.iloc[1]
    # Get full language name
    try:
        lang_obj = pycountry.languages.get(alpha_2=second_most_lang)
        lang_name = lang_obj.name if lang_obj else second_most_lang
    except:
        lang_name = second_most_lang
    print(f"Second most frequent language (excluding English): {lang_name} ({second_most_lang}) with {second_most_count} occurrences.")
    
    # Translate a sample text from the second most language to English
    sample_texts = df[df['language'] == second_most_lang]['text'].head(3)  # Get up to 3 samples
    if not sample_texts.empty:
        print(f"\nTranslating sample texts from {lang_name} to English:")
        translator = GoogleTranslator(source=second_most_lang, target='en')
        for i, text in enumerate(sample_texts, 1):
            try:
                translated = translator.translate(text)
                print(f"Sample {i}: {text[:50]}... -> {translated}")
            except Exception as e:
                print(f"Sample {i}: Translation failed - {str(e)}")
    else:
        print(f"No sample texts found for {lang_name}.")
elif len(lang_counts_no_en) == 1:
    only_lang = lang_counts_no_en.index[0]
    try:
        lang_obj = pycountry.languages.get(alpha_2=only_lang)
        lang_name = lang_obj.name if lang_obj else only_lang
    except:
        lang_name = only_lang
    print(f"Only one language other than English: {lang_name} ({only_lang})")
else:
    print("No languages other than English detected.")

# Visualize the distribution using a bar chart
plt.figure(figsize=(10, 6))
plt.bar(lang_counts.index, lang_counts.values, color='skyblue')
plt.xlabel('Language')
plt.ylabel('Count')
plt.title('Language Distribution in Dataset')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot to a file
plt.savefig('language_distribution.png')

# Save the processed data to a new CSV file
df.to_csv('output.csv', index=False)

print("Language detection completed. Processed data saved to 'output.csv' and plot saved to 'language_distribution.png'.")