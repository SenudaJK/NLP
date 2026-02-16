import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
import spacy
import re
import argparse
import sys
from collections import Counter
from tqdm import tqdm

# --- CONFIGURATION ---
# Try to load Spacy model, handle error if not installed
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Spacy model not found. Please run: python -m spacy download en_core_web_sm")
    sys.exit(1)

def load_data(file_path, limit=None):
    """Loads CSV and optionally limits rows for testing."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} rows from {file_path}")
        if limit:
            print(f"Limiting processing to first {limit} rows.")
            return df.head(limit).copy()
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)

def infer_country_efficient(texts):
    """Uses Spacy's nlp.pipe for faster batch processing to find GPEs."""
    countries = []
    # nlp.pipe is much faster than calling nlp() in a loop
    for doc in tqdm(nlp.pipe(texts, disable=["parser", "tagger"], batch_size=50), total=len(texts), desc="Inferring countries"):
        # Find all GPEs (Geopolitical Entities) in the text
        gpes = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
        # If GPEs found, pick the most common one in that specific text, else 'Unknown'
        if gpes:
            countries.append(Counter(gpes).most_common(1)[0][0])
        else:
            countries.append('Unknown')
    return countries

def infer_date(text):
    """Extracts dates using regex."""
    if not isinstance(text, str): return 'Unknown'
    # Pattern for DD/MM/YYYY or YYYY-MM-DD
    date_pattern = r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'
    match = re.search(date_pattern, text)
    return match.group(0) if match else 'Unknown'

def infer_topic(text):
    """
    Classifies topic based on weighted keyword matching.
    Returns the category with the highest keyword overlap.
    """
    if not isinstance(text, str): return 'General'
    text_lower = text.lower()
    
    # Enhanced Dictionary with broader and more specific terms
    topic_keywords = {
        'Politics': [
            'election', 'government', 'president', 'senate', 'parliament', 'congress', 
            'minister', 'diplomat', 'policy', 'campaign', 'voter', 'ballot', 
            'democrat', 'republican', 'legislation', 'treaty', 'sanctions', 'sovereignty',
            'prime minister', 'cabinet', 'referendum', 'white house', 'corrupt', 'scandal'
        ],
        'Sports': [
            'game', 'match', 'tournament', 'league', 'championship', 'olympic', 
            'score', 'goal', 'touchdown', 'athlete', 'coach', 'stadium', 'roster',
            'medal', 'world cup', 'nba', 'fifa', 'nfl', 'cricket', 'rugby', 'tennis',
            'referee', 'squad', 'final', 'semi-final', 'trophy'
        ],
        'Technology': [
            'software', 'hardware', 'app', 'ai', 'artificial intelligence', 'algorithm', 
            'cyber', 'internet', 'device', 'silicon', 'startup', 'innovation', 
            'data', 'cloud', 'server', 'blockchain', 'robot', 'automation', 
            'feature', 'update', 'browser', 'google', 'apple', 'microsoft', 'coding'
        ],
        'Health': [
            'virus', 'pandemic', 'epidemic', 'vaccine', 'hospital', 'doctor', 
            'nurse', 'patient', 'treatment', 'surgery', 'clinical', 'symptom', 
            'infection', 'cancer', 'mental health', 'drug', 'medicine', 'fda', 
            'who', 'disease', 'wellness', 'diet', 'obesity', 'research study'
        ],
        'Finance': [
            'stock', 'market', 'economy', 'inflation', 'recession', 'currency', 
            'bank', 'crypto', 'bitcoin', 'investment', 'revenue', 'profit', 
            'quarterly', 'shares', 'trade', 'wall street', 'tax', 'budget', 
            'fiscal', 'interest rate', 'federal reserve', 'dividend', 'shareholder'
        ],
        'Entertainment': [
            'movie', 'film', 'cinema', 'actor', 'actress', 'director', 'hollywood', 
            'music', 'album', 'song', 'concert', 'grammy', 'oscar', 'celebrity', 
            'star', 'series', 'episode', 'netflix', 'streaming', 'box office', 'drama'
        ],
        'Environment': [
            'climate', 'carbon', 'emission', 'global warming', 'energy', 'solar', 
            'oil', 'gas', 'renewable', 'wildlife', 'conservation', 'pollution', 
            'weather', 'storm', 'earthquake', 'flood', 'temperature', 'sustainability'
        ]
    }
    
    # Scoring system: Count occurrences of keywords for each topic
    scores = {topic: 0 for topic in topic_keywords}
    
    for topic, keywords in topic_keywords.items():
        for kw in keywords:
            # Add 1 point for every occurrence of the keyword
            # Using specific boundary checking helps avoid partial matches (e.g., "star" in "start")
            if f" {kw} " in f" {text_lower} ": 
                scores[topic] += 1
                
    # Find the topic with the maximum score
    best_topic = max(scores, key=scores.get)
    
    # If no keywords matched (score is 0), return General
    if scores[best_topic] == 0:
        return 'General'
        
    return best_topic

def safe_detect_language(text):
    """Detects language with error handling and text truncation for speed."""
    if not isinstance(text, str) or len(text.strip()) < 3:
        return 'unknown'
    try:
        # Truncating to first 500 chars speeds up detection significantly without losing much accuracy
        return detect(text[:500])
    except LangDetectException:
        return 'unknown'

def translate_samples(df, target_lang='en'):
    """Finds the second most common language and translates samples."""
    lang_counts = df['language'].value_counts()
    
    # Exclude English and Unknown
    filtered_counts = lang_counts.drop(['en', 'unknown'], errors='ignore')
    
    if filtered_counts.empty:
        print("\nNo foreign languages found to translate.")
        return

    top_foreign_lang = filtered_counts.index[0]
    count = filtered_counts.iloc[0]
    
    print(f"\n--- Translation Task ---")
    print(f"Top foreign language: '{top_foreign_lang}' with {count} entries.")
    
    samples = df[df['language'] == top_foreign_lang]['text'].head(3).tolist()
    translator = GoogleTranslator(source=top_foreign_lang, target=target_lang)
    
    print(f"Translating 3 samples to English:")
    for i, text in enumerate(samples, 1):
        try:
            # Translate first 200 chars to avoid API timeouts
            snippet = text[:200]
            translated = translator.translate(snippet)
            print(f"\n[Sample {i}] Original: {snippet}...")
            print(f"[Sample {i}] Translated: {translated}...")
        except Exception as e:
            print(f"[Sample {i}] Translation failed: {e}")

def visualize_results(df):
    """Creates a cleaner visualization of language distribution."""
    plt.figure(figsize=(12, 6))
    
    # Get top 10 languages, group others as 'Other'
    lang_counts = df['language'].value_counts()
    if len(lang_counts) > 10:
        top_10 = lang_counts[:10]
        other_count = lang_counts[10:].sum()
        top_10['Other'] = other_count
        plot_data = top_10
    else:
        plot_data = lang_counts

    sns.barplot(x=plot_data.index, y=plot_data.values, hue=plot_data.index, palette='viridis', legend=False)
    plt.title('Top Languages Distribution')
    plt.xlabel('Language Code')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('language_distribution_enhanced.png')
    print("\nPlot saved as 'language_distribution_enhanced.png'")

# --- MAIN EXECUTION ---
def main(input_file, output_file, row_limit):
    df = load_data(input_file, row_limit)
    
    # 1. Infer Categories (if missing)
    if 'text' not in df.columns:
        print("Error: Dataset must have a 'text' column.")
        return

    print("\n--- Processing Metadata ---")
    # Only run inference if columns don't exist
    if 'country' not in df.columns:
        print("Inferring countries (using Spacy batching)...")
        df['country'] = infer_country_efficient(df['text'].fillna('').tolist())
        
    if 'date' not in df.columns:
        print("Inferring dates...")
        df['date'] = list(tqdm(df['text'].apply(infer_date), total=len(df), desc="Inferring dates"))
        
    if 'type' not in df.columns:
        print("Inferring topics...")
        df['type'] = list(tqdm(df['text'].apply(infer_topic), total=len(df), desc="Inferring topics"))

    # 2. Detect Languages
    print("\n--- Detecting Languages ---")
    # Using apply is cleaner than a manual for loop
    df['language'] = list(tqdm(df['text'].apply(safe_detect_language), total=len(df), desc="Detecting languages"))

    # 3. Analyze and Translate
    translate_samples(df)

    # 4. Save and Visualize
    df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to '{output_file}'")
    
    visualize_results(df)

if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Detect and categorize languages in a dataset.")
    parser.add_argument("--input", type=str, default="multilingual_dataset_test_final.csv", help="Path to input CSV")
    parser.add_argument("--output", type=str, default="enhanced_output.csv", help="Path to output CSV")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for testing (e.g., 100)")
    
    args = parser.parse_args()
    
    main(args.input, args.output, args.limit)