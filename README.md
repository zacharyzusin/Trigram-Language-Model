# Trigram Language Model for Essay Classification

This project implements a trigram language model in Python for text analysis and essay classification. The model is specifically designed to analyze and classify TOEFL essays based on writing proficiency levels (high/low).

## Features

- Trigram language model implementation with:
  - N-gram extraction with padding (unigrams, bigrams, trigrams)
  - Frequency counting for n-grams
  - Raw probability calculations
  - Linear interpolation smoothing
  - Sentence probability computation
  - Perplexity calculation
- Text generation capabilities
- Essay classification system using perplexity scores

## Technical Details

### Language Model Components

1. **N-gram Extraction**
   - Implements padding with START/STOP tokens
   - Handles different n-gram sizes (1, 2, 3)
   - Returns tuples of tokens

2. **Probability Calculations**
   - Raw trigram probabilities: P(w₃|w₁,w₂)
   - Raw bigram probabilities: P(w₂|w₁)
   - Raw unigram probabilities: P(w₁)
   - Smoothed probabilities using linear interpolation (λ₁=λ₂=λ₃=1/3)

3. **Evaluation Metrics**
   - Log probability calculation for sentences
   - Perplexity computation for corpus evaluation
   - Classification accuracy for essay scoring

### Data Handling

- Supports corpus reading with automatic lexicon generation
- Handles unknown words using UNK token
- Processes both single-sentence and full-essay inputs

## Dataset

The project uses two main datasets:
1. Brown Corpus: American written English from the 1950s
2. ETS TOEFL Essays

## Usage

```python
# Initialize the model with a training corpus
model = TrigramModel("brown_train.txt")

# Generate a random sentence
sentence = model.generate_sentence()

# Calculate perplexity on test data
test_perplexity = model.perplexity(test_corpus)

# Classify essays
accuracy = essay_scoring_experiment(
    training_file_high,
    training_file_low,
    test_dir_high,
    test_dir_low
)
```
