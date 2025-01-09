import sys
from collections import defaultdict
import math
import random
import os
import os.path

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    if n == 1:
        padded_sequence = ["START"] + sequence + ["STOP"]
    else:
        padded_sequence = ["START"] * (n-1) + sequence + ["STOP"]
    
    ngrams = []
    for i in range(len(padded_sequence) - n + 1):
        ngrams.append(tuple(padded_sequence[i:i+n]))

    return ngrams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        Populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.trigram_counts = defaultdict(int)
        self.total_words = 0

        for sentence in corpus:
            unigrams = get_ngrams(sentence, 1)
            bigrams = get_ngrams(sentence, 2)
            trigrams = get_ngrams(sentence, 3)
            
            for unigram in unigrams:
                self.unigram_counts[unigram] += 1
            
            for bigram in bigrams:
                self.bigram_counts[bigram] += 1

            for trigram in trigrams:
                self.trigram_counts[trigram] += 1

            self.total_words += len(sentence)

    def raw_trigram_probability(self,trigram):
        """
        Returns the raw (unsmoothed) trigram probability
        """
        if trigram in self.trigram_counts:
            trigram_count = self.trigram_counts[trigram]
        else:
            trigram_count = 0

        bigram = (trigram[0], trigram[1])
        if bigram in self.bigram_counts:
            bigram_count = self.bigram_counts[bigram]
        else:
            bigram_count = 0
            
        if bigram_count == 0:
            return 1 / self.total_words
        else:
            return trigram_count / bigram_count

    def raw_bigram_probability(self, bigram):
        """
        Returns the raw (unsmoothed) bigram probability
        """
        if bigram in self.bigram_counts:
            bigram_count = self.bigram_counts[bigram]
        else:
            bigram_count = 0

        unigram = (bigram[0],)
        if unigram in self.unigram_counts:
            unigram_count = self.unigram_counts[unigram]
        else:
            unigram_count = 0
            
        if unigram_count == 0:
            return 1 / self.total_words
        else:
            return bigram_count / unigram_count

    
    def raw_unigram_probability(self, unigram):
        """
        Returns the raw (unsmoothed) unigram probability.
        """
        if unigram in self.unigram_counts:
            return self.unigram_counts[unigram] / self.total_words
        else:
            return 0.0
        
    def generate_sentence(self,t=20): 
        """
        Generates a random sentence from the trigram model.
        """
        sentence = []
        current_bigram = ("START", "START")
        for i in range(t):
            next_words = []
            for trigram in self.trigram_counts:
                if trigram[:2] == current_bigram:
                    next_words.append(trigram[2])
            if not next_words:
                break
            probs = [self.raw_trigram_probability[(current_bigram, word)] for word in next_words]
            next_word = random.choices(next_words, weights=probs)[0]
            sentence.append(next_word)
            if next_word == "STOP":
                break
            else:
                current_bigram = (current_bigram[1], next_word)
        return sentence

    def smoothed_trigram_probability(self, trigram):
        """
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        bigram = (trigram[1], trigram[2])
        unigram = (trigram[2],)

        raw_trigram_prob = self.raw_trigram_probability(trigram)
        raw_bigram_prob = self.raw_bigram_probability(bigram)
        raw_unigram_prob = self.raw_unigram_probability(unigram)

        smoothed_prob = lambda1 * raw_trigram_prob + lambda2 * raw_bigram_prob + lambda3 * raw_unigram_prob

        return smoothed_prob
        
    def sentence_logprob(self, sentence):
        """
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        log_prob = 0
        for trigram in trigrams:
            smoothed_prob = self.smoothed_trigram_probability(trigram)
            if smoothed_prob > 0:
                log_prob += math.log2(smoothed_prob)

        return log_prob

    def perplexity(self, corpus):
        """
        Returns the log probability of an entire sequence.
        """
        word_count = 0
        log_prob = 0
        for sentence in corpus:
            log_prob += self.sentence_logprob(sentence)
            word_count += len(sentence)

        l_exponent = log_prob / word_count
        perplexity = 2 ** (-l_exponent)
        return perplexity

def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp1 < pp2:
                correct += 1
            total += 1
    
        for f in os.listdir(testdir2):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            if pp2 < pp1:
                correct += 1
            total += 1             
            
        return correct / total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    train_corpus = corpus_reader("hw1_data/brown_train.txt", model.lexicon)
    train_pp = model.perplexity(train_corpus)
    print("Perplexity on Training data:", train_pp)

    test_corpus = corpus_reader("hw1_data/brown_test.txt", model.lexicon)
    test_pp = model.perplexity(test_corpus)
    print("Perplexity on Test data:", test_pp)

    accuracy = essay_scoring_experiment("hw1_data/ets_toefl_data/train_high.txt", "hw1_data/ets_toefl_data/train_low.txt", "hw1_data/ets_toefl_data/test_high", "hw1_data/ets_toefl_data/test_low")
    print("Essay Scoring Accuracy:", accuracy)