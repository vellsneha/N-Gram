"""
Homework 2:N-Gram Language Modeling and Evaluation


Implemented various N-gram language models with different
smoothing and backoff strategies, and evaluated them using perplexity.
"""

import re
import math
import random
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
import numpy as np


class NGramLanguageModel:
    """Base class for N-gram language models."""
    
    def __init__(self, n: int):
        """
        Initialize N-gram model.
        
        Args:
            n: Order of the N-gram (1 for unigram, 2 for bigram, etc.)
        """
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab = set()
        self.vocab_size = 0
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text: tokenization and normalization.
        
        For PTB dataset:
        - Text is already tokenized (space-separated)
        - Contains <unk> tokens for unknown words
        - No additional tokenization needed
        
        Args:
            text: Text string (already tokenized in PTB format)
            
        Returns:
            List of tokens
        """
        # PTB data is already tokenized, just split on whitespace
        tokens = text.split()
        
        # Filter out empty tokens
        tokens = [t for t in tokens if t]
        
        return tokens
    
    def add_sentence_boundaries(self, tokens: List[str]) -> List[str]:
        """
        Add sentence start and end markers.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Tokens with <s> and </s> markers
        """
        # Add n-1 start tokens for N-gram context
        start_tokens = ['<s>'] * (self.n - 1)
        end_token = ['</s>']
        return start_tokens + tokens + end_token
    
    def get_ngrams(self, tokens: List[str]) -> List[Tuple[str, ...]]:
        """
        Extract N-grams from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of N-gram tuples
        """
        ngrams = []
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            ngrams.append(ngram)
        return ngrams
    
    def train(self, corpus: List[str], verbose: bool = True):
        """
        Train the N-gram model on a corpus.
        
        Args:
            corpus: List of sentences (strings)
            verbose: Whether to print training progress
        """
        if verbose:
            print(f"Training {self.n}-gram model")
        
        for sentence in corpus:
            # Preprocess and add boundaries
            tokens = self.preprocess_text(sentence)
            tokens = self.add_sentence_boundaries(tokens)
            
            # Build vocabulary
            self.vocab.update(tokens)
            
            # Count N-grams
            ngrams = self.get_ngrams(tokens)
            for ngram in ngrams:
                self.ngram_counts[ngram] += 1
                # Context is all but the last word
                if self.n > 1:
                    context = ngram[:-1]
                    self.context_counts[context] += 1
        
        self.vocab_size = len(self.vocab)
        if verbose:
            print(f"✓ Vocabulary size: {self.vocab_size:,}")
            print(f"✓ Total {self.n}-grams: {len(self.ngram_counts):,}")
    
    def probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Calculate probability of an N-gram (to be overridden by subclasses).
        
        Args:
            ngram: Tuple of n tokens
            
        Returns:
            Probability value
        """
        raise NotImplementedError
    
    def perplexity(self, test_corpus: List[str]) -> float:
        """
        Calculate perplexity on test corpus.
        
        Args:
            test_corpus: List of test sentences
            
        Returns:
            Perplexity value
        """
        log_prob_sum = 0
        token_count = 0
        
        for sentence in test_corpus:
            tokens = self.preprocess_text(sentence)
            tokens = self.add_sentence_boundaries(tokens)
            
            ngrams = self.get_ngrams(tokens)
            
            for ngram in ngrams:
                prob = self.probability(ngram)
                
                # If probability is zero, return infinity
                if prob == 0:
                    return float('inf')
                
                log_prob_sum += math.log2(prob)
                token_count += 1
        
        # Perplexity = 2^(-1/N * sum(log2(p)))
        if token_count == 0:
            return float('inf')
        
        avg_log_prob = log_prob_sum / token_count
        perplexity = 2 ** (-avg_log_prob)
        
        return perplexity


class MLEModel(NGramLanguageModel):
    """Maximum Likelihood Estimation N-gram model."""
    
    def probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Calculate MLE probability: P(w|context) = Count(context, w) / Count(context)
        
        Args:
            ngram: Tuple of n tokens
            
        Returns:
            MLE probability
        """
        if self.n == 1:
            # Unigram: P(w) = Count(w) / Total tokens
            total = sum(self.ngram_counts.values())
            return self.ngram_counts[ngram] / total if total > 0 else 0
        else:
            # N-gram: P(w|context) = Count(context, w) / Count(context)
            context = ngram[:-1]
            context_count = self.context_counts[context]
            
            if context_count == 0:
                return 0
            
            return self.ngram_counts[ngram] / context_count


class AddOneModel(NGramLanguageModel):
    """Add-1 (Laplace) Smoothing model."""
    
    def probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Calculate Add-1 smoothed probability.
        P(w|context) = (Count(context, w) + 1) / (Count(context) + V)
        
        Args:
            ngram: Tuple of n tokens
            
        Returns:
            Smoothed probability
        """
        if self.n == 1:
            # Unigram with Add-1
            total = sum(self.ngram_counts.values())
            return (self.ngram_counts[ngram] + 1) / (total + self.vocab_size)
        else:
            # N-gram with Add-1
            context = ngram[:-1]
            context_count = self.context_counts[context]
            ngram_count = self.ngram_counts[ngram]
            
            return (ngram_count + 1) / (context_count + self.vocab_size)


class InterpolationModel:
    """Linear Interpolation model combining unigram, bigram, and trigram."""
    
    def __init__(self, lambda1: float, lambda2: float, lambda3: float):
        """
        Initialize interpolation model.
        
        Args:
            lambda1: Weight for unigram
            lambda2: Weight for bigram
            lambda3: Weight for trigram
        """
        assert abs(lambda1 + lambda2 + lambda3 - 1.0) < 1e-6, "Lambdas must sum to 1"
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        
        # Create component models
        self.unigram = MLEModel(1)
        self.bigram = MLEModel(2)
        self.trigram = MLEModel(3)
        
        self.n = 3  # We're building a trigram model
        self.vocab = set()
        self.vocab_size = 0
    
    def train(self, corpus: List[str]):
        """Train all component models."""
        # Train each component with minimal output
        self.unigram.train(corpus, verbose=False)
        self.bigram.train(corpus, verbose=False)
        self.trigram.train(corpus, verbose=False)
        
        # Use trigram's vocabulary
        self.vocab = self.trigram.vocab
        self.vocab_size = self.trigram.vocab_size
    
    def probability(self, trigram: Tuple[str, ...]) -> float:
        """
        Calculate interpolated probability.
        P(w|w1,w2) = λ1*P(w) + λ2*P(w|w1) + λ3*P(w|w1,w2)
        
        Args:
            trigram: Tuple of 3 tokens
            
        Returns:
            Interpolated probability
        """
        word = (trigram[2],)  # Unigram
        bigram = trigram[1:]  # Bigram
        
        p1 = self.unigram.probability(word)
        p2 = self.bigram.probability(bigram)
        p3 = self.trigram.probability(trigram)
        
        return self.lambda1 * p1 + self.lambda2 * p2 + self.lambda3 * p3
    
    def perplexity(self, test_corpus: List[str]) -> float:
        """Calculate perplexity using interpolation."""
        log_prob_sum = 0
        token_count = 0
        
        for sentence in test_corpus:
            tokens = self.trigram.preprocess_text(sentence)
            tokens = self.trigram.add_sentence_boundaries(tokens)
            
            ngrams = self.trigram.get_ngrams(tokens)
            
            for ngram in ngrams:
                prob = self.probability(ngram)
                
                if prob == 0:
                    return float('inf')
                
                log_prob_sum += math.log2(prob)
                token_count += 1
        
        if token_count == 0:
            return float('inf')
        
        avg_log_prob = log_prob_sum / token_count
        return 2 ** (-avg_log_prob)


class StupidBackoffModel:
    """Stupid Backoff model."""
    
    def __init__(self, alpha: float = 0.4):
        """
        Initialize Stupid Backoff model.
        
        Args:
            alpha: Backoff discount factor (commonly 0.4)
        """
        self.alpha = alpha
        
        # Create component models
        self.unigram = MLEModel(1)
        self.bigram = MLEModel(2)
        self.trigram = MLEModel(3)
        
        self.n = 3
        self.vocab = set()
        self.vocab_size = 0
    
    def train(self, corpus: List[str]):
        """Train all component models."""
        # Train each component with minimal output
        self.unigram.train(corpus, verbose=False)
        self.bigram.train(corpus, verbose=False)
        self.trigram.train(corpus, verbose=False)
        
        self.vocab = self.trigram.vocab
        self.vocab_size = self.trigram.vocab_size
    
    def score(self, ngram: Tuple[str, ...], level: int = 3) -> float:
        """
        Calculate Stupid Backoff score (not a probability).
        
        Args:
            ngram: Tuple of tokens
            level: Current N-gram level (3, 2, or 1)
            
        Returns:
            Score value
        """
        if level == 3:
            # Try trigram
            context = ngram[:-1]
            context_count = self.trigram.context_counts[context]
            
            if context_count > 0:
                return self.trigram.ngram_counts[ngram] / context_count
            else:
                # Backoff to bigram
                return self.alpha * self.score(ngram[1:], level=2)
        
        elif level == 2:
            # Try bigram
            context = (ngram[0],)
            context_count = self.bigram.context_counts[context]
            
            if context_count > 0:
                bigram = ngram
                return self.bigram.ngram_counts[bigram] / context_count
            else:
                # Backoff to unigram
                return self.alpha * self.score((ngram[1],), level=1)
        
        else:  # level == 1
            # Unigram
            total = sum(self.unigram.ngram_counts.values())
            return self.unigram.ngram_counts[ngram] / total if total > 0 else 1e-10
    
    def probability(self, trigram: Tuple[str, ...]) -> float:
        """
        Calculate probability using Stupid Backoff.
        
        Args:
            trigram: Tuple of 3 tokens
            
        Returns:
            Score (treated as probability for perplexity calculation)
        """
        return self.score(trigram, level=3)
    
    def perplexity(self, test_corpus: List[str]) -> float:
        """Calculate perplexity using Stupid Backoff scores."""
        log_score_sum = 0
        token_count = 0
        
        for sentence in test_corpus:
            tokens = self.trigram.preprocess_text(sentence)
            tokens = self.trigram.add_sentence_boundaries(tokens)
            
            ngrams = self.trigram.get_ngrams(tokens)
            
            for ngram in ngrams:
                score = self.probability(ngram)
                
                if score == 0:
                    score = 1e-10  # Use small value instead of zero
                
                log_score_sum += math.log2(score)
                token_count += 1
        
        if token_count == 0:
            return float('inf')
        
        avg_log_score = log_score_sum / token_count
        return 2 ** (-avg_log_score)


class TextGenerator:
    """Generate text using a trained language model."""
    
    def __init__(self, model):
        """
        Initialize text generator.
        
        Args:
            model: Trained language model
        """
        self.model = model
        self.n = model.n
    
    def generate_sentence(self, max_length: int = 20) -> str:
        """
        Generate a sentence using the language model.
        
        Args:
            max_length: Maximum number of tokens to generate
            
        Returns:
            Generated sentence string
        """
        # Start with context
        if self.n == 1:
            tokens = []
        else:
            tokens = ['<s>'] * (self.n - 1)
        
        for _ in range(max_length):
            # Get context for prediction
            if self.n == 1:
                context = tuple()
            else:
                context = tuple(tokens[-(self.n-1):])
            
            # Find all possible next words
            candidates = []
            probabilities = []
            
            for word in self.model.vocab:
                if word not in ['<s>', '</s>']:
                    if self.n == 1:
                        ngram = (word,)
                    else:
                        ngram = context + (word,)
                    
                    prob = self.model.probability(ngram)
                    
                    if prob > 0:
                        candidates.append(word)
                        probabilities.append(prob)
            
            if not candidates:
                break
            
            # Normalize probabilities
            total_prob = sum(probabilities)
            probabilities = [p / total_prob for p in probabilities]
            
            # Sample next word
            next_word = np.random.choice(candidates, p=probabilities)
            
            if next_word == '</s>':
                break
            
            tokens.append(next_word)
        
        # Remove start tokens and return
        generated = [t for t in tokens if t != '<s>']
        return ' '.join(generated)


def load_data(filepath: str) -> List[str]:
    """
    Load PTB data from file.
    
    PTB format:
    - Each line is already tokenized (space-separated words)
    - Contains <unk> for unknown/rare words
    - One sentence per line
    
    Args:
        filepath: Path to PTB data file
        
    Returns:
        List of sentences (already tokenized)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Filter empty lines
    sentences = [line.strip() for line in lines if line.strip()]
    
    print(f"Loaded {len(sentences)} sentences from {filepath}")
    return sentences


def tune_interpolation_weights(train_data: List[str], dev_data: List[str]) -> Tuple[float, float, float]:
    """
    Find optimal interpolation weights using development data.
    
    Args:
        train_data: Training sentences
        dev_data: Development sentences
        
    Returns:
        Tuple of (lambda1, lambda2, lambda3)
    """
    print("Tuning interpolation weights on development set")
    print("┌" + "─"*28 + "┬" + "─"*16 + "┐")
    print("│ Weight Combinations        │ Dev Perplexity │")
    print("├" + "─"*28 + "┼" + "─"*16 + "┤")
    
    best_perplexity = float('inf')
    best_weights = (0.33, 0.33, 0.34)
    
    # Try different weight combinations (assignment requires at least 3)
    weight_combinations = [
        (0.1, 0.3, 0.6),   # Trigram-heavy
        (0.2, 0.3, 0.5),   # Balanced
        (0.1, 0.4, 0.5),   # Bigram-heavy
        (0.33, 0.33, 0.34), # Equal weights
        (0.2, 0.4, 0.4),   # Balanced bigram/trigram
        (0.15, 0.35, 0.5), # Bigram-focused
        (0.05, 0.25, 0.7), # Very trigram-heavy
        (0.3, 0.3, 0.4),   # Unigram/bigram focused
    ]
    
    for lambda1, lambda2, lambda3 in weight_combinations:
        model = InterpolationModel(lambda1, lambda2, lambda3)
        model.train(train_data)
        pp = model.perplexity(dev_data)
        
        # Format the weights nicely
        weight_str = f"λ1={lambda1:.2f}, λ2={lambda2:.2f}, λ3={lambda3:.2f}"
        print(f"│ {weight_str:<26} │ {pp:>14.2f} │")
        
        if pp < best_perplexity:
            best_perplexity = pp
            best_weights = (lambda1, lambda2, lambda3)
    
    print("└" + "─"*28 + "┴" + "─"*16 + "┘")
    print(f"✅ Best weights: λ1={best_weights[0]:.2f}, λ2={best_weights[1]:.2f}, λ3={best_weights[2]:.2f}")
    print(f"✅ Best dev perplexity: {best_perplexity:.2f}")
    
    return best_weights


def tune_backoff_alpha(train_data: List[str], dev_data: List[str]) -> float:
    """
    Find optimal alpha parameter for Stupid Backoff using development data.
    Assignment requirement: "find an optimal α using the Dev Data"
    
    Args:
        train_data: Training sentences
        dev_data: Development sentences
        
    Returns:
        Optimal alpha value
    """
    print("\nTuning Stupid Backoff alpha parameter")
    print("┌" + "─"*15 + "┬" + "─"*16 + "┐")
    print("│ Alpha Value   │ Dev Perplexity  │")
    print("├" + "─"*15 + "┼" + "─"*16 + "┤")
    
    best_perplexity = float('inf')
    best_alpha = 0.4
    
    # Try different alpha values (common choices)
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    for alpha in alpha_values:
        model = StupidBackoffModel(alpha=alpha)
        model.train(train_data)
        pp = model.perplexity(dev_data)
        
        print(f"│ {alpha:>13.1f} │ {pp:>14.2f} │")
        
        if pp < best_perplexity:
            best_perplexity = pp
            best_alpha = alpha
    
    print("└" + "─"*15 + "┴" + "─"*16 + "┘")
    print(f"Best alpha: {best_alpha} (Dev Perplexity: {best_perplexity:.2f})")
    
    return best_alpha


def main():
    """Main execution function."""
    
    print("="*60)
    print("N-Gram Language Modeling and Evaluation")
    print("="*60)
    

    
    # Load data
    print("\nLoading data")
    train_data = load_data('ptb.train.txt')
    dev_data = load_data('ptb.valid.txt')
    test_data = load_data('ptb.test.txt')
    
    print(f"Train sentences: {len(train_data)}")
    print(f"Dev sentences: {len(dev_data)}")
    print(f"Test sentences: {len(test_data)}")
    
    # Use subset for faster experimentation (remove for full run)
    # train_data = train_data[:1000]
    # dev_data = dev_data[:200]
    # test_data = test_data[:200]
    
    results = {}
    
    # ==================== PART 1: MLE Models ====================
    print("\n" + "="*60)
    print("PART 1: Maximum Likelihood Estimation Models")
    print("="*60)
    print("┌" + "─"*28 + "┬" + "─"*17 + "┐")
    print("│ Model                      │ Test Perplexity │")
    print("├" + "─"*28 + "┼" + "─"*17 + "┤")
    
    for n in [1, 2, 3, 4]:
        print(f"│ Training {n}-gram MLE", end="", flush=True)
        model = MLEModel(n)
        model.train(train_data)
        pp = model.perplexity(test_data)
        results[f'MLE-{n}gram'] = pp
        pp_str = f"{pp:.2f}" if pp != float('inf') else 'INF'
        print(f"\r│ {n}-gram MLE Model           │  {pp_str:>14} │")
    
    print("└" + "─"*28 + "┴" + "─"*17 + "┘")
    
    # ==================== PART 2: Smoothing ====================
    print("\n" + "="*60)
    print("PART 2: Add-1 Smoothing")
    print("="*60)
    
    print("│ Training Trigram with Add-1 Smoothing ", end="", flush=True)
    add1_model = AddOneModel(3)
    add1_model.train(train_data)
    pp = add1_model.perplexity(test_data)
    results['Add-1'] = pp
    print(f"\r│ Add-1 Smoothing (3-gram)   │ {pp:>14.2f} │")
    
    # ==================== PART 3: Interpolation ====================
    print("\n" + "="*60)
    print("PART 3: Linear Interpolation")
    print("="*60)
    
    # Tune weights on dev set
    best_weights = tune_interpolation_weights(train_data, dev_data)
    
    # Train final model with best weights
    print("\n│ Training final interpolation model", end="", flush=True)
    interp_model = InterpolationModel(*best_weights)
    interp_model.train(train_data)
    pp = interp_model.perplexity(test_data)
    results['Interpolation'] = pp
    print(f"\r│ Linear Interpolation      │ {pp:>14.2f} │")
    
    # ==================== PART 4: Stupid Backoff ====================
    print("\n" + "="*60)
    print("PART 4: Stupid Backoff")
    print("="*60)
    
    # Tune alpha parameter on dev set (assignment requirement)
    best_alpha = tune_backoff_alpha(train_data, dev_data)
    
    print("\n│ Training Stupid Backoff with optimal alpha", end="", flush=True)
    backoff_model = StupidBackoffModel(alpha=best_alpha)
    backoff_model.train(train_data)
    pp = backoff_model.perplexity(test_data)
    results['Stupid Backoff'] = pp
    print(f"\r│ Stupid Backoff (α={best_alpha})      │ {pp:>14.2f} │")
    
    # ==================== PART 5: Results Summary ====================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    print("\n🏆 Test Set Perplexity Results:")
    print("┌" + "─"*32 + "┬" + "─"*16 + "┬" + "─"*8 + "┐")
    print("│ Model                          │ Perplexity     │  Rank  │")
    print("├" + "─"*32 + "┼" + "─"*16 + "┼" + "─"*8 + "┤")
    
    # Sort results by perplexity (lower is better)
    sorted_results = sorted(results.items(), key=lambda x: x[1] if x[1] != float('inf') else float('inf'))
    
    for rank, (model_name, pp) in enumerate(sorted_results, 1):
        pp_str = f"{pp:.2f}" if pp != float('inf') else 'INF'
        print(f"│ {model_name:<28}   │ {pp_str:>14} │  {rank:>4d}  │")
    
    print("└" + "─"*32 + "┴" + "─"*16 + "┴" + "─"*8 + "┘")
    
    # Find best model
    best_model_name, best_pp = sorted_results[0]
    best_pp_str = f"{best_pp:.2f}" if best_pp != float('inf') else 'INF'
    print(f"\n🥇 Best Model: {best_model_name} (Perplexity: {best_pp_str})")
    
    # ==================== PART 6: Text Generation ====================
    print("\n" + "="*60)
    print("TEXT GENERATION")
    print("="*60)
    
    # Use best performing model for generation
    best_model_name = min(results.items(), key=lambda x: x[1] if x[1] != float('inf') else float('inf'))[0]
    
    if 'Interpolation' in best_model_name:
        gen_model = interp_model
    elif 'Backoff' in best_model_name:
        gen_model = backoff_model
    else:
        gen_model = add1_model
    
    print(f"\n Generating sentences using best model: {best_model_name}")
    generator = TextGenerator(gen_model)
    
    print("\n Generated Sentences:")
    
    for i in range(5):
        sentence = generator.generate_sentence(max_length=15)
        print(f"{i+1:>2d} . {sentence:<63} ")
   
    
    print("\n" + "="*60)
    print("Analysis complete! All models evaluated successfully.")
    print("="*60)


if __name__ == "__main__":
    main()