# should use this to concat as a feature on every window to LSTM
# precompute all log avg values for windows?
import pandas as pd
import numpy as np
from collections import Counter


class DataLoader():
    def __init__(self, filename : str):
        self.df = pd.read_csv(filename)

class WordFreqDataLoader(DataLoader):
    def __init__(self, filename, scaling_constant=10):
        super(WordFreqDataLoader, self).__init__(filename)
        
        # reusable metrics
        self.scaling_constant = scaling_constant
        self.total_count = self.df['count'].sum()
        self.oov_count = max(self.df['count'].min() - 1, 1.0)
        self.lowest = -1 * np.log(self.oov_count / self.total_count)
        # create word_freq column
        self.df['word_freq'] = self.df.apply(lambda x: x['count'] / self.total_count, axis=1)
        self.df['log_freq'] = self.df.apply(lambda x: -1 * np.log(scaling_constant * x['word_freq']), axis=1)

        self.dict = dict(zip(self.df.word, self.df.log_freq))

    def get_log(self, word):
        return self.dict[word] if word in self.dict else self.lowest

    def get_list_logs(self, words : list):
        return np.array([self.get_log(word) for word in words], dtype=np.float32)

    def get_log_avg(self, words : list) -> float:
        """
        Deprecated method, not really used
        """
        count_words = Counter(words)
        if len(count_words) == 0:
            return 0 # no words, just give it 0
        else:
            sum = 0
            actual_count = 0
            for (word, count) in count_words.items():
                res = self.get_log(word)
                if res is not None:
                    actual_count += count
                    sum += count * res
            if actual_count > 0:
                return sum / actual_count
            return 0 # no words found
    
