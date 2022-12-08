# should use this to concat as a feature on every window to LSTM
# precompute all log avg values for windows?
import pandas as pd
import numpy as np
from collections import Counter


class DataLoader():
    def __init__(self, filename : str):
        self.df = pd.read_csv(filename)

class WordFreqDataLoader(DataLoader):
    def __init__(self, *args):
        super(WordFreqDataLoader, self).__init__(*args)
        
        # reusable metrics
        self.total_count = self.df['count'].sum()
        self.oov_count = max(self.df['count'].min() - 1, 1.0)
        # create word_freq column
        self.df['word_freq'] = self.df.apply(lambda x: x['count'] / self.total_count, axis=1)

    def get_log(self, word):
        masked = self.df[self.df['word'] == word]
        if len(masked) == 0:
            return -1 * np.log(self.oov_count / self.total_count) # assign it smallest possible value
        return -1 * np.log(self.df[self.df['word'] == word]['word_freq'].iloc[0])


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
    
