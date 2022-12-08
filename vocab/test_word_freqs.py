from word_freqs import WordFreqDataLoader
import numpy as np
# testing
def test_init():
    wfdl = WordFreqDataLoader('../datasets/test_csv.csv')
    assert wfdl.total_count == 10
    assert len(wfdl.df) == 4
    assert wfdl.df['word'].iloc[0] == 'the'
    assert wfdl.oov_count == 1

def test_word_freq():
    wfdl = WordFreqDataLoader('../datasets/test_csv.csv')
    assert wfdl.df.word_freq.to_numpy().tolist() == [0.4, 0.3, 0.2, 0.1]

def test_log_avg():
    wfdl = WordFreqDataLoader('../datasets/test_csv.csv')
    assert wfdl.get_log_avg([]) == 0.0
    assert wfdl.get_log_avg(['the']) == -1 * np.log(0.4)
    assert wfdl.get_log_avg(['the', 'and']) == -1 * (np.log(0.4) + np.log(0.2)) / 2.0
    assert wfdl.get_log_avg(['the', 'the', 'and']) == -1* (2 * np.log(0.4) + np.log(0.2))/ 3.0

def test_not_in():
    wfdl = WordFreqDataLoader('../datasets/test_csv.csv')
    assert wfdl.get_log('ploopy') == -1 * np.log(0.1) # should be assigned count 1