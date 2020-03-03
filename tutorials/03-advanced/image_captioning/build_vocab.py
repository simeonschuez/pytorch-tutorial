import nltk
nltk.download('punkt')
import pickle
from collections import Counter

import configparser
import pandas as pd
import os
import json
config = configparser.ConfigParser()
config.read('config.ini')

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
    
def build_vocab(df, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    ids = df.index
    for i, id in enumerate(ids):
        caption = str(df.loc[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(caption_path, vocab_path, threshold):    
    
    with open(caption_path + 'captions_train2014.json') as file: 
        file = json.load(file)
        captions = pd.DataFrame(file['annotations']).set_index('id')
    
    vocab = build_vocab(df=captions, threshold=threshold)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))
    
if __name__ == '__main__':
    
    coco_data_dir = config['MSCOCO']['data-path']
    coco_annotations = coco_data_dir+'annotations/'
    out_dir = config['ALL']['output_dir']
    
    main(
        caption_path=coco_annotations,
        vocab_path=out_dir+'vocab.pkl',
        threshold=4
    )