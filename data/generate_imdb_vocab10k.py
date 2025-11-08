import numpy as np
import os
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from collections import Counter
import re

TOP_WORDS = 10000
MAX_LEN = 290

###### create features ######

class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab, **kwargs):
        super().__init__(**kwargs)
        self.vocab = vocab
        self.ids_to_tokens = {id: token for token, id in vocab.items()}
        self.add_special_tokens({
            'unk_token': '[UNK]',
            'pad_token': '[PAD]',
        })

    def _tokenize(self, text):
        # simple whitespace tokenization - customize as needed
        return re.findall(r'\w+', text.lower())

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def get_vocab(self):
        return self.vocab

    def save_vocabulary(self, save_directory):
        vocab_file = os.path.join(save_directory, "vocab.txt")
        with open(vocab_file, "w") as f:
            for token, id in self.vocab.items():
                f.write(f'{id} {token}\n')
        return (vocab_file,)


def build_vocab(dataset, vocab_size=TOP_WORDS):
    """Build vocabulary from IMDB training dataset"""
    counter = Counter()
    for example in dataset:
        text = example['text']
        # simple whitespace tokenization for vocabulary building
        tokens = re.findall(r'\w+', text.lower())
        counter.update(tokens)
    
    vocab = {word: i+2 for i, (word, _) in enumerate(counter.most_common(vocab_size))}
    vocab['[PAD]'] = 0
    vocab['[UNK]'] = 1
    return vocab


def tokenize_and_pad(tokenizer, example, key_name, max_length=MAX_LEN):
    """Tokenize and pad/truncate a single example"""
    encoding = tokenizer(
        example[key_name],
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_token_type_ids=False
    )
    length = sum(encoding['attention_mask']) # actual length of each sequence
    return {'input_ids': encoding['input_ids'], 'length': length}


train_dataset, test_dataset = load_dataset('imdb', split=['train', 'test'])
test_dataset = test_dataset.shuffle().select(range(2000))
public_dataset = load_dataset('amazon_polarity', split=['train'])
public_dataset = public_dataset[0].shuffle().select(range(1000)) # 4% of train size

train_size = len(train_dataset)
test_size = len(test_dataset)
public_size = len(public_dataset)
print('train size:', train_size, 'test size:', test_size, 'public size:', public_size)
print('public 0', sum(np.array(public_dataset['label'])==0), 'public 1', sum(np.array(public_dataset['label'])==1))
print('test 0', sum(np.array(test_dataset['label'])==0), 'test 1', sum(np.array(test_dataset['label'])==1))


# build vocabulary from IMDB training data
vocab = build_vocab(train_dataset, vocab_size=TOP_WORDS)

tokenizer = CustomTokenizer(
    vocab=vocab,
    model_max_length=MAX_LEN,
    padding_side='right',
    truncation=True,
    pad_token='[PAD]',
    unk_token='[UNK]',
    clean_text=True,
    lowercase=True,
    tokenize_chinese_chars=True,
    strip_accents=True,
)
tokenizer.save_vocabulary(os.path.join(os.path.dirname(__file__), 'imdb10k'))

def process_dataset(dataset, key_name):
    return dataset.map(lambda x: tokenize_and_pad(tokenizer, x, key_name), batched=False)

train_dataset = process_dataset(train_dataset, 'text')
test_dataset =  process_dataset(test_dataset, 'text')
public_dataset = process_dataset(public_dataset, 'content')

path = os.path.dirname(__file__)
for p in ['train', 'test', 'public']:
    data = eval(p + '_dataset')
    np.savez(os.path.join(path, f'imdb10k/{p}'), x=data['input_ids'], y=data['label'], length=data['length'])
print('tokenized raw data saved')


###### decode some sequences ######
inverted_vocab = dict((i, word) for (word, i) in vocab.items())

for p in ['train', 'test', 'public']:
    data = np.load(os.path.join(os.path.dirname(__file__), f'imdb10k/{p}.npz'))
    for i in range(3):
        x, y = data['x'][i], data['y'][i]
        decoded_x = " ".join(inverted_vocab[i] for i in x)
        print(x.shape, y, x, decoded_x)