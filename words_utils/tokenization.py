import collections
import jieba
import re

class JiebaTokenizer():
    def __init__(self, stop_words, punctuations, vocab=None):
        self.stop_words = stop_words
        self.init_vocab = {'<pad>': 0,
                          '<unk>': 1,
                          '[CLS]': 2,
                          '[SEP]': 3,
                          '[MASK]': 4}
        self.punctuation_list = punctuations
        self.punctuation = set()
        for punctuation in punctuations:
            self.punctuation.update(punctuation)

        if isinstance(vocab, str):
            self.vocab = self.load_vocab(vocab)
        else:
            self.vocab = self.build_vocab(vocab)

        self.id_to_vocab = self.make_id_vocab()
        self.count = {}

    def build_vocab(self, add_vocab):
        vocab = collections.OrderedDict(self.init_vocab)
        if add_vocab is not None:
            for word in add_vocab:
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab

    def tokenize(self, text):
        tokens = []
        for token in jieba.cut(text, cut_all=False):
            word = self.preprocess_token(token)
            if word is None: continue
            if word in self.vocab:
                tokens.append(word)
        return tokens

    def update_vocab(self, text):
        for token in jieba.cut(text, cut_all=False):
            word = self.preprocess_token(token)
            if word is None: continue
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
            if word not in self.count:
                self.count[word] = 0
            self.count[word] += 1

    def preprocess_token(self, token):
        if token in self.stop_words:
            return None
        if token in self.punctuation:
            return None
        if token == ' ' or token == '\n' or \
                token[0].isnumeric() or '__' in token:
            return None
        if len(token) == 1 and token[0].isalpha():
            return None
        return token.lower()

    def clean_text(self, content):
        content = content.translate(str.maketrans('', '', self.punctuation_list))
        content = re.sub('\xa0 ?|\u3000+', '', content)
        content = re.sub(' ?\n+', '\n', content)
        content = content.strip('\n')
        return content

    def save_count(self, path):
        items = list(self.count.items())
        items = sorted(items, key=lambda x:x[1], reverse=True)
        with open(path, 'w', encoding='UTF-8') as file:
            for w, count in items:
                file.write(w + '->' + str(count) + '\n')

    def save_vocab(self, path):
        with open(path, 'w', encoding='UTF-8') as file:
            for w, id, in self.vocab.items():
                file.write(w + '->' + str(id) + '\n')

    @staticmethod
    def load_vocab(path):
        vocab = collections.OrderedDict()
        with open(path, 'r',encoding='UTF-8') as file:
            for line in file:
                line = line.strip().split('->')
                assert len(line) == 2
                vocab[line[0]] = int(line[1])
        return vocab

    def truncate_vocab(self, count_path, max_number=50000):
        vocab_count = {}
        location = 0
        with open(count_path, 'r') as file:
            for line in file:
                word, _ = line.strip().split('->')
                vocab_count[word] = location
                location += 1
                if len(vocab_count) == max_number - len(self.init_vocab):
                    break

        vocab = self.init_vocab.copy()
        for w in self.vocab:
            if w in vocab_count:
                vocab[w] = len(self.init_vocab) + vocab_count[w]

        self.vocab = collections.OrderedDict(sorted(vocab.items(), key=lambda x: x[1]))
        assert len(self.vocab) == max_number

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.id_to_vocab, ids)

    def make_id_vocab(self):
        self.id_to_vocab = {v: k for k, v in self.vocab.items()}
        return self.id_to_vocab


def read_vocab_count(count_path, max_number):
    vocab_count = {}
    location = 0
    with open(count_path, 'r', encoding='UTF-8') as file:
        for line in file:
            word, _ = line.strip().split('->')
            vocab_count[word] = location
            location += 1
            if len(vocab_count) == max_number:
                break
    return vocab_count


def convert_by_vocab(vocab, items):
  output = []
  for item in items:
    output.append(vocab[item])
  return output


def read_stop_words():
    stop_words = set()
    with open('words_utils/stopwords','r', encoding='UTF-8') as file:
        for word in file:
            word = word.strip()
            stop_words.add(word)
    return stop_words