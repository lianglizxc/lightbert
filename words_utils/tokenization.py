import collections
import jieba

class JiebaTokenizer():
    def __init__(self, stop_words, punctuations, vocab=None):
        self.stop_words = stop_words
        self.punctuation = set()
        for punctuation in punctuations:
            self.punctuation.update(set(punctuation))

        if isinstance(vocab, str):
            self.vocab = self.load_vocab(vocab)
        else:
            self.vocab = self.init_vocab(vocab)

        self.id_to_vocab = self.make_id_vocab()

    def init_vocab(self, vocab):
        init_vocab = {'<pad>': 0,
                      '<unk>': 1,
                      'CLS': 2,
                      'SEP': 3,
                      'MASK': 4}
        init_vocab = collections.OrderedDict(init_vocab)
        if vocab is not None:
            for word in vocab:
                init_vocab[word] = len(init_vocab)
        return init_vocab

    def tokenize(self, text):
        tokens = []
        for token in jieba.cut(text, cut_all=False):
            if token in self.stop_words:
                continue
            if token in self.punctuation:
                continue
            if token == ' ' or token == '\n' or token[0].isnumeric():
                continue
            if len(token) == 1 and token[0].isalpha():
                continue
            word = token.lower()
            tokens.append(word)
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        return tokens

    def save_vocab(self, path):
        with open(path, 'w', encoding='UTF-8') as file:
            for w, id, in self.vocab.items():
                file.write(w + '->' + str(id) + '\n')

    def load_vocab(self, path):
        init_vocab = collections.OrderedDict()
        with open(path, 'r',encoding='UTF-8') as file:
            for line in file:
                line = line.strip().split('->')
                assert len(line) == 2
                init_vocab[line[0]] = int(line[1])
        return init_vocab

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.id_to_vocab, ids)

    def make_id_vocab(self):
        self.id_to_vocab = {v: k for k, v in self.vocab.items()}
        return self.id_to_vocab


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