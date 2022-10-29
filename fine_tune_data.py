import pandas as pd
import numpy as np
from gensim.summarization import summarizer
from words_utils.tokenization import ChineseTokenizer
from dataprocess.utils import truncate_input_tokens


def create_training_instance(data, max_encoder_length, max_decoder_length, vocab):
    np.random.shuffle(data)
    for document, title in data:
        document = truncate_input_tokens(document)
        input_encoder_ids = [vocab[word] for word in document]
        input_decoder_ids = [vocab[word] for word in title]

        while len(input_encoder_ids) < max_encoder_length:
            input_encoder_ids.append(0)

        while len(input_decoder_ids) < max_decoder_length:
            input_decoder_ids.append(0)


def get_fine_tune_data():
    finance_news = pd.read_excel('finance_data/SmoothNLP专栏资讯数据集样本10k.xlsx')
    finance_news = finance_news[(~finance_news['content'].isna()) & (~finance_news['title'].isna())].reset_index(drop=True)

    tokenizers = ChineseTokenizer()
    tokenizers.load_vocab('finance_data/ch_vocab_count')

    data = []
    title_count = []
    for index, row in finance_news.iterrows():
        try:
            summary = summarizer.summarize(row['content'])
        except Exception as e:
            print(e)
            continue
        if len(summary)==0:
            continue
        tokens = tokenizers.tokenize(summary)

        title = tokenizers.tokenize(row['title'])
        title_count.append(len(title))
        data.append((tokens, title))

    print(pd.DataFrame(title_count).describe())
    create_training_instance(data, max_encoder_length=512,
                             max_decoder_length=70, vocab=tokenizers.vocab)


if __name__ == '__main__':
    get_fine_tune_data()