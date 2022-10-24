from transformers import BertTokenizer, BartForConditionalGeneration, SummarizationPipeline
from words_utils.tokenization import ChineseTokenizer, read_stop_words
import words_utils.tokenization as finance_tokenize
import pandas as pd

def get_sample_summary():
    tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
    model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
    summary_generator = SummarizationPipeline(model, tokenizer)

    finance_data = pd.read_excel('finance_data/SmoothNLP专栏资讯数据集样本10k.xlsx')
    summary = summary_generator(finance_data['content'].iloc[0][:512], max_length=512, do_sample=False)
    return summary


def print_tokens(tokens, num_per_line = 10):
    i = 0
    while i < len(tokens):
        output=' # '.join(tokens[i:i+num_per_line])
        i=i+num_per_line
        print(output)


def get_chinese_tokenizer():
    finance_news = pd.read_excel('finance_data/SmoothNLP专栏资讯数据集样本10k.xlsx')
    stop_words = read_stop_words()

    tokenizer = ChineseTokenizer(stop_words)
    tokens = tokenizer.tokenize(finance_news['content'].iloc[0])
    print(len(tokens))

    tokenizer = ChineseTokenizer()
    tokens = tokenizer.tokenize(finance_news['content'].iloc[0])
    print(len(tokens))


def save_vocab_count():
    finance_news = pd.read_excel('finance_data/SmoothNLP专栏资讯数据集样本10k.xlsx')
    finance_news = finance_news[~finance_news['content'].isna()]
    finance_news = finance_news[~finance_news['title'].isna()]
    tokenizer = ChineseTokenizer()
    for index, row in finance_news.iterrows():
        tokenizer.update_vocab(row['content'])
        tokenizer.update_vocab(row['title'])

    finance_tokenize.save_vocab('finance_data/ch_vocab', tokenizer.vocab)
    finance_tokenize.save_count('finance_data/ch_vocab_count', tokenizer.count)


def get_bart_tokenizer():
    from transformers import BartTokenizer, BartForConditionalGeneration, SummarizationPipeline
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    summary_generator = SummarizationPipeline(model, tokenizer)
    tokens = summary_generator('This is the configuration class to store the configuration of a BartModel. It is used to instantiate a BART model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the BART facebook/bart-large architecture.')
    print(tokens)


if __name__ == '__main__':
    get_bart_tokenizer()
