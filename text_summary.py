from transformers import BertTokenizer, BartForConditionalGeneration, SummarizationPipeline
import pandas as pd

def get_sample_summary():
    tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
    model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
    summary_generator = SummarizationPipeline(model, tokenizer)

    finance_data = pd.read_excel('finance_data/SmoothNLP专栏资讯数据集样本10k.xlsx')
    summary = summary_generator(finance_data['content'].iloc[0][:512], max_length=512, do_sample=False)
    print(summary)


if __name__ == '__main__':
    get_sample_summary()
