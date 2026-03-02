import pandas as pd
from datasets import load_dataset, Value, Dataset, ClassLabel, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import config  # 项目配置文件
import pke

def extract_keyphrases(text):
    """保留原有关键词提取逻辑"""

    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(input=text, language='zh')
    extractor.candidate_selection()
    extractor.candidate_weighting(threshold=0.74, method='average')
    keyphrases = extractor.get_n_best(n=10)
    return keyphrases

def clean_text(text):
    """保留原有文本清洗逻辑"""
    text = text.replace("\n", " ").replace("\t", " ")
    text = text.strip()
    return text

