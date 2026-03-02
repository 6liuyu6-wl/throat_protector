import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# 中文停用词（护嗓场景优化，过滤无意义词汇）
STOP_WORDS = set([
    "的", "了", "吗", "呢", "啊", "怎么", "怎么办", "如何", "什么", "我", "你", "他",
    "给", "能", "有", "是", "都", "就", "也", "很", "太", "非常", "总", "经常", "可以",
    "建议", "方法", "办法", "缓解", "治疗", "护嗓", "嗓子", "喉咙", "咽喉"
])

def clean_text(text):
    """
    统一文本清洗逻辑：训练和预测必须调用同一个方法
    """
    # 1. 去除特殊符号、标点、数字，只保留中文
    text = re.sub(r"[^\u4e00-\u9fa5]", " ", text)
    # 2. 去除多余空格
    text = re.sub(r"\s+", " ", text).strip()
    # 3. jieba分词
    words = jieba.lcut(text)
    # 4. 过滤停用词、单字
    words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    # 5. 返回空格拼接的分词结果（适配TF-IDF）
    return " ".join(words)

class VoiceCareTokenizer:
    def __init__(self):
        # 护嗓场景优化的TF-IDF向量化模型
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # 支持单字+二元词组，匹配更精准
            min_df=1,  # 最小文档频率，适配小数据集
            max_df=0.9,  # 过滤高频无意义词
            stop_words=list(STOP_WORDS)
        )
        self.is_trained = False

    def fit(self, text_list):
        """
        训练向量化模型：训练时调用，输入所有question的清洗后文本
        """
        # 统一清洗所有文本
        cleaned_texts = [clean_text(text) for text in text_list]
        # 训练TF-IDF模型
        self.vectorizer.fit(cleaned_texts)
        self.is_trained = True
        print(f"✅ 向量化模型训练完成，词汇量：{len(self.vectorizer.get_feature_names_out())}")

    def vectorize(self, text):
        """
        文本向量化：训练和预测都调用这个方法，保证逻辑完全一致
        """
        if not self.is_trained:
            raise ValueError("向量化模型未训练，请先调用fit()")
        # 统一清洗文本
        cleaned_text = clean_text(text)
        # 向量化
        return self.vectorizer.transform([cleaned_text])