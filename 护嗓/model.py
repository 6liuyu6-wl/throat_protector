import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tokenizer import VoiceCareTokenizer, clean_text
from dataset import load_knowledge_base
from config import MODEL_SAVE_PATH

class VoiceCareAssistant:
    def __init__(self):
        self.tokenizer = VoiceCareTokenizer()
        self.knowledge_base = None  # 知识库DataFrame
        self.question_vectors = None  # 所有问题的向量化结果
        # 置信度阈值优化：降低到0.15，适配小数据集，避免误过滤
        self.CONFIDENCE_THRESHOLD = 0.15

    def fit(self, knowledge_base):
        """
        训练模型：加载知识库，训练向量化模型，预计算所有问题的向量
        """
        # 1. 保存知识库
        self.knowledge_base = knowledge_base
        if len(self.knowledge_base) == 0:
            raise ValueError("知识库不能为空")
        print(f"✅ 知识库加载完成，共 {len(self.knowledge_base)} 条问答")

        # 2. 训练向量化模型
        all_questions = self.knowledge_base["question"].tolist()
        self.tokenizer.fit(all_questions)

        # 3. 预计算所有问题的向量，避免预测时重复计算
        self.question_vectors = self.tokenizer.vectorizer.transform(
            [clean_text(q) for q in all_questions]
        )
        print("✅ 知识库问题向量预计算完成")

    def retrieve_topk(self, question, top_k=3):
        """
        检索Top-K最匹配的问答对
        """
        # 1. 对用户输入向量化
        user_vector = self.tokenizer.vectorize(question)

        # 2. 计算余弦相似度（最适合文本匹配的算法，替代你之前的错误算法）
        similarity_scores = cosine_similarity(user_vector, self.question_vectors)[0]

        # 3. 取Top-K最高相似度的结果
        topk_indices = np.argsort(similarity_scores)[::-1][:top_k]

        # 4. 组装结果
        topk_results = []
        for idx in topk_indices:
            row = self.knowledge_base.iloc[idx]
            topk_results.append({
                "question": row["question"],
                "advice": row["advice"],
                "scene": row["scene"],
                "similarity": float(similarity_scores[idx])
            })
        return topk_results

    def answer(self, question):
        """
        核心问答方法：返回最终回答+置信度
        """
        # 1. 检索Top-1最匹配的结果
        top_results = self.retrieve_topk(question, top_k=1)
        best_result = top_results[0]
        confidence = best_result["similarity"]

        # 2. 打印匹配日志（方便调试）
        print(f"\n匹配问题: {best_result['question']}")
        print(f"文本相似度: {confidence:.4f}")

        # 3. 置信度判断，低于阈值返回兜底回答
        if confidence < self.CONFIDENCE_THRESHOLD:
            return {
                "answer": "抱歉，我暂时无法回答这个问题。建议您咨询专业耳鼻喉科医生，同时注意科学用嗓、多喝温水、避免辛辣刺激食物。",
                "confidence": confidence,
                "scene": "未知"
            }

        # 4. 返回匹配到的回答
        return {
            "answer": best_result["advice"],
            "confidence": confidence,
            "scene": best_result["scene"]
        }

    def save(self):
        """保存模型到本地"""
        with open(MODEL_SAVE_PATH, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ 模型已保存到: {MODEL_SAVE_PATH}")

    @classmethod
    def load(cls, model_path=None):
        """
        从本地加载模型
        :param model_path: 模型文件路径，不传则使用config里的默认路径
        """
        from config import MODEL_SAVE_PATH
        load_path = model_path if model_path else MODEL_SAVE_PATH
        with open(load_path, "rb") as f:
            model = pickle.load(f)
        print(f"✅ 模型加载成功，路径：{load_path}")
        return model