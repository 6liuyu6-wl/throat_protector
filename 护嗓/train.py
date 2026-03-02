# train.py 简化版（适配修正后的代码）
from dataset import load_knowledge_base
from model import VoiceCareAssistant
from config import MODEL_SAVE_PATH

def main():
    print("🚀 开始训练护嗓助手模型")
    # 1. 加载知识库
    kb = load_knowledge_base()
    # 2. 初始化并训练模型
    assistant = VoiceCareAssistant()
    assistant.fit(kb)
    # 3. 保存模型
    assistant.save()
    print("🎉 模型训练完成！")

if __name__ == "__main__":
    main()