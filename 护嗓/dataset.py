import pandas as pd
from datasets import load_from_disk
from torch.utils.data import DataLoader

import config

def load_knowledge_base():
    """加载护嗓问答知识库"""
    df = pd.read_csv(config.KNOWLEDGE_BASE_PATH, encoding="utf-8-sig")
    # 清洗：过滤核心列空值、去重，保证数据干净
    df = df.dropna(subset=["question", "advice", "scene"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)
    return df



def create_sample_data():
    """生成示例护嗓问答数据（可用于测试）"""
    sample_data = [
        {
            "question": "嗓子干痒怎么办？",
            "answer": "建议多喝温水，避免辛辣刺激食物，可含服润喉糖，减少用嗓时间。",
            "tags": ["干痒", "润喉", "用嗓"],
            "keyphrases": ["嗓子干痒", "润喉糖", "用嗓时间"]
        },
        {
            "question": "唱歌后嗓子疼怎么缓解？",
            "answer": "立即停止用嗓，用温盐水漱口，避免清嗓动作，必要时就医检查声带。",
            "tags": ["唱歌", "嗓子疼", "声带"],
            "keyphrases": ["唱歌", "嗓子疼", "声带"]
        }
    ]
    return sample_data

