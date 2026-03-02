from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# 护嗓知识库路径
KNOWLEDGE_BASE_PATH = ROOT_DIR / "data" / "voice_care_knowledge.csv"
PROCESSED_DATA_DIR=ROOT_DIR / "data" / "processed"
TEST_DATA_PATH = ROOT_DIR/ "data" / "test.csv"

# PKE 关键词提取配置
PKE_MODEL = "topicrank"  # 可选: topicrank, textrank, multipartiterank
KEYPHRASE_N = 5          # 提取的关键词数量
KEYPHRASE_MIN_LEN = 2    # 关键词最小长度

# 问答匹配阈值
MATCH_THRESHOLD = 0.5

# 划分比例（训练集80%，测试集20%）
TRAIN_RATIO = 0.8
# 随机种子（保证划分结果可复现）
RANDOM_SEED = 42
# 分层列（按场景划分，保证训练/测试集场景分布一致）
STRATIFY_COL = "scene"

# 模型保存路径
MODEL_SAVE_PATH = ROOT_DIR / "models" / "voice_care_model.pkl"
PROCESSED_DATA_DIR=ROOT_DIR / "data" / "processed"