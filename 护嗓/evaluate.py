import pandas as pd
import numpy as np
from model import VoiceCareAssistant
from config import TEST_DATA_PATH, MODEL_SAVE_PATH


def load_test_data():
    """加载测试集数据"""
    df = pd.read_csv(TEST_DATA_PATH, encoding="utf-8-sig")
    # 清洗：过滤空值
    df = df.dropna(subset=["question", "advice", "scene"]).reset_index(drop=True)
    print(f"测试集加载完成，共 {len(df)} 条测试数据")
    return df


def evaluate_scene_accuracy(assistant, test_df):
    """评估场景分类准确率"""
    print("\n" + "=" * 50)
    print("开始评估场景分类准确率...")

    correct_count = 0
    total_count = len(test_df)
    wrong_samples = []

    for idx, row in test_df.iterrows():
        question = row["question"]
        true_scene = row["scene"]

        try:
            # 调用模型answer方法，获取预测场景
            result = assistant.answer(question)
            pred_scene = result.get("scene", "未知")

            if pred_scene == true_scene:
                correct_count += 1
            else:
                wrong_samples.append({
                    "question": question,
                    "true_scene": true_scene,
                    "pred_scene": pred_scene
                })
        except Exception as e:
            print(f"第{idx + 1}条数据预测失败: {e}")
            continue

    # 计算准确率
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"场景分类准确率: {accuracy:.2%} ({correct_count}/{total_count})")

    # 输出错误样本（前5个）
    if wrong_samples:
        print("\n错误样本示例（前5个）:")
        for sample in wrong_samples[:5]:
            print(f"  问题: {sample['question']}")
            print(f"  真实场景: {sample['true_scene']} | 预测场景: {sample['pred_scene']}")
            print("-" * 30)

    return accuracy, wrong_samples


def evaluate_topk_accuracy(assistant, test_df, k_list=[1, 3, 5]):
    """评估Top-K问答推荐准确率"""
    print("\n" + "=" * 50)
    print("开始评估Top-K问答推荐准确率...")

    topk_correct = {k: 0 for k in k_list}
    total_count = len(test_df)

    for idx, row in test_df.iterrows():
        question = row["question"]
        true_advice = row["advice"].strip()

        try:
            # 调用模型检索Top-K结果
            topk_results = assistant.retrieve_topk(question, top_k=max(k_list))
            pred_advices = [res["advice"].strip() for res in topk_results]

            # 检查每个K值是否命中
            for k in k_list:
                if k <= len(pred_advices) and true_advice in pred_advices[:k]:
                    topk_correct[k] += 1
        except Exception as e:
            print(f"第{idx + 1}条数据预测失败: {e}")
            continue

    # 输出结果
    print("Top-K问答推荐准确率:")
    for k in k_list:
        acc = topk_correct[k] / total_count if total_count > 0 else 0
        print(f"  Top-{k}: {acc:.2%} ({topk_correct[k]}/{total_count})")

    return topk_correct


def main():
    print("=" * 50)
    print("护嗓助手模型评估")
    print("=" * 50)

    # 1. 加载测试集
    test_df = load_test_data()
    if len(test_df) == 0:
        print("测试集为空，请检查TEST_DATA_PATH配置")
        return

    # 2. 加载训练好的模型
    print(f"\n加载模型: {MODEL_SAVE_PATH}")
    try:
        # 修复后的load方法，支持传入路径，无参数也能正常运行
        assistant = VoiceCareAssistant.load(MODEL_SAVE_PATH)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 3. 执行评估
    scene_acc, wrong_samples = evaluate_scene_accuracy(assistant, test_df)
    topk_acc = evaluate_topk_accuracy(assistant, test_df, k_list=[1, 3, 5])

    # 4. 输出最终评估报告
    print("\n" + "=" * 50)
    print("最终评估报告")
    print("=" * 50)
    print(f"测试集规模: {len(test_df)} 条")
    print(f"场景分类准确率: {scene_acc:.2%}")
    print("Top-K问答推荐准确率:")
    for k, count in topk_acc.items():
        print(f"  Top-{k}: {count / len(test_df):.2%}")
    print("=" * 50)
    print("评估全部完成")


if __name__ == "__main__":
    main()