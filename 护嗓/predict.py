from model import VoiceCareAssistant


def main():
    # 加载模型
    assistant = VoiceCareAssistant.load()

    print("AI 护嗓助手已启动，输入问题开始对话（输入 'quit' 退出）")
    while True:
        user_input = input("你: ")
        if user_input.lower() == "quit":
            break
        response = assistant.answer(user_input)
        print(f"助手: {response['answer']}")
        print(f"置信度: {response['confidence']:.2f}")

# 我是一名教师，经常上很多节课，喉咙感到干痒，能给我提供些建议吗

if __name__ == "__main__":
    main()