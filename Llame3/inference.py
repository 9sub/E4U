import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

model_name = '9sub/llama3_10epoch_no_quantize'

# Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 모델 로드 (MPS 사용)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="mps"
)

# Inference 코드 (MPS 사용)
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=128)

# tqdm 사용하여 반복 질문 받기
print("질문을 입력하세요 (종료하려면 'exit' 입력):")

while True:
    input_text = input("You: ")

    if input_text.lower() == "exit":
        print("대화를 종료합니다.")
        break

    # tqdm 사용하여 진행 상황 표시
    for _ in tqdm(range(1), desc="답변 생성 중"):
        response = pipe(input_text)

    print("Bot:", response[0]["generated_text"])