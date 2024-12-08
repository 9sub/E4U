import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


MODEL_PATH='./models/et5_dental_model'

# 장치 설정 (MPS 또는 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 토크나이저 로드
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
tokenizer = T5Tokenizer(
    vocab_file=f"{MODEL_PATH}/spiece.model",
    config=f"{MODEL_PATH}/config.json",
)

# 예측 함수 정의
def generate_answer(input_text: str, max_length: int) -> str:
    model.eval()
    input_ids = tokenizer(
        input_text,
        return_tensors='pt',
    ).input_ids.to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length ,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,  # 상위 50개의 후보를 고려
            top_p=0.8,
            temperature=0.1,
        )

    decoded_output = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return decoded_output