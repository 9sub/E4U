import tiktoken


def check_max_tokens(prompt: str):
    encoding = tiktoken.get_encoding("cl100k_base")
    # 주어진 모델 이름에 대해 올바른 인코딩을 자동으로 로드
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    max_tokens=len(encoding.encode(prompt))# - 150
    return max_tokens