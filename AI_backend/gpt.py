from openai import OpenAI
from fastapi import HTTPException


client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="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",
)

# GPT API 호출 함수
def call_gpt(prompt: str, max_tokens: int = 100):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))