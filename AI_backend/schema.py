from pydantic import BaseModel

class InputText(BaseModel):
    text: str


class GPTRequest(BaseModel):
    prompt: str