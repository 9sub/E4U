from pydantic import BaseModel
from typing import List, Dict, Optional

class InputText(BaseModel):
    text: str


class GPTRequest(BaseModel):
    prompt: str


class UserStatus(BaseModel):
    user_id: Optional[int] = None
    bounding_box: Optional[List[float]] = []
    image_path: Optional[str] = None
    pain_level: Optional[int] = None
    #chating: Optional[Dict[str, str]] = {}