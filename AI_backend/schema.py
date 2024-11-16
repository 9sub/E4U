from pydantic import BaseModel
from typing import List, Dict, Optional

class InputText(BaseModel):
    text: str


class GPTRequest(BaseModel):
    prompt: str


class UserStatus(BaseModel):
    user_id: Optional[int] = None
    image_size: Optional[List[int]] = []
    bounding_box: Optional[List[Dict[str, List[float]]]] = []
    detection_image_path: Optional[str] = None
    segmentation_image_path: Optional[str] = None
    segmentation_data: Optional[List[Dict[str, List[float]]]] = []
    pain_level: Optional[int] = None
    #chating: Optional[Dict[str, str]] = {}