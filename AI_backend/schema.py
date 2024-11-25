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
    result: Optional[Dict[str, List[Dict[str, float]]]] = {}
    chating: Optional[Dict[str, str]] = {}



class Disease(BaseModel):
    disease_name: str

class result_report(BaseModel):
    gum_diseases: Dict[str, List[Disease]]
    symptomArea: List[str]
    symptomText: List[str]
    painLevel: int
    tooth_diseases: Dict[str, List[Disease]] 