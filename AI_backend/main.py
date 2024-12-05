from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import FileResponse
from io import BytesIO
import torch
from torchvision.models.mobilenetv2 import MobileNetV2
import torch
from ultralytics import YOLO
import uuid
import os
import shutil
import time
from fastapi.middleware.cors import CORSMiddleware
import re
import json

from predict.is_mouth_predict import preprocess_image, infer, load_model
from predict.et5_predict import generate_answer
from schema import InputText, GPTRequest, UserStatus, result_report
from gpt import call_gpt
from utils.calculate_max_tokens import check_max_tokens
from utils.read_bounding_box import find_bounding_box
from utils.read_segmentation import read_segmentation_file
from utils.check_image_size import check_image_size
from utils.check_disease_teeth_num import match_diseases_to_teeth_and_gums
from utils.remove_dup import remove_dup
from utils.return_json_format import return_json_format
from danger_point import calculate_danger_score
from utils.analysis_results_form import analysis_results_form
from utils.result_report_form import result_report_form
from utils.return_weight_from_et5 import extract_disease_names,adjust_and_weight_conf
from utils.visualization import visualization

app = FastAPI()

user_status = UserStatus()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Model server API"}



@app.post("/is_mouth/")
async def predict(file: UploadFile = File(...)):
    try:
        # 이미지 읽기 및 전처리
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes)
        

        # MPS 또는 CPU 선택
        device = torch.device("cuda" if torch.cuda.is_available else "cpu")

        # 모델 로드
        model_path = './models/mobilenetv2_epoch20_lr1e-5_batch16.pth'
        model = load_model(model_path, device)
        # 추론 수행
        predicted_class = infer(model, image_tensor, device)

        is_mouth_image = (predicted_class == 1)
        print(is_mouth_image)
        return {"is_mouth_image": is_mouth_image}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/")
def detect_image(file: UploadFile = File(...), additionalData: str = Form(...)):
    try:
        additional_data = json.loads(additionalData)
    except json.JSONDecodeError as e:
        return {"error": "Invalid JSON format in additionalData", "details": str(e)}
    print(additionalData)
    temp_file = f"./result/before_inference/{uuid.uuid4()}.jpg"
    with open(temp_file, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    #print(temp_file)
    user_status.image_size = check_image_size(temp_file)
    #print(user_status.image_size)
    # detection 모델 로드 및 추론
    detection_model = YOLO('./models/detection_final.pt')
    detection_results = detection_model.predict(temp_file, conf=0.1, save=True, project="result", name="detection", exist_ok=True)
    #print(detection_results)

    # 결과 파일 경로 설정 및 확인
    detection_output_image_path = f"result/detection/{os.path.basename(temp_file)}"
    before_detection_image_path = f"result/before_inference/{os.path.basename(temp_file)}"

    #detection 결과 이미지 path 저장, bounding box 저장
    user_status.detection_image_path = detection_output_image_path
    user_status.bounding_box=find_bounding_box(before_detection_image_path, detection_results)
    #print(user_status.bounding_box)

    #segmentation 모델 로드 및 추론
    segmentation_model = YOLO('./models/segmentation_last.pt')
    segmentation_model.predict(temp_file, save=True, project="result", name="segmentation", save_txt=True, exist_ok=True)

    # 결과 파일 경로 설정 및 확인
    segmentation_output_image_path = f"result/segmentation/{os.path.basename(temp_file)}"

    #segmentation 결과 이미지 path 저장
    user_status.segmentation_image_path = segmentation_output_image_path

    txt_filename = os.path.splitext(os.path.basename(temp_file))[0] + ".txt"
    txt_file_path = os.path.join("result/segmentation/labels", txt_filename)
    
    user_status.segmentation_image_path = segmentation_output_image_path
    user_status.segmentation_data = read_segmentation_file(txt_file_path)

    #print(user_status.segmentation_data, user_status.bounding_box)
    #print(result)
    result = match_diseases_to_teeth_and_gums(user_status.bounding_box, user_status.segmentation_data, user_status.image_size[0], user_status.image_size[1])
    # print('************************')
    # print(result)
    #환자 텍스트 증상
    symptomlist=[
        "치아가 시린 통증은 어떤 질환인가요?",
        "잇몸이 쑤셔요"
    ]

    #symptom = "치아가 시린 통증은 어떤 질환인가요?"
    symptom = additional_data["symptomText"]

    if(symptom != ""):
        et5_output = generate_answer(symptom,300)
    else:
        et5_output = ""
    weight_disease=extract_disease_names(et5_output)

    # target_locations = {'tooth_disease': [12, 21], 'gum_diseases': ['하악_좌측중간', '하악_우측중간']}
    tooth_disease = []
    gum_disease = []
    etc = []

    # Classification logic
    for area in additional_data.get("symptomArea", []):
        if area.isdigit():  # 숫자형 값
            tooth_disease.append(int(area))
        elif "상악" in area or "하악" in area:  # "상악" 또는 "하악"
            gum_disease.append(area)
        elif area in ["혀", "입천장", "입술"]:  # "혀", "입천장", "입술"
            etc.append(area)

    # Output dictionary
    target_locations = {
        "tooth_disease": tooth_disease,
        "gum_diseases": gum_disease,
        "etc": etc
    }
    pain_level = additional_data.get("painLevel",0)

    print("before:",result)
    print('************************')
    result = adjust_and_weight_conf(result, target_locations, weight_disease,pain_level)
    print("after:",result)



    '''오류나면 여기 풀기'''
    #result = remove_dup(result)

    user_status.result = result
    #시각화
    visualization(before_detection_image_path,result)

    #print(user_status.result)
    #print(user_status.result)
    # 출력 로직
    # print("===== 치아 질환 =====")
    # for tooth_num, diseases in result['tooth_diseases'].items():
    #     print(f"치아 {tooth_num}번의 질환:")
    #     for disease in diseases:
    #         print(f"  - 질환 ID: {disease['disease_id']}")
    #         print(f"    질환 이름: {disease['disease_name']}")
    #         print(f"    신뢰도: {disease['confidence']:.2f}")
    #         print(f"    위치: {disease['location']}")

    # print("\n===== 잇몸 질환 =====")
    # for region, diseases in result['gum_diseases'].items():
    #     print(f"잇몸 부위: {region}")
    #     for disease in diseases:
    #         print(f"  - 질환 ID: {disease['disease_id']}")
    #         print(f"    질환 이름: {disease['disease_name']}")
    #         print(f"    신뢰도: {disease['confidence']:.2f}")
    #         print(f"    위치: {disease['location']}")

    # 결과 파일이 생성될 때까지 최대 30초 대기
    timeout = 30
    while not os.path.exists(detection_output_image_path) and os.path.exists(segmentation_output_image_path) and timeout > 0:
        time.sleep(1)
        timeout -= 1
    if not os.path.exists(detection_output_image_path) and os.path.exists(segmentation_output_image_path):
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(detection_output_image_path, media_type="image/jpg")


@app.post("/detection_result/")
def get_detection_result():
    json_format = return_json_format(user_status.result)
    print(json_format)
    return json_format


@app.post('/danger_point/')
async def get_danger_point(pain_level: int):
    #print(user_status.result_report_form)
    user_status.pain_level = pain_level
    danger_point=calculate_danger_score(user_status.result_report_form.dict(), user_status.pain_level)
    #print(danger_point)
    return {"danger_point": danger_point}
    


@app.post("/et5_predict/")
async def predict(input_data: InputText):
    input_text = input_data.text
    result = generate_answer(input_text, 300)
    return {"output": result}


#"\n 환자의 증상과 질환을 통해 원인과 증상만 자세하게 분석하고 치아위치는 제외해. 하나의 텍스트로 작성해. 말투는 정중하게 사용자에게 말하는 것처럼, 마지막 인사는 제외해."


#result 포맷 정해서 넣기

@app.post('/result_report/')
async def result_report(data : result_report):
    user_status.result_report_form = data
    tooth_disease, gum_disease,etc_disease,symptom_area_str, tooth_disease_name, gum_disease_name,etc_diseases_names_str = result_report_form(data)

    # 치아번호와 질환을 추출
    tooth_conditions = re.findall(r'치아번호 (\d+) : ([^,>]+)', tooth_disease)

    # 기타 부위와 질환 추출
    gum_conditions = re.findall(r'<\s*(.*?)\s*:\s*(.*?)\s*>', gum_disease)

    etc_conditions = re.findall(r"<\s*(.*?)\s*:\s*(.*?)\s*>", etc_disease)
    result = []
    for region, diseases in etc_conditions:
        # 질환을 쉼표로 분리하고 중복 제거 후 정렬
        unique_diseases = ", ".join(sorted(set(d.strip() for d in diseases.split(","))))
        # 결과를 "부위 : 질환" 형식으로 저장
        result.append(f"{region} : {unique_diseases}")

    # 최종 결과 문자열로 변환
    etc_conditions = " , ".join(result)
    #print(tooth_conditions)
    #print(etc_conditions)


    # 질환별로 치아번호를 그룹화
    condition_map = {}
    for tooth, condition in tooth_conditions:
        if condition not in condition_map:
            condition_map[condition] = []
        condition_map[condition].append(tooth)

    # 출력 생성
    # 잇몸 영역 고치기
    output = "예측된 구강질환 위치는 다음과 같습니다.\n"

    if tooth_disease_name=="질환없음" and gum_disease_name=="질환없음" and etc_diseases_names_str=="질환없음":
        output += "치아 질환: 질환없음\n"
        output += "잇몸 질환: 질환없음\n"
        output += "예측된 결과에서 환자분께서 가지고 있는 구강질환이 없습니다."
        if symptom_area_str == "[]":
            detailed_result = "환자분께서 가지고 있는 구강질환이 없습니다."
        else:
            if ("상악" in symptom_area_str or "하악" in symptom_area_str):
            # 상악/하악으로 구성된 경우
                detailed_result = f"예측된 결과에서 나타나는 질환은 없지만,\n{symptom_area_str.replace('[', '').replace(']', '')} 부분의 잇몸 통증이 있는것으로 보입니다. 치과에 방문에 자세한 진료를 받으시는 것을 추천드립니다"
            elif "혀" in symptom_area_str or "입술" in symptom_area_str or "입천장" in symptom_area_str:
                detailed_result = f"예측된 결과에서 나타나는 질환은 없지만,\n{symptom_area_str.replace('[', '').replace(']', '')} 부분의 통증이 있는것으로 보입니다. 치과에 방문에 자세한 진료를 받으시는 것을 추천드립니다."
            else:
            # 숫자로 구성된 경우
                detailed_result = f"예측된 결과에서 나타나는 질환은 없지만,\n{symptom_area_str.replace('[', '').replace(']', '')} 번 치아의 통증이 있는것으로 보입니다. 치과에 방문에 자세한 진료를 받으시는 것을 추천드립니다."
        
        return {"result": output, "detailed_result": detailed_result, "care_method": "환자분께서 가지고 있는 구강질환이 없습니다. 지금처럼 구강 건강을 유지하기 위한 관리 방법을 알려드립니다.\n 1. 올바른 양치 습관\n 하루 2~3회, 식후 30분 이내에 양치질을 합니다.\n 2분 이상 꼼꼼히 닦으며, 칫솔은 치아와 잇몸 경계 부위를 포함해 닦아야 합니다.\n 치실이나 치간 칫솔을 사용해 칫솔이 닿지 않는 부위의 플라크를 제거합니다.\n 2. 정기적인 구강검진\n 6개월에 한 번 치과를 방문해 정기 검진과 스케일링을 받습니다.\n조기 발견 및 예방을 위해 치아와 잇몸 상태를 정기적으로 점검합니다."}
    
    for condition, teeth in condition_map.items():
        output += f"치아번호 {', '.join(teeth)} {condition}\n"

    for location, condition in gum_conditions:
        output += f"{location} {condition}\n"

    output += f"{etc_conditions}\n"

    if symptom_area_str == "[]":
        output += f"\n환자분께서 느끼시는 통증은 없지만, 위와같은 질환으로 인해 발생한 치아질환을 관리하시는 것이 좋습니다."
    else:
        if ("상악" in symptom_area_str or "하악" in symptom_area_str):
            # 상악/하악으로 구성된 경우
            output += f"위와 같은 질환으로 인해 환자분께서 느끼시는 {symptom_area_str.replace('[', '').replace(']', '')} 부분의 잇몸 통증이 발생한 것으로 보입니다."
        elif "혀" in symptom_area_str or "입술" in symptom_area_str or "입천장" in symptom_area_str:
            output += f"위와 같은 질환으로 인해 환자분께서 느끼시는 {symptom_area_str.replace('[', '').replace(']', '')} 부분의 통증이 발생한 것으로 보입니다."
        else:
            # 숫자로 구성된 경우
            output += f"위와 같은 질환으로 인해 환자분께서 느끼시는 {symptom_area_str.replace('[', '').replace(']', '')} 번 치아의 통증이 발생한 것으로 보입니다."

        #output += f"\n위와같은 질환으로 인해 환자분께서 느끼시는 {generate_message(symptom_area_str)} 번 치아의 통증이 발생한 것으로 보입니다."

    #{symptom_area_str.replace('[', '').replace(']', '')}

    # 결과 출력
    #print(output)


    #result = call_gpt(prompt=input_text_result+"구강질환 보고서를 뽑아줘")
    #print(result)
    #detailed_result = call_gpt(prompt=input_text_detailed_result)

    disease_name = tooth_disease_name + gum_disease_name
    input_text_detailed_result_disease_name = disease_name + "증상"
    #input_text_care_method1 = disease_name + "치료"
    input_text_care_method2 = disease_name + "예방"


    #result = generate_answer(input_text_detailed_result,300)
    detailed_result = generate_answer(input_text_detailed_result_disease_name,300)
    #care_method = call_gpt(prompt=input_text_care_method)
    care_method = generate_answer(input_text_care_method2,300)
    
    return {"result": output, "detailed_result": detailed_result, "care_method" : care_method}

@app.post('/chat/')
async def chat(input_data: InputText):
    input_text = input_data.text
    result = call_gpt(prompt=input_text)
    
    return {"output": result}

'''
{
  "text": "(tooth_num 11 : 충치 tooth_num 31 : 충치 , 위와 아래 치아가 너무 시린 증상이 있어) 괄호 안에는 각 치아별로 질환이 적혀져 있고 쉼표 뒤에는 환자의 증상이 있어 너가 구강질환에 대한 진단을 해줘"
}
'''
