# 풋살 경기 자동 분석 - 볼 소유율, 히트맵
## 💭 About
오늘날 대중들이 다양한 스포츠를 즐기는 가운데, 저희가 집중한 스포츠는 "풋살"입니다. 많은 풋살 아마추어 팀들이 다양한 방법으로 경기를 즐기고 있지만 실제로 직접 경기 기록을 남기기는 쉽지 않습니다. 드론 촬영 업체를 이용하여 기록을 남기는 아마추어 팀들이 있는 것을 통해 경기 기록에 대한 수요를 확인할 수 있습니다. 그러나, 경기 영상이 있더라도 피드백을 '수치화'하기는 어렵습니다. 따라서 저희는 영상 처리 기술과 딥러닝 기술을 이용하여 사람과 공을 감지해 공을 소유한 팀을 판별하여 볼 소유율을 제공하는 등 자동 경기 분석을 구현하고자 합니다. 또한 이때 측정한 선수들의 좌표값을 바탕으로 히트맵을 제공합니다.

## ✨ Developer
| <img src="https://avatars.githubusercontent.com/u/97292379?v=4" width="200" height="200"/> | <img src="https://avatars.githubusercontent.com/u/97084864?v=4" width="200" height="200" /> |
| :---: | :---: |
| <div align = "center"><b>재민</b></div> | <div align = "center"><b>서현</b></div> |
| [@HwangJaemin49](https://github.com/HwangJaemin49) | [@seobbang](https://github.com/seobbang) |
| **히트맵** 구현 | **볼 소유율** 구현|

## 💙 핵심기술
### 0. 실행 환경 세팅🛠
#### 1) 아나콘다 설치
#### 2) 가상 환경 설치
```
conda create -n yolov5_deepsort python=[`version`] # version에 해당하는 version 입력
conda activate yolov5_deepsort # 가상환경 activate
```
#### 3) yolov5 + deepsort 깃허브 클론
```
git clone --recurse-submodules <https://github.com/mikel-brostrom/Yolov5_DeepSort_OSNet.git>
```
#### 4) requirement 설치
```
pip install -r requirements.txt
```

### 1. 경기 분석 : 패스 카운트 📊<br>
사용 기술 : Yolov5 + OpenCV<br>
데이터 셋 : 드론으로 찍은 경기 영상<br>
경기 조건 : 특정 색의 조끼를 입은 각 팀<br>
#### Algorithm
1️⃣ 얻어낸 좌표값을 비교해 왼쪽 선수와 오른쪽 선수를 파악<br/>
2️⃣ 적절한 각 선수의 바운더리를 정해, 공의 좌표가 바운더리 안으로 들어갈 경우 해당 선수가 공을 가졌다고 판단<br/>
3️⃣ 계속적인 추적 중 공을 가진 선수가 바뀐 경우 이를 패스로 인식


#### 실행
```
python trackPass.py --source [영상경로] --classes 0 32
```

### 2. 경기 분석 : 히트맵 ⚽<br>
사용 기술 : OpenCV<br>

#### Algorithm
1️⃣ 선수들이 공을 터치한 좌표를 입력받음<br/>
2️⃣ 히트맵 라이브러리로 경기장 사진 위에 좌표를 시각적으로 표시

#### 실행
```
python heatmapfunction.py
```

### 3. 볼 소유율과 히트맵을 통한 피드백 도출 예시 📉
볼 소유율이 높더라도 히트맵이 팀 골대 주변으로 분포하고 있으면 수비 위주로 경기가 진행되었다는 걸 알 수 있고, 따라서 공격 전술에 대한 논의가 필요하다는 결론을 낼 수 있다.
