## 핵심기술
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
