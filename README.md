# Clash of Clans 클랜 데이터 분석 프로젝트 포트폴리오

## 요약 파트

## Clash of Clans 클랜 데이터 기반 생존/리그 예측 모델

### 1. 프로젝트 개요
**프로젝트 일정/인원**
- 2026.01 / 4인 팀 프로젝트
- 팀명: 파이브 가이즈(5팀)

**문제 정의**
- 유령 클랜 비중이 높아 신규 유저가 비활성 클랜에 유입되고 이탈로 이어질 위험이 큼
- 클랜장은 클랜 성장 가능성과 위험도를 직관이 아닌 수치로 판단할 필요가 있음
- 클랜장이 멤버 이탈을 줄이고 성장을 유도하려면 생존 가능성과 리그 성장 잠재력 예측이 필요함

**수행 역할**
- Python 기반 데이터 로드, 전처리, EDA 수행
- 이원화 모델링 설계: 생존 예측(이진 분류) + 리그 예측(다중 분류)
- LightGBM/XGBoost, SMOTE, Optuna 기반 성능 개선
- Streamlit 앱으로 예측 결과와 개선 가이드 시각화


### 2. 결과 및 직무에 적용할 점
- 생존 예측으로 이탈 위험 클랜을 조기 식별하고, 우선 관리 대상을 빠르게 정할 수 있습니다.
- 리그 예측으로 클랜의 현실적 목표 티어와 개선 방향을 제시할 수 있습니다. (`±1 티어 정확도 97.88%`)
- 분석 결과를 Streamlit 서비스로 연결해 클랜장이 바로 실행할 수 있는 액션 도구로 구현했습니다.
- 직무적으로는 데이터 전처리, 불균형 대응, 모델링, 서비스화까지 End-to-End 경험을 확보했습니다.

### 3. 주요 액션(시스템 아키텍처)
![System Architecture](docs/image/System%20Architecture.png)

## 상세 파트

### 1. 데이터 로드
- 원천 데이터: `coc_clans_dataset.csv` (Kaggle 기반 CoC 클랜 데이터)
- 데이터 규모: `3,559,743` rows
- 주요 사용 변수
  - 규모/활동: `num_members`, `war_frequency`, `war_wins`, `war_losses`
  - 경쟁력: `clan_level`, `clan_points`, `clan_war_league`
  - 협동지표: `clan_capital_points`, `clan_capital_hall_level`
  - 멤버품질: `mean_member_level`, `mean_member_trophies`

### 2. 데이터 샘플링
- 유령 클랜 분리 기준(`is_ghost=True`)
  - `num_members < 5`
  - `clan_level >= 2` and `clan_capital_points == 0`
  - `war_total == 0`
- 분리 결과
  - 비활성 클랜: `3,222,737`개 (`90.5%`)
  - 활성 클랜: `337,006`개 (`9.5%`)
- 모델링/핵심 EDA는 활성 클랜 중심으로 진행
- 리그 예측(모델 B)은 클래스 불균형 완화를 위해 SMOTE 적용

### 3. EDA
- 분석 소스
  - `notebooks/01_EDA_Model_A.ipynb`
  - `notebooks/02_Modeling_Model_B.ipynb`
- EDA 범위
  - 분포 확인: 기초 통계, 승률 분포, 전쟁 횟수 분포
  - 관계 분석: 상관분석, 산점도, 박스플롯(가입 조건/리그/성과)
  - 구간 분석: 레벨 구간, 요구 트로피 구간, 멤버 수 구간
  - 이탈 분석: 레벨별 유령 비율(Death Valley), 활성 클랜 위험군 비율

#### 3-1. 데이터 품질 및 기준선 점검
- 유령 클랜 비율이 `90.5%`(3,222,737개)로 매우 높아, 활성 클랜 `337,006`개를 중심으로 분석
- `isFamilyFriendly` 분포 확인
  - 0(일반): `197,299`
  - 1(가족친화): `139,707`
- 활성 클랜 기준 승률 분포
  - 평균 `56.32%`, 중앙값 `53.62%`

#### 3-2. 승률 지표 신뢰도 검증(허수 성과 제거)
- 100% 승률 클랜이 `79,735`개로 과다하게 관측되어 추가 검증 수행
- 100% 승률 중 `1~3판` 전쟁만 치른 클랜이 `12,898`개(`16.2%`)로 확인
- 신뢰도 확보를 위해 `war_total >= 20` 기준 적용
  - 데이터: `337,006 -> 197,861`
  - 재계산 승률: 평균 `67.21%`, 중앙값 `60.61%`
- 해석
  - 전쟁 횟수 필터링 전후로 성과 지표 해석이 달라져, 모델 학습 전 신뢰도 필터가 필요함

#### 3-3. 리그/가입조건/성과 관계 분석
- 리그 분포 상위권이 Gold~Crystal 구간에 집중
  - Gold League II `23,018`, Gold League III `22,892`, Gold League I `20,897`
- 가입 유형별 평균 스펙 비교
  - `inviteOnly`: 평균 클랜 포인트 `23,798.2`, 평균 멤버 트로피 `2,371.6`
  - `open`: 평균 클랜 포인트 `20,696.2`, 평균 멤버 트로피 `2,053.6`
  - `closed`: 평균 클랜 포인트 `22,219.0`, 평균 멤버 트로피 `2,298.0`
- 요구 트로피 구간별 성과
  - 평균 캐피탈 포인트: `0-1k(883) -> 1k-2k(1337) -> 2k-3k(1771) -> 3k-4k(2053)`로 상승
  - `4k+`에서 `1604`로 하락해, 과도한 진입 장벽은 효율 저하 가능성 확인
  - 리그 랭킹 평균도 `3k-4k` 구간이 최고(`11.5`)

#### 3-4. 성장 단계별 이탈 패턴(Death Valley) 분석
- 레벨별 유령 비율 상위 구간
  - Lv2 `96.1%`, Lv1 `95.5%`, Lv3 `89.7%`, Lv4 `84.7%`, Lv5 `82.0%`
- 활성 클랜 내 위험군(멤버 10명 미만) 비율
  - Lv1 `85.5%`, Lv2 `43.5%`, Lv3 `26.6%`, Lv4 `17.6%`, Lv5 `12.5%`
- 해석
  - 초반 레벨에서 이탈이 집중되며, 초반 모집/정착 지원이 핵심 구간임

#### 3-5. 멤버 규모별 생존율(핵심 인사이트)
- 멤버 수 구간별 생존율(`clan_capital_points > 0`)
  - 1~5명: `10.7%`
  - 6~10명: `28.8%`
  - 11~15명: `75.1%`
  - 16~20명: `91.8%`
  - 21+명: `98.4%`
- 최종 해석(클랜장 액션)
  - 소수 정예 전략보다 `11명 이상 활동 인원 확보`가 생존 안정화의 1차 조건
  - 초반에는 무리한 상위 티어 목표보다 인원 유지와 활동성 확보가 우선

### 4. 데이터 전처리
- 제거 컬럼: `clan_name`, `clan_description`, `clan_location`, `clan_badge_url`
- 파생변수 생성
  - `war_total = war_wins + war_ties + war_losses`
  - `win_rate = war_wins / war_total` (0 division 방지)
  - `activity_ratio = mean_member_trophies / (mean_member_level + 1)`
  - `entry_gap = mean_member_trophies - required_trophies`
- 인코딩
  - `isFamilyFriendly` -> 0/1
  - `war_frequency`, `clan_type` -> Label Encoding

### 5. 모델링
#### 5-1. 모델 A: 클랜 생존 예측(이진 분류)
- 타깃: `is_retained = (clan_capital_points > 0)`
- 알고리즘 후보: RandomForest / XGBoost / LightGBM (+ Optuna)
- 핵심 조치: 초기 AUC 1.0000 구간에서 데이터 누수 의심 변수 제거 및 피처 재구성
- 최종 성능(LightGBM, 5개 변수)
  - AUC `0.8911`
  - Accuracy `0.8149`
  - Precision `0.8864`
  - Recall `0.8318`
  - F1 `0.8582`
- 추가 결과
  - 소규모 클랜(5~15명) 155,259개 중 잠재 우수 클랜(생존확률 85% 이상) 17,085개 발굴(`11.0%`)

#### 5-2. 모델 B: 리그 티어 예측(다중 분류)
- 타깃: Bronze/Silver/Gold/Crystal/Master/Champion 6클래스
- 피처 선택: RFE/RFECV(13개 핵심 변수 도출 후 9개로 압축)
- 알고리즘: XGBoost, LightGBM 비교 + SMOTE/Optuna 실험
- 성능
  - 기본 모델 정확도: XGBoost `70.02%`, LightGBM `70.08%`
  - SMOTE 적용 후: 정확도 약 `64%`대로 하락, 소수 클래스 Recall 개선
    - Bronze `33% -> 64%`
    - Champion `32% -> 58%`
  - SMOTE + Optuna(LightGBM): 정확도 `65.23%`, ±1 티어 허용 정확도 `97.88%`

### 6. 스트림릿 시각화
- 서비스 파일: `src/app_unified.py`
- 제공 기능
  - 생존 확률 예측(위험 클랜 조기 탐지)
  - 리그 등급 예측 및 목표 티어 개선 가이드
- 연동 아티팩트 (models/ 디렉토리에 위치)
  - `clan_retention_model.pkl`
  - `league_prediction_model.pkl`
  - `war_frequency_encoder.pkl`
  - `clan_type_encoder.pkl`
  - `league_label_encoder.pkl`
  - `tier_standards.pkl`

### 7. 결과 및 기대효과
- 기대효과
  - 비활성 위험 클랜 조기 식별로 클랜장의 모집/관리 액션 시점 단축
  - 클랜 성장 방향을 정량 지표로 제시해 코칭 자동화 가능
  - 감각 기반 판단에서 데이터 기반 클랜 운영으로 전환
- 한계
  - 텍스트/채팅 행동 데이터 부재
  - 시간축 정보 제한으로 순수 시계열 리텐션 분석 미적용
  - 소수 클래스 성능과 전체 정확도의 트레이드오프 지속
- 개선 방향
  - 시계열 스냅샷 누적 후 생존분석(Survival Analysis) 적용
  - 비용민감학습/커스텀 loss 기반 불균형 대응 고도화
  - SHAP 설명 결과를 알림/추천 룰로 자동 연결

---

### 사용 기술 스택
- Python, Pandas, NumPy
- Scikit-learn, LightGBM, XGBoost, Optuna, imbalanced-learn(SMOTE)
- SHAP
- Streamlit, Joblib

### 프로젝트 파일 및 실행법
- 보고서(PDF)
  - `docs/reports/결과보고서(클래시오브클랜 게임 데이터 분석).pdf`
- 분석 노트북
  - `notebooks/01_EDA_Model_A.ipynb`
  - `notebooks/02_Modeling_Model_B.ipynb`
- 서비스 코드 및 실행
  - `src/app_unified.py`
  - 실행: `uv run streamlit run src/app_unified.py`

> 주의: 위 성능 수치는 노트북 실행 결과 기준이며, 데이터 버전/재학습 시 소폭 변동될 수 있습니다.
