# AI-tikitaka

페로브스카이트 태양전지(MAPbI3, one-step 스핀코팅)의 공정 조건으로 효율(PCE)을 예측하고, 최종적으로 **목표 PCE에 맞는 공정 조건을 역으로 추천**하는 프로젝트입니다.

데이터는 [The Perovskite Database Project](https://www.perovskitedatabase.com/)(Jacobsson et al., *Nature Energy* 2022)를 씁니다.

## 폴더 구조

```
splits/
  doi_split.csv      논문(DOI)별 train/test 역할. EDA·v2·실험 1이 모두 이 파일을 씀
  make_split.py      위 파일 생성 (v2 논문은 기존 v2 분할 그대로, 나머지는 DOI 해시)
src/                 여러 스크립트가 같이 쓰는 코드
  data.py            NOMAD 다운로드, 로드, MAPbI3 one-step 필터, LLM·실내광 제외, 논문 단위 분할
  features.py        DB 문자열 파싱 (용매 첫 단계, DMSO 분율, 반용매 분류), 마크다운 표 출력
models/
  v1_baseline/       v1: 어닐링 온도·시간·용매·첨가제 유무 → PCE (HistGradientBoosting)
  v2/                v2: 공정+조건+보정 변수 14개 → PCE (LightGBM)
    model_v2.py      정제, 모델 3종 비교, 평가 1~7
    diagnose.py   진단 (예측 vs 실제, 구간별·그룹별 오차)
    calibration.py   목표 PCE 도달 확률 표 (calibration_table.csv)
    results.md       실행 시 생성되는 전체 결과 표
    best_params.json 선택된 모델·설정 (전체 연도로 고름. 실험 1·2에서는 쓰지 않음)
    tune_past.py     2017년까지 train 논문만으로 다시 튜닝 → best_params_past.json (실험 1·2용)
    figures/         변수 중요도, 공정 조합 분포
eda/
  eda.py             변수 선정용 EDA (1~8절), `scan` 옵션은 전체 컬럼 스캔 (9절)
  figures/           EDA 그래프 7개
reports/             결과 리포트
  eda_report.md      EDA와 변수 추천
  model_v1_report.md
  model_v2_report.md
  exp1_report.md
  exp2_report.md
  exp3_report.md
config/
  v2.py              v2 변수 그룹, 상위 범주 개수, 튜닝 후보값, 조합 구간
  exp1.py            실험 1 설정 (숨길 비율, κ, 부트스트랩 수, 화학적 제약)
experiments/
  PLAN_corrected_reruns.md  외부 검토 2차 반영 수정 재실행 계획 (실행 전 커밋)
  exp1/              실험 1: 2017년까지 안에서 숨긴 고효율 조합 찾기 (철회. exp1_crossfit.py 교차 채점 수정판은 실행 중단)
  exp2/              실험 2: 과거로 학습 → 미래 조합 추천 (PLAN.md 사전 계획과 원래 결과 results_exp2.md, exp2_corrected.py 사후 수정 재실행, smoke/ 시험 실행)
  exp3/              실험 3: 순위 일치도 채점 (1단계 PLAN_phase1.md·exp3_backtest.py: 2017년까지 시간 순서 재비교, 2단계 PLAN_phase2.md·원래 결과 results_phase2.md, 수정 재실행은 exp2/exp2_corrected.py)
data/                원본 데이터 (git 제외, 아래 방법으로 받음)
```

## 설치

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # macOS/Linux: .venv/bin/python
```

## 데이터

- **NOMAD 데이터** (`data/perovskite_db.csv`, 약 120MB): 처음 실행할 때 자동으로 받습니다. 약 7분 걸립니다.
- **원본 2022 CSV** (`data/Perovskite_database_content_all_data.csv`): EDA 6절의 파일 비교에만 필요합니다. 아래 명령으로 받습니다.

  ```bash
  curl -L -o data/Perovskite_database_content_all_data.csv \
    https://media.githubusercontent.com/media/Jesperkemist/perovskitedatabase_data/main/data/Perovskite_database_content_all_data.csv
  ```

## 실행

레포 루트에서 실행합니다. 다른 폴더에서 실행해도 경로는 레포 기준으로 잡힙니다.

| 명령 | 내용 |
|---|---|
| `python splits/make_split.py` | 논문별 train/test 역할 파일 생성 (모델 학습·평가 없음) |
| `python models/v2/tune_past.py` | 2017년까지 데이터로 v2 하이퍼파라미터 선택 → `best_params_past.json` (evaluate_past.py가 사용) |
| `python models/v1_baseline/baseline.py 1` | 데이터 다운로드 + 행 수·컬럼·결측률 리포트 |
| `python models/v1_baseline/baseline.py` | v1 정제 → 학습·평가 → 상위 5개 조건 추천 |
| `python eda/eda.py` | EDA 1~8절 수치 출력 + `eda/figures/` 그래프 생성 |
| `python eda/eda.py scan` | EDA 9절 전체 컬럼 스캔 |
| `python experiments/exp3/exp3_backtest.py` | 실험 3 1단계: 2015·2016·2017년을 그 전 해까지로 학습해 평가, 같은 입력끼리 방법 비교 (약 9분, 2018년 이후 안 씀) |
| `python experiments/exp2/exp2_corrected.py` | 실험 2·3 2단계 사후 수정 재실행: 2017년까지 학습 → 2018~2019년 (약 10분). 결과 `experiments/exp2/results_exp2_corrected.md`, `experiments/exp3/results_phase2_corrected.md` |
| `python experiments/exp1/exp1_crossfit.py [R] [작업 수]` | 실험 1 수정판 (실행 중단, 결과 없음): 교차 채점 + 논문 재표집마다 모든 모델 재학습 (R=100, 병렬 12개로 약 8~9시간). 중단해도 다시 실행하면 이어서 진행 |
| 원래 실험 1·2·3 2단계 스크립트 | 커밋 81577e4에 남아 있음 (`exp1_*.py`, `exp2_future.py`, `exp3_phase2.py`) |
| `python models/v2/diagnose.py` | v2 진단 (예측 vs 실제, PCE 구간별·그룹별 오차). 표는 `models/v2/diagnostics.md` |
| `python models/v2/calibration.py` | 점수 → 목표 PCE(15/18/20%) 도달 확률 표. 역추천 화면용 조회표 `models/v2/calibration_table.csv` |
| `python models/v2/model_v2.py` | v2 정제 → 튜닝 → 평가 1~7. 콘솔에는 요약만, 전체 표는 `models/v2/results.md`. 교차검증 결과는 `models/v2/cache/`에 저장되어 중단 후 다시 실행하면 이어서 진행 |

`python` 대신 `.venv/Scripts/python`(Windows)이나 `.venv/bin/python`을 쓰거나, 가상환경을 활성화한 뒤 실행하세요.

## 버전별 결과

평가는 논문(DOI) 단위로 나눈 test 20%에서 했습니다.

| 버전 | 데이터 (행 / 논문) | 입력 변수 | 모델 | MAE (%p) | R² | 리포트 |
|---|---|---|---|---|---|---|
| 기준 | 12,823 / 2,591 | 없음 (train 평균) | 평균 | 3.88 | −0.006 | |
| **v1** | 12,823 / 2,591 | 어닐링 온도·시간, 주 용매, 첨가제 유무 | HistGradientBoosting | **3.50** | **0.136** | [model_v1_report.md](reports/model_v1_report.md) |
| **v2** | 15,945 / 3,183 | 공정 7 + 조건 5 + 보정 2 ([config/v2.py](config/v2.py)) | LightGBM | **3.18** | **0.26** | [model_v2_report.md](reports/model_v2_report.md) |

- v1과 v2는 정제 기준이 달라 test 행도 다릅니다. 같은 test 행(2,346행)에서 비교하면 v1 R² 0.08, v2 R² 0.26입니다.
- v2에서 공정 변수만 쓰면 R² 0.15입니다. 조건 변수를 더하면 0.21, 보정 변수까지 더하면 0.26입니다.

## 실험

| 실험 | 질문 | 주요 결과 | 리포트 |
|---|---|---|---|
| 1 | 2017년까지 데이터에서 상위 10% 조합을 숨기면 순위가 찾아내는가 | **철회.** 튜닝 단계의 정답 노출과 후보 전체 지표의 부풀림이 확인되어 수치를 결론에 쓰지 않음. 같은 질문은 실험 3 1단계가 수정된 방식으로 답함. 교차 채점 수정판(`exp1_crossfit.py`)은 새로 얻을 정보가 적어 실행 중 중단 | [exp1_report.md](reports/exp1_report.md) |
| 2 | 2017년까지 학습 → 2018~2019년 고효율 조합 찾기 (사전 계획 [PLAN.md](experiments/exp2/PLAN.md)) | **사전 성공 기준 미달**: 주 방법(LightGBM)의 추천 상위 10% 안 정답 비율 0.23 (95% 구간 0.00–0.44, 무작위 0.10). 모든 방법의 평균이 무작위보다 높지만 정답 15개로는 구분 불가. 소자 단위 고효율 판별 AUC 0.68 (0.65–0.71)은 무작위보다 높음. 미래로 갈수록 효율 수준이 올라 R²는 0.09로 하락. **사후 수정 재실행**(튜닝을 학습 데이터 안에서만, 판정은 바꾸지 않음): LightGBM 0.33 (0.00–0.45), 기준 미달 그대로 | [exp2_report.md](reports/exp2_report.md) |
| 3 | 모든 후보를 쓰는 채점법(순위 일치도)으로 평가 | **1단계 (2017년까지, 시간 순서, 같은 입력 비교):** 모든 방법이 무작위보다 높음(순위 일치도 0.36~0.49). 방법 간 짝지은 비교 9개는 모두 우열 판단 증거 부족(비선형 모델·소자 조건 정보·저효율 제외·앙상블·보너스·kNN 대비). **2단계 (2018~2019년, 사후 분석, 실험 2 판정을 대체하지 않음):** LightGBM 0.44 (재표집 구간 0.32–0.52); 소자 구성까지 쓰는 LightGBM이 공정만 보는 kNN보다 높았으나 같은 정보끼리는 우열 판단 증거 부족. **2단계 사후 수정 재실행:** LightGBM(전체) 0.44 (0.29–0.53), kNN 0.29 (0.13–0.38). LightGBM(전체) − kNN은 앞이 높음(정보량 다름), 같은 공정 정보의 LightGBM·선형 회귀 − kNN은 우열 판단 증거 부족 | [exp3_report.md](reports/exp3_report.md) |

### 지금까지의 결론과 다음 단계 후보 (2026-10-05)

- **신호는 있지만 거칠다.** 모든 방법이 무작위보다 꾸준히 낫습니다(조합 순위 일치도 0.3~0.5, 고효율 판별 AUC 0.68~0.78). 하지만 같은 정보를 준 방법끼리는(선형 회귀 vs LightGBM, 모델 vs 비슷한 레시피 평균) 우열을 판단할 증거가 부족합니다.
- **한계는 모델보다 데이터에 있다.** 공정 변수만으로는 효율 차이의 약 15%만 설명됩니다. 기록되지 않은 변수(스핀 속도, 전구체 농도, 막 두께, 습도), 실험실 간 숙련도 차이, 좋은 결과 위주의 보고가 큰 잡음입니다. 소자 구성 정보를 더하면 순위가 좋아지는 것도 실험실 차이를 일부 잡기 때문으로 보입니다.
- **다음 연구질문 후보** (최종 목표 "목표 PCE → 공정 역추천"은 장기 목표로 유지):
  - A. 같은 논문 안에서 공정 하나만 바꿨을 때의 효율 변화 (실험실 차이를 상쇄하는 비교)
  - B. 기록된 정보로 효율을 어디까지 설명할 수 있나 (논문 간 / 논문 안 차이 분해 → 예측 상한)
  - C. 최고 조합 고르기 대신 나쁜 레시피 거르기 (고효율 확률 기반 선별)
  - D. 공정 선택의 연도별 변화와 효율 향상의 관계 (서술적 분석)
- **RAPID 단결정 데이터**(`data/RAPID_*.xlsx`)는 PCE가 없고 공정이 달라(용액 속 단결정 합성) 본 연구에는 쓰지 않고, 나중에 별도로 분석합니다.

EDA 결과와 변수 추천은 [reports/eda_report.md](reports/eda_report.md)에 있습니다.
