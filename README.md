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
    diagnose_v2.py   진단 (예측 vs 실제, 구간별·그룹별 오차)
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
  v1_baseline_report.md
  model_v2_report.md
  exp1_report.md
config/
  v2.py              v2 변수 그룹, 상위 범주 개수, 튜닝 후보값, 조합 구간
  exp1.py            실험 1 설정 (숨길 비율, κ, 부트스트랩 수, 화학적 제약)
experiments/
  exp1/              실험 1: 숨긴 고효율 조합 찾기 (exp1_hide_top.py 기본, exp1_more_methods.py 방법 추가, exp1_bootstrap.py·exp1_bootstrap_baselines.py 주 결과)
  exp2/              실험 2 (예정)
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
| `python models/v2/tune_past.py` | 2017년까지 데이터로 v2 하이퍼파라미터 선택 → `best_params_past.json`. 실험 1 전에 실행 |
| `python models/v1_baseline/baseline.py 1` | 데이터 다운로드 + 행 수·컬럼·결측률 리포트 |
| `python models/v1_baseline/baseline.py` | v1 정제 → 학습·평가 → 상위 5개 조건 추천 |
| `python eda/eda.py` | EDA 1~8절 수치 출력 + `eda/figures/` 그래프 생성 |
| `python eda/eda.py scan` | EDA 9절 전체 컬럼 스캔 |
| `python experiments/exp1/exp1_hide_top.py` | 실험 1. 콘솔에는 요약만, 전체 표는 `experiments/exp1/results_hide_top.md` |
| `python experiments/exp1/exp1_bootstrap.py` | 실험 1 주 결과: 논문 단위 클러스터 부트스트랩 (약 15분). 원시 결과 `bootstrap_runs_<시각>.csv`, 요약 `results_bootstrap.md` |
| `python experiments/exp1/exp1_bootstrap_baselines.py 200 bootstrap_runs_20261002_233453.csv` | 같은 재표집에서 Ridge·kNN 기준선과 v2를 짝지어 비교 (약 10분). 요약 `results_bootstrap_baselines.md` |
| `python experiments/exp1/exp1_more_methods.py` | 실험 1 2차: 방법 A1·A2·B·C 추가, 누수 진단 D1·D2, 적중 수 95% 신뢰구간. 표는 `experiments/exp1/results_more_methods.md` |
| `python models/v2/diagnose_v2.py` | v2 진단 (예측 vs 실제, PCE 구간별·그룹별 오차). 표는 `models/v2/diagnostics.md` |
| `python models/v2/calibration.py` | 점수 → 목표 PCE(15/18/20%) 도달 확률 표. 역추천 화면용 조회표 `models/v2/calibration_table.csv` |
| `python models/v2/model_v2.py` | v2 정제 → 튜닝 → 평가 1~7. 콘솔에는 요약만, 전체 표는 `models/v2/results.md`. 교차검증 결과는 `models/v2/cache/`에 저장되어 중단 후 다시 실행하면 이어서 진행 |

`python` 대신 `.venv/Scripts/python`(Windows)이나 `.venv/bin/python`을 쓰거나, 가상환경을 활성화한 뒤 실행하세요.

## 버전별 결과

평가는 논문(DOI) 단위로 나눈 test 20%에서 했습니다.

| 버전 | 데이터 (행 / 논문) | 입력 변수 | 모델 | MAE (%p) | R² | 리포트 |
|---|---|---|---|---|---|---|
| 기준 | 12,823 / 2,591 | 없음 (train 평균) | 평균 | 3.88 | −0.006 | |
| **v1** | 12,823 / 2,591 | 어닐링 온도·시간, 주 용매, 첨가제 유무 | HistGradientBoosting | **3.50** | **0.136** | [v1_baseline_report.md](reports/v1_baseline_report.md) |
| **v2** | 15,945 / 3,183 | 공정 7 + 조건 5 + 보정 2 ([config/v2.py](config/v2.py)) | LightGBM | **3.18** | **0.26** | [model_v2_report.md](reports/model_v2_report.md) |

- v1과 v2는 정제 기준이 달라 test 행도 다릅니다. 같은 test 행(2,346행)에서 비교하면 v1 R² 0.08, v2 R² 0.26입니다.
- v2에서 공정 변수만 쓰면 R² 0.15입니다. 조건 변수를 더하면 0.21, 보정 변수까지 더하면 0.26입니다.

## 실험

| 실험 | 질문 | 주요 결과 | 리포트 |
|---|---|---|---|
| 1 | 2017년까지 데이터에서 상위 10% 조합을 숨기면 v2 순위가 찾아내는가 | 논문 단위 클러스터 부트스트랩(200회): 예측 상위 10% 안 정답 비율이 v2 0.33 (95% 구간 0.14–0.54, 정답 논문 제외 설정)으로 무작위(0.10)보다 높음. B·앙상블 κ=1과 v2의 차이는 구간이 0을 포함해 우열을 판단할 증거 부족, A2는 v2보다 낮은 재표집이 더 많음. 단순 기준선: Ridge 0.31 (0.14–0.46), kNN 0.26 (0.07–0.50, 무작위와 구분 안 됨). v2와의 차이는 모두 우열 판단 증거 부족. 외부 검토 반영(과거 전용 튜닝 등) 후 수치. 방법 선택용이며 최종 성능은 실험 2에서 확인 | [exp1_report.md](reports/exp1_report.md) |
| 2 | 2017년까지 학습 → 2018~2019년 새 조합 찾기 | 예정 | |

EDA 결과와 변수 추천은 [reports/eda_report.md](reports/eda_report.md)에 있습니다.
