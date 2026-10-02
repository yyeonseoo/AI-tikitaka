# AI-tikitaka

페로브스카이트 태양전지(MAPbI3, one-step 스핀코팅)의 공정 조건으로 효율(PCE)을 예측하고, 최종적으로 **목표 PCE에 맞는 공정 조건을 역으로 추천**하는 프로젝트입니다.

데이터는 [The Perovskite Database Project](https://www.perovskitedatabase.com/)(Jacobsson et al., *Nature Energy* 2022)를 씁니다.

## 폴더 구조

```
src/                 여러 스크립트가 같이 쓰는 코드
  data.py            NOMAD 다운로드, 로드, MAPbI3 one-step 필터, LLM·실내광 제외, 논문 단위 분할
  features.py        DB 문자열 파싱 (용매 첫 단계, DMSO 분율, 반용매 분류), 마크다운 표 출력
models/
  v1_baseline/       v1: 어닐링 온도·시간·용매·첨가제 유무 → PCE (HistGradientBoosting)
  v2/                v2: 공정+조건+보정 변수 14개 → PCE (LightGBM)
    model_v2.py      정제, 모델 3종 비교, 평가 1~7
    results.md       실행 시 생성되는 전체 결과 표
    best_params.json 선택된 모델·설정
    figures/         변수 중요도, 공정 조합 분포
eda/
  eda.py             변수 선정용 EDA (1~8절), `scan` 옵션은 전체 컬럼 스캔 (9절)
  figures/           EDA 그래프 7개
reports/             결과 리포트
  eda_report.md      EDA와 변수 추천
  v1_baseline_report.md
  model_v2_report.md
config/
  v2.py              v2 변수 그룹, 상위 범주 개수, 튜닝 후보값, 조합 구간
experiments/         실험 1·2 (예정)
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
| `python models/v1_baseline/baseline.py 1` | 데이터 다운로드 + 행 수·컬럼·결측률 리포트 |
| `python models/v1_baseline/baseline.py` | v1 정제 → 학습·평가 → 상위 5개 조건 추천 |
| `python eda/eda.py` | EDA 1~8절 수치 출력 + `eda/figures/` 그래프 생성 |
| `python eda/eda.py scan` | EDA 9절 전체 컬럼 스캔 |
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

EDA 결과와 변수 추천은 [reports/eda_report.md](reports/eda_report.md)에 있습니다.
