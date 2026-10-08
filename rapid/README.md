# RAPID: 처음 보는 아민으로 결정 생성 예측

이전 PCE 연구(`experiments/`, `models/`, `reports/`)와 별개인 새 연구입니다.

## 질문

로봇이 만든 할라이드 페로브스카이트 단결정 합성 실험(역온도 결정화)에서, **학습 때 보지 못한 유기 아민**으로 실험하면 큰 결정(결정 점수 4)이 생길지 예측할 수 있는가? 화학 지식을 모델 구조에 넣으면 일반화가 좋아지는가?

## 데이터

- `0045.perovskitedata.csv`: 9,483건, 아민 45종 (Pendleton et al., *J. Phys. Chem. C* 2020; [github.com/ipendlet/MLScripts](https://github.com/ipendlet/MLScripts), CC-BY 4.0)
- 입력: PbI₂·아민·포름산 농도, 반응 온도·시간, 교반 속도, 혼합 시간, 아민 분자 물성 67개
- 출력: 결정 점수 1~4 (4 = 0.1 mm 넘는 큰 결정). 점수 0(미기록) 49건 제외 → 9,434건, 4점 비율 19.6%
- 받기: `git clone --depth 1 https://github.com/ipendlet/MLScripts.git rapid/data/MLScripts` (`rapid/data/`는 git 제외)

## 진행 상황

### 1단계: 기준선 (2026-10-08) — 완료

| 결과 | 무작위 5겹 | 아민 하나 빼기 |
|---|---|---|
| 원 논문 최고 모델 (MCC, 아민별 평균) | 0.56~0.65 | **0.15** (라벨을 섞은 대조군 0.05) |
| 우리 LightGBM, 반응 조건 + 아민 물성 (MCC, 전체) | 0.68 | **0.04** |
| 우리 LightGBM, 같은 입력 (AUC, 아민별 평균) | 0.84 | **0.66** |

- 원 논문의 결론을 재현했습니다. 무작위로 나누면 잘 맞히지만, 처음 보는 아민에서는 MCC가 무작위(0) 근처입니다.
- **새로 본 점:** 처음 보는 아민 안에서의 순위(AUC 0.62~0.68)는 무작위(0.5)보다 낫습니다. 무너지는 것은 주로 "이 아민이 애초에 결정을 잘 만드는가"(아민마다 4점 비율이 크게 다름)입니다. 아민 수준의 판단과 아민 안 조건 순위를 나눠서 다루는 것이 다음 단계의 출발점입니다.
- 표: [results_original_baselines.md](results_original_baselines.md) (원 논문 로그 요약), [results_loo_baseline.md](results_loo_baseline.md) (우리 기준선)

### 2단계: 지식 결합 모델 — 계획 작성 예정

실행 전에 계획서(지표, 비교 대상, 성공 기준)를 먼저 커밋합니다.

## 실행

| 명령 | 내용 |
|---|---|
| `python rapid/original_baselines.py` | 원 논문 모델 로그(ML_Logs.zip) 요약 |
| `python rapid/loo_baseline.py` | 우리 기준선: LightGBM·로지스틱 회귀 × 입력 2종 × 평가 2종 (약 3분) |
