# AI-tikitaka

페로브스카이트 실험 데이터로 머신러닝 모델을 만드는 연구 레포입니다. 연구 두 개가 서로 독립된 폴더에 있습니다.

| 폴더 | 연구 | 상태 |
|---|---|---|
| [pce/](pce/README.md) | **문헌 데이터로 태양전지 효율(PCE) 예측과 공정 추천.** 논문 수천 편에서 모은 소자 기록 사용 | 마무리. 모든 방법이 무작위보다는 낫지만, 실험실마다 다른 조건과 기록 안 된 변수 때문에 방법 간 차이를 가릴 수 없었음 |
| [rapid/](rapid/README.md) | **처음 보는 원료(아민)로 결정이 생길지 예측.** 로봇이 같은 절차로 수행한 결정 합성 실험 9,483건 사용 | 진행 중 (1단계 기준선 완료) |

## 설치

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # macOS/Linux: .venv/bin/python
```

명령은 레포 루트에서 실행합니다. 각 연구의 실행 방법은 폴더 안 README에 있습니다.
