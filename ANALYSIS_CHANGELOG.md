# 고객 세분화 분석 변경사항

## 작업 정보

- 작업 브랜치: `fix/apply-storyline-review-k-selection`
- 기준 브랜치: 최신 `main` (`39ab51c`)
- 분석 변경 커밋: `fd32977`
- 변경 목적: `STORYLINE_ANALYSIS_REVIEW.md`의 우선 보완사항 반영 및 최적 군집 수 검증

## 요약

기존 분석 코드 `analysis/segmentation_and_reviews.py`를 수정했다. 이번 커밋에서 새로 추가한 코드 변경은 **30일 재구매 지표 계산과 군집별 집계**이며, 기존 데이터 결합 방식이나 K-Means 입력 변수·전처리·학습 결과는 변경하지 않았다.

최적 `k`를 계산하는 Elbow와 Silhouette 결합 로직은 작업 시작 시점의 최신 `main`에 이미 포함되어 있었다. 이번 작업에서는 해당 로직을 원본 데이터로 다시 실행하고 결과가 재현되는지 검증했다.

## 기존 분석 코드에서 수정한 부분

대상 파일: `analysis/segmentation_and_reviews.py`

### 1. 30일 재구매 타겟 생성

고객별 구매 시각을 날짜 단위로 변환하고 동일 날짜의 주문은 하나의 구매일로 처리한다.

- `first_purchase_date`: 고객의 최초 구매일
- `days_after_first`: 최초 구매일로부터 각 구매일까지의 경과일
- `target_30d`: 최초 구매 후 1~30일 사이에 다른 구매일이 있으면 `True`
- `target_30d_eligible`: 데이터 종료일까지 최소 30일의 관찰 기간이 확보되면 `True`

동일 날짜 주문을 재구매에서 제외한 이유는 분리 결제나 주문 분할을 별도 재구매로 잘못 계산할 가능성을 줄이기 위해서다. 데이터 종료 직전 고객은 30일 동안 재구매할 기회가 없으므로 재구매율 분모에서 제외한다.

### 2. 전체 30일 재구매 지표 추가

데이터 품질 결과에 다음 값을 추가했다.

- 30일 관찰 가능 고객: 89,895명
- 전체 30일 재구매율: 0.66%

### 3. 군집별 30일 재구매 지표 추가

`cluster_profile.csv`와 Markdown/JSON 결과에 다음 열을 추가했다.

- `target_30d_eligible_customers`
- `target_30d_repurchase_customers`
- `target_30d_repurchase_rate`

| 고객군 | 재구매 고객/관찰 고객 | 30일 재구매율 |
|---|---:|---:|
| 고평점 장기 미방문군 | 348/54,161 | 0.64% |
| 저평점 장기 미방문군 | 95/18,509 | 0.51% |
| 최근 구매 고평점군 | 127/14,390 | 0.88% |
| 희소 다중 카테고리군 | 11/615 | 1.79% |

재구매율은 K-Means 입력 변수가 아니라 군집 생성 후 계산한 기술 지표다. 특히 희소 다중 카테고리군은 표본과 재구매 건수가 작으므로 군집 간 재구매율 차이를 인과효과로 해석하면 안 된다.

## 최적 군집 수 검증

`k=2~8`에 대해 Inertia와 Silhouette score를 계산했다.

| k | Inertia | Silhouette | Elbow·Silhouette 결합 점수 |
|---:|---:|---:|---:|
| 2 | 294,673.73 | 0.3650 | 0.0000 |
| 3 | 204,158.20 | 0.3501 | 0.8251 |
| 4 | 149,300.75 | 0.3520 | **0.9605** |
| 5 | 110,085.74 | 0.3321 | 0.9539 |
| 6 | 97,811.70 | 0.2910 | 0.7436 |
| 7 | 88,228.67 | 0.2833 | 0.5230 |
| 8 | 79,497.46 | 0.2765 | 0.0000 |

- 단독 Elbow 기준: `k=5`
- 단독 Silhouette 최고점: `k=2`
- 정규화한 Elbow 강도와 Silhouette 최고점 유지율의 기하평균 최고점: `k=4`
- 최종 선택: `k=4`

단일 지표의 극단적인 선택을 피하기 위해 두 지표를 결합했고, 최소 군집 크기 300명 조건도 적용했다. 최종 `k=4`에서 최소 군집은 668명이다.

## 수정하지 않은 기존 분석 로직

이번 커밋에서는 다음 로직을 변경하지 않았다.

- 주문 단위 결제액 선집계 및 중복 결합 방지
- K-Means 입력 변수: Recency, Monetary, 첫 주문 카테고리 수, 첫 주문 리뷰 점수
- Monetary 상위 1% 값을 거리 계산에서만 상한 처리하는 방식
- 로그 변환과 표준화
- K-Means 설정: `n_init=20`, `random_state=42`
- Elbow·Silhouette 결합 점수 계산 및 `k` 선택 방식
- 리뷰 중복 제거와 포르투갈어 키워드 기반 멀티라벨 분류

따라서 기존 군집 배정, 군집별 고객 수와 평균 프로필은 유지되고 재구매 관련 열만 추가됐다.

## 함께 갱신된 파일

- `STORYLINE_REVISED.md`: 전체·군집별 30일 재구매 결과와 해석 주의사항 추가
- `analysis/README.md`: 분석 실행 단계에 30일 재구매 계산 추가
- `analysis/results/ANALYSIS_RESULTS.md`: 재실행된 최종 보고서
- `analysis/results/analysis_report.json`: 전체 결과 및 감사 정보
- `analysis/results/cluster_profile.csv`: 군집별 재구매 지표 추가
- `analysis/results/cluster_validation.csv`: `k=2~8` 검증 결과 재생성

## 재현 방법

프로젝트 루트에서 다음 명령을 실행한다.

```powershell
python -m pip install -r analysis/requirements.txt
python analysis/segmentation_and_reviews.py --data-dir data --output-dir analysis/results
```

정상 실행 시 콘솔에 `selected_k=4`가 출력되고 `analysis/results`의 CSV, JSON, Markdown 파일이 갱신된다.

## 검증 내용

- Python 문법 검사 통과
- 전체 분석 스크립트 실행 완료
- Elbow 최대 거리 지점 `k=5` 확인
- Silhouette 최고점 `k=2` 확인
- 결합 점수 최고점 및 최종 선택 `k=4` 확인
- 군집 고객 합계 92,718명 확인
- 군집별 30일 재구매 고객 합계 581명 확인

