# 고객 세분화 및 리뷰 분석 실행 방법

```powershell
python analysis/segmentation_and_reviews.py --data-dir data --output-dir analysis/results
```

분석 과정은 다음과 같다.

1. 결제·상품·리뷰 데이터를 주문 단위로 먼저 집계한다.
2. 고객별 Recency, Frequency, Monetary를 계산한다.
3. 첫 주문의 카테고리 수와 리뷰 점수를 결합한다.
4. `k=2~8`의 Silhouette, Davies–Bouldin, Calinski–Harabasz, Inertia를 비교한다.
5. 정규화된 Inertia 곡선과 양 끝점을 잇는 직선 사이의 거리로 엘보우 강도를 계산한다.
6. 엘보우 강도와 Silhouette 최고점 유지율의 기하평균이 가장 높은 해를 선택한다.
7. 선택된 군집별 평균과 중앙값을 산출한다.
8. 리뷰 ID로 중복을 제거하고 포르투갈어 키워드 규칙으로 저평점 불만을 멀티라벨 분류한다.

`analysis/results`의 CSV와 Markdown 결과는 실행으로 재생성할 수 있다. 리뷰 유형은 서로 중복될 수 있으므로 유형별 건수 합계가 전체 리뷰보다 클 수 있다.
