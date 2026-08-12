# RFM 이후 추가 코드 쉬운 설명서

## 이 문서는 무엇을 설명하나요?

이 문서는 `Olist_Analysis.ipynb`의 **RFM 분석 이후에 추가된 고객 세분화 코드**를 비전공자도 이해할 수 있도록 순서대로 설명합니다.

전체 흐름을 한 문장으로 요약하면 다음과 같습니다.

> 주문 데이터를 정확하게 정리한 뒤 고객마다 구매 특성을 계산하고, 비슷한 고객끼리 다섯 그룹으로 나누어 각 그룹의 특징과 재구매 현황을 살펴봅니다.

코드는 크게 네 단계로 나뉩니다.

1. 주문마다 한 줄만 남도록 데이터 정리
2. 고객별 RFM과 첫 구매 특성 계산
3. 고객을 몇 그룹으로 나눌지 결정
4. 최종 군집을 만들고 의미 해석

---

## 1. 먼저 알아두면 좋은 용어

### RFM이란?

RFM은 고객의 구매 행동을 세 가지 숫자로 요약하는 방법입니다.

| 항목 | 뜻 | 이 분석에서의 계산 방법 | 일반적인 해석 |
|---|---|---|---|
| Recency | 얼마나 최근에 구매했는가 | 분석 기준일 − 마지막 구매일 | 작을수록 최근 고객 |
| Frequency | 몇 번 구매했는가 | 서로 다른 주문 ID의 수 | 클수록 반복 구매 고객 |
| Monetary | 얼마를 결제했는가 | 고객의 주문별 결제액 합계 | 클수록 고액 구매 고객 |

예를 들어 A 고객이 10일 전에 마지막으로 구매했고 총 3회 주문하여 500헤알을 결제했다면 다음과 같이 표현할 수 있습니다.

```text
Recency = 10일
Frequency = 3회
Monetary = 500헤알
```

### K-Means란?

K-Means는 비슷한 특징을 가진 데이터를 같은 그룹으로 묶는 방법입니다. 이 분석에서는 고객을 다음 네 숫자로 표현합니다.

```text
[최근성, 총 결제액, 첫 구매 카테고리 수, 첫 리뷰 점수]
```

K-Means는 이 숫자들이 서로 비슷한 고객을 찾아 같은 군집(Cluster)에 배치합니다. `K`는 만들 그룹의 개수이며, `k=5`는 고객을 다섯 그룹으로 나눈다는 뜻입니다.

---

## 2. 분석 도구 불러오기

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
```

| 코드 | 역할 |
|---|---|
| `numpy` | 로그 변환, 난수 추출 등 숫자 계산 |
| `pandas` | 표 형태의 데이터 정리와 집계 |
| `matplotlib` | Elbow 그래프 출력 |
| `KMeans` | 고객 군집 생성 |
| `StandardScaler` | 서로 단위가 다른 변수를 같은 기준으로 조정 |

`RANDOM_STATE = 42`는 실행할 때마다 같은 표본과 같은 군집 결과가 나오도록 난수의 시작점을 고정하는 설정입니다. 숫자 42 자체에 특별한 의미는 없습니다.

---

## 3. 주문마다 한 줄이 되도록 데이터 정리

### 3.1 정상적인 구매 주문만 선택

```python
valid_status = ['approved', 'processing', 'invoiced', 'shipped', 'delivered']
orders_clean = orders_df[orders_df['order_status'].isin(valid_status)].copy()
```

주문 데이터에는 취소되거나 구매가 성립하지 않은 주문도 있습니다. 고객의 실제 구매 행동을 분석하기 위해 결제 승인, 처리 중, 송장 발행, 배송 중, 배송 완료 단계에 들어간 주문만 선택합니다.

### 3.2 날짜 형식 변환

```python
orders_clean['order_purchase_timestamp'] = pd.to_datetime(
    orders_clean['order_purchase_timestamp']
)
```

CSV에서 읽은 날짜는 처음에는 단순한 글자일 수 있습니다. 날짜 차이를 계산하려면 실제 날짜·시간 형식으로 바꿔야 합니다.

### 3.3 결제 정보를 주문 단위로 합산

```python
order_payment = order_payments_df.groupby('order_id', as_index=False).agg(
    order_payment=('payment_value', 'sum')
)
```

한 주문을 카드와 쿠폰으로 나누어 결제하는 등 결제 행이 여러 개일 수 있습니다. 따라서 `order_id`별로 결제액을 먼저 합쳐 주문 하나의 실제 결제액을 계산합니다.

이 작업은 매우 중요합니다. 상품 테이블과 결제 테이블을 곧바로 합치면 한 주문에 상품이 여러 개 있을 때 같은 결제액이 상품 수만큼 반복될 수 있기 때문입니다.

```text
실제 결제액 100헤알, 상품 3개

상품 A — 결제액 100
상품 B — 결제액 100
상품 C — 결제액 100
잘못 계산한 합계 = 300헤알
```

주문 단위로 결제액을 먼저 합치면 실제 합계인 100헤알을 유지할 수 있습니다.

실행 후 `order_payment`는 **주문 한 건당 한 행**인 다음 형태가 됩니다. 아래 값은 구조를 설명하기 위한 축약 예시입니다.

| order_id | order_payment |
|---|---:|
| order_A | 100.00 |
| order_B | 75.50 |

원본 결제 테이블에 `order_A`가 카드 70헤알과 쿠폰 30헤알의 두 행으로 존재해도 결과에는 합계 100헤알인 한 행만 남습니다.

### 3.4 주문별 카테고리 수 계산

```python
item_category = order_items_df.merge(
    products_df[['product_id', 'product_category_name']],
    on='product_id', how='left'
)
order_item_features = item_category.groupby('order_id', as_index=False).agg(
    category_count=('product_category_name', 'nunique')
)
```

상품 ID에 카테고리명을 연결한 뒤 주문마다 서로 다른 카테고리가 몇 개 들어 있는지 셉니다. `nunique`는 같은 카테고리가 여러 번 나와도 한 번만 셉니다.

```text
가구, 가구, 생활용품 → 카테고리 수 2개
```

상품 정보를 연결한 직후의 `item_category`는 상품 단위라 같은 주문이 여러 행일 수 있습니다.

| order_id | product_id | product_category_name |
|---|---|---|
| order_A | product_1 | 가구 |
| order_A | product_2 | 가구 |
| order_A | product_3 | 생활용품 |

이를 주문별로 집계한 `order_item_features`는 다음처럼 바뀝니다.

| order_id | category_count |
|---|---:|
| order_A | 2 |

즉 세 상품을 샀더라도 서로 다른 카테고리가 두 종류이면 `category_count=2`입니다.

### 3.5 리뷰 중복 제거 및 주문별 평점 계산

```python
order_review = order_reviews_df.drop_duplicates('review_id') \
    .groupby('order_id', as_index=False).agg(
        review_score=('review_score', 'mean')
    )
```

같은 리뷰가 상품이나 결제 행 때문에 반복되지 않도록 `review_id` 기준으로 중복을 제거합니다. 주문에 리뷰가 여러 개라면 평균 점수를 사용합니다.

집계된 `order_review`의 모습은 다음과 같습니다.

| order_id | review_score |
|---|---:|
| order_A | 4.0 |
| order_B | 2.0 |

이 표 역시 주문 한 건당 최대 한 행입니다. 리뷰가 없는 주문은 이후 병합 결과에서 `review_score`가 빈 값으로 나타납니다.

### 3.6 정리한 표 연결

```python
order_level = (
    orders_clean.merge(
        customer_df[['customer_id', 'customer_unique_id']],
        on='customer_id',
        validate='many_to_one'
    )
    .merge(
        order_payment,
        on='order_id',
        how='left',
        validate='one_to_one'
    )
    .merge(
        order_item_features,
        on='order_id',
        how='left',
        validate='one_to_one'
    )
    .merge(
        order_review,
        on='order_id',
        how='left',
        validate='one_to_one'
    )
)
```

주문, 고객, 결제, 카테고리, 리뷰 정보를 연결합니다. `validate`는 연결 방식이 예상과 같은지 검사하는 안전장치입니다.

| 설정 | 의미 |
|---|---|
| `many_to_one` | 여러 주문이 한 고객에게 연결될 수 있음 |
| `one_to_one` | 주문 하나는 집계 결과 한 행에만 연결돼야 함 |

모든 정보를 합친 `order_level`의 핵심 컬럼은 다음과 같습니다.

| order_id | customer_unique_id | order_purchase_timestamp | order_payment | category_count | review_score |
|---|---|---|---:|---:|---:|
| order_A | customer_1 | 2018-01-10 10:00 | 100.00 | 2 | 4.0 |
| order_B | customer_1 | 2018-02-03 14:30 | 75.50 | 1 | 2.0 |
| order_C | customer_2 | 2018-02-05 09:10 | 42.00 | 1 | 5.0 |

`customer_1`이 두 번 등장하는 것은 정상입니다. 이 표의 행 기준은 고객이 아니라 **주문**이기 때문입니다. 중요한 점은 같은 `order_id`가 두 번 등장하지 않는다는 것입니다.

### 3.7 주문 ID 중복 검사

```python
assert order_level['order_id'].is_unique, \
    'order_level은 주문당 한 행이어야 합니다.'
```

`assert`는 조건이 틀리면 분석을 즉시 중단합니다. 같은 주문 ID가 두 번 이상 나오지 않는지 확인하여 결제액이 부풀려지는 문제를 막습니다.

---

## 4. 고객별 RFM 계산

### 4.1 분석 기준일 설정

```python
snapshot_date = order_level['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
```

Recency를 계산하려면 “오늘”에 해당하는 기준일이 필요합니다. 과거 데이터이므로 실제 오늘이 아니라 데이터의 마지막 주문일 다음 날을 사용합니다.

### 4.2 고객별 집계

```python
rfm = order_level.groupby('customer_unique_id', as_index=False).agg(
    Recency=('order_purchase_timestamp', lambda x: (snapshot_date - x.max()).days),
    Frequency=('order_id', 'nunique'),
    Monetary=('order_payment', 'sum')
)
```

- `Recency`: 기준일에서 마지막 구매일을 뺀 일수
- `Frequency`: 서로 다른 주문 수
- `Monetary`: 주문별 결제액의 합계

Olist 고객은 대부분 한 번만 구매하여 Frequency가 거의 1에 몰려 있습니다. 그래서 최종 K-Means 입력에서는 Frequency를 제외하지만 군집 설명용 참고 지표로는 유지합니다.

`order_level`을 고객별로 집계한 `rfm`은 **고객 한 명당 한 행**으로 바뀝니다.

| customer_unique_id | Recency | Frequency | Monetary |
|---|---:|---:|---:|
| customer_1 | 25 | 2 | 175.50 |
| customer_2 | 23 | 1 | 42.00 |

앞의 예시에서 두 주문을 가진 `customer_1`은 Frequency가 2이고 Monetary는 100.00 + 75.50 = 175.50입니다.

---

## 5. 첫 주문 특성 선택

```python
first_order = (
    order_level
    .sort_values(['customer_unique_id', 'order_purchase_timestamp', 'order_id'])
    .drop_duplicates('customer_unique_id')
    [['customer_unique_id', 'category_count', 'review_score']]
)
```

고객의 주문을 시간순으로 정렬한 뒤 첫 번째 주문만 남겨 첫 주문 카테고리 수와 첫 리뷰 점수를 가져옵니다. 이 방식은 이후 주문의 정보가 첫 주문에 섞이는 데이터 누수를 막습니다.

`first_order`는 다음처럼 고객별 최초 주문의 특성만 남습니다.

| customer_unique_id | category_count | review_score |
|---|---:|---:|
| customer_1 | 2 | 4.0 |
| customer_2 | 1 | 5.0 |

`customer_1`의 두 번째 주문 평점 2점은 이 표에 들어가지 않습니다.

---

## 6. 30일 이내 재구매 여부 계산

### 6.1 같은 날짜 주문은 한 번으로 처리

```python
purchase_dates = order_level.assign(
    purchase_date=order_level['order_purchase_timestamp'].dt.normalize()
)[['customer_unique_id', 'purchase_date']].drop_duplicates()
```

시간·분·초를 제거해 날짜만 남기고 같은 고객이 같은 날 만든 주문은 한 구매일로 처리합니다. 같은 날의 주문 분할이나 분리 결제를 재구매로 잘못 세는 것을 줄이기 위한 조치입니다.

#### 코드를 문법 단위로 나눠 보기

위 코드는 여러 작업을 한 줄로 연결한 **메서드 체이닝**입니다. 아래처럼 단계별로 나누면 같은 결과가 만들어집니다.

```python
# 1단계: 주문 시각에서 시간 부분을 00:00:00으로 맞춘 Series 생성
normalized_dates = order_level['order_purchase_timestamp'].dt.normalize()

# 2단계: order_level을 복사하면서 purchase_date 열 추가
orders_with_purchase_date = order_level.assign(
    purchase_date=normalized_dates
)

# 3단계: 분석에 필요한 두 열만 DataFrame 형태로 선택
customer_purchase_dates = orders_with_purchase_date[
    ['customer_unique_id', 'purchase_date']
]

# 4단계: 같은 고객과 같은 구매일의 중복 행 제거
purchase_dates = customer_purchase_dates.drop_duplicates()
```

##### `order_level['order_purchase_timestamp']`

`order_level`에서 `order_purchase_timestamp` 한 열을 선택합니다. 대괄호 안에 하나의 컬럼명을 문자열로 넣었으므로 반환값은 1차원 `Series`입니다.

```text
0   2018-01-10 10:15:23
1   2018-01-10 16:42:05
2   2018-02-03 14:30:00
Name: order_purchase_timestamp, dtype: datetime64[ns]
```

##### `.dt`

`.dt`는 pandas의 날짜·시간 전용 접근자입니다. `Series` 안의 각 날짜 값에 날짜 관련 함수나 속성을 적용할 수 있게 해줍니다.

```python
order_level['order_purchase_timestamp'].dt.year       # 연도
order_level['order_purchase_timestamp'].dt.month      # 월
order_level['order_purchase_timestamp'].dt.day        # 일
order_level['order_purchase_timestamp'].dt.normalize() # 시간을 자정으로 변경
```

`.dt` 자체가 값을 바꾸는 함수는 아니며, 뒤에 붙는 `year`, `month`, `normalize()` 같은 날짜 기능으로 연결해주는 역할을 합니다. 해당 열이 문자열이면 `.dt`를 사용할 수 없으므로 앞 단계에서 `pd.to_datetime()`으로 변환해야 합니다.

##### `.normalize()`

`normalize()`는 각 날짜·시간 값의 연·월·일은 유지하고 시·분·초를 `00:00:00`으로 바꿉니다.

| 변환 전 | 변환 후 |
|---|---|
| 2018-01-10 10:15:23 | 2018-01-10 00:00:00 |
| 2018-01-10 16:42:05 | 2018-01-10 00:00:00 |
| 2018-02-03 14:30:00 | 2018-02-03 00:00:00 |

화면에는 `2018-01-10`처럼 날짜만 표시될 수 있지만 내부 자료형은 날짜·시간형인 `datetime64[ns]`로 유지됩니다. 문자열이나 Python의 `date` 객체로 바꾸는 작업은 아닙니다. 따라서 이후 날짜 차이 계산을 그대로 수행할 수 있습니다.

##### `.assign(purchase_date=normalized_dates)`

`assign()`은 DataFrame에 새 열을 추가한 **새 DataFrame을 반환**합니다. 이 코드에서는 `normalized_dates`를 `purchase_date`라는 이름의 열로 추가합니다.

```python
orders_with_purchase_date = order_level.assign(
    purchase_date=normalized_dates
)
```

| customer_unique_id | order_purchase_timestamp | purchase_date |
|---|---|---|
| customer_1 | 2018-01-10 10:15:23 | 2018-01-10 00:00:00 |
| customer_1 | 2018-01-10 16:42:05 | 2018-01-10 00:00:00 |
| customer_1 | 2018-02-03 14:30:00 | 2018-02-03 00:00:00 |

원본 `order_level`을 직접 수정하지 않는다는 점이 `order_level['purchase_date'] = normalized_dates` 방식과의 차이입니다. 반환된 결과를 변수에 저장해야 새 열을 계속 사용할 수 있습니다.

##### `[['customer_unique_id', 'purchase_date']]`

이중 대괄호는 여러 열을 선택할 때 사용합니다. 바깥 대괄호는 DataFrame 선택 문법이고, 안쪽 대괄호는 선택할 컬럼명의 리스트입니다.

```python
# Series 반환: 대괄호 한 쌍과 컬럼명 하나
order_level['customer_unique_id']

# DataFrame 반환: 대괄호 안에 컬럼명 리스트
order_level[['customer_unique_id', 'purchase_date']]
```

이 단계에서 주문 ID, 결제액, 리뷰 점수 등은 제외되고 고객 ID와 정규화된 구매일만 남습니다.

##### `.drop_duplicates()`

`drop_duplicates()`는 선택된 모든 열의 값이 동일한 중복 행을 제거하고 첫 번째 행만 남깁니다. 이 시점에는 두 열만 존재하므로 `customer_unique_id`와 `purchase_date`의 조합을 기준으로 중복을 판단합니다.

| customer_unique_id | purchase_date | 처리 결과 |
|---|---|---|
| customer_1 | 2018-01-10 | 유지 |
| customer_1 | 2018-01-10 | 중복 제거 |
| customer_1 | 2018-02-03 | 유지 |
| customer_2 | 2018-01-10 | 유지 |

날짜가 같아도 고객이 다르면 다른 구매 기록이므로 유지됩니다. 고객이 같아도 날짜가 다르면 역시 별개의 구매일로 유지됩니다.

#### 전체 변화 요약

```text
주문별 날짜·시간
→ normalize()로 시간 제거
→ assign()으로 purchase_date 열 추가
→ 필요한 고객 ID와 구매일 열만 선택
→ drop_duplicates()로 고객별 같은 날짜 중복 제거
→ 고객별 고유 구매일 표 완성
```

`purchase_dates`의 행 기준은 **고객별 고유 구매일**입니다.

| customer_unique_id | purchase_date |
|---|---|
| customer_1 | 2018-01-10 |
| customer_1 | 2018-02-03 |
| customer_2 | 2018-02-05 |

### 6.2 첫 구매일부터 지난 일수 계산

```python
first_dates = purchase_dates.groupby(
    'customer_unique_id'
)['purchase_date'].transform('min')

purchase_dates['days_after_first'] = (
    purchase_dates['purchase_date'] - first_dates
).dt.days
```

첫 구매일은 0일입니다. 다음 구매가 12일 뒤라면 `days_after_first=12`가 됩니다.

### 6.3 1~30일 사이 재구매 판별

```python
target_30d=('days_after_first', lambda days: bool(days.between(1, 30).any()))
```

- 1~30일 사이에 다른 구매일이 있으면 `True`
- 없으면 `False`
- 같은 날인 0일은 재구매에서 제외

### 6.4 관측창 통제

```python
repurchase_30d['target_30d_eligible'] = (
    repurchase_30d['first_purchase_date']
    <= observation_end - pd.Timedelta(days=30)
)
```

데이터 종료 직전에 처음 구매한 고객은 30일 동안 재구매할 기회가 없습니다. 이 고객을 비재구매자로 처리하지 않도록 데이터 종료일보다 최소 30일 전에 첫 구매한 고객만 30일 재구매율 계산에 사용합니다.

최종 `repurchase_30d`는 고객별로 다음과 같은 모습입니다.

| customer_unique_id | first_purchase_date | target_30d | target_30d_eligible |
|---|---|---|---|
| customer_1 | 2018-01-10 | True | True |
| customer_2 | 2018-02-05 | False | True |
| customer_3 | 데이터 종료 10일 전 | False | False |

`customer_3`은 재구매하지 않은 고객으로 확정한 것이 아니라 관찰 기간이 부족한 고객입니다. 따라서 30일 재구매율의 분모에서 빠집니다.

---

## 7. 군집 분석용 고객 표 만들기

```python
customer_features = (
    rfm.merge(first_order, on='customer_unique_id', validate='one_to_one')
    .merge(repurchase_30d, on='customer_unique_id', validate='one_to_one')
)
```

RFM, 첫 주문 특성, 재구매 여부를 고객 한 명당 한 행인 표로 합칩니다.

병합된 `customer_features`는 군집 분석 전 고객 마스터 테이블에 해당합니다.

| customer_unique_id | Recency | Frequency | Monetary | category_count | review_score | target_30d | target_30d_eligible |
|---|---:|---:|---:|---:|---:|---|---|
| customer_1 | 25 | 2 | 175.50 | 2 | 4.0 | True | True |
| customer_2 | 23 | 1 | 42.00 | 1 | 5.0 | False | True |

```python
clustered = customer_features.dropna(
    subset=['Recency', 'Monetary', 'category_count', 'review_score']
).query('category_count > 0').copy()
```

K-Means는 빈 값을 바로 처리할 수 없습니다. 카테고리 수가 0인 경우도 실제 미구매보다 상품 정보 누락일 가능성이 있어 제외합니다. 따라서 군집 결과는 첫 주문 리뷰와 카테고리 정보가 모두 있는 고객에 대한 결과입니다.

`clustered`는 `customer_features`와 컬럼 구조가 거의 같지만, 군집 입력에 필요한 값이 모두 존재하는 고객만 남은 표입니다. 실제 데이터에서는 전체 고객 94,986명 중 92,718명이 이 표에 포함됩니다.

---

## 8. 극단적인 구매액 영향 줄이기

```python
clustered['Monetary_for_clustering'] = clustered['Monetary'].clip(
    upper=clustered['Monetary'].quantile(0.99)
)
```

극소수 고객의 매우 큰 결제액이 군집 전체를 끌고 가지 않도록 상위 1% 경계보다 큰 값은 군집 계산에서만 경계값으로 바꿉니다. 고객을 삭제하지 않으며 군집별 실제 평균은 원래 `Monetary`로 계산합니다.

---

## 9. 로그 변환과 표준화

```python
cluster_input = pd.DataFrame({
    'Recency': np.log1p(clustered['Recency']),
    'Monetary': np.log1p(clustered['Monetary_for_clustering']),
    'category_count': np.log1p(clustered['category_count']),
    'review_score': clustered['review_score']
})
scaled_features = StandardScaler().fit_transform(cluster_input)
```

### 로그 변환

Recency와 Monetary처럼 값의 범위가 크고 한쪽으로 치우친 변수의 큰 값 간격을 줄입니다.

```text
원래 값: 10, 100, 1,000
로그값:  약 2.4, 4.6, 6.9
```

`log1p(x)`는 `log(1+x)`를 계산해 값이 0이어도 사용할 수 있습니다. 리뷰 점수는 이미 1~5 범위라 로그 변환하지 않습니다.

### 표준화

Monetary는 수백 단위이고 리뷰는 1~5점입니다. 그대로 거리를 계산하면 금액이 결과를 거의 결정합니다. `StandardScaler`는 각 변수를 평균 0, 표준편차 1의 공통 척도로 바꿉니다.

키는 cm, 몸무게는 kg인 서로 다른 단위를 “평균에서 얼마나 떨어져 있는가”라는 같은 기준으로 바꾸는 것과 비슷합니다.

변환 단계의 표 모양은 다음처럼 이해할 수 있습니다.

| 단계 | Recency | Monetary | category_count | review_score |
|---|---:|---:|---:|---:|
| 원본 고객 값 | 300일 | 1,000헤알 | 2개 | 4점 |
| 로그 변환 후 | 5.71 | 6.91 | 1.10 | 4.00 |
| 표준화 후 예시 | 0.85 | 1.42 | 3.10 | 0.45 |

표준화 결과인 `scaled_features`는 컬럼명이 없는 숫자 배열이지만 열의 순서는 `Recency`, `Monetary`, `category_count`, `review_score`와 동일합니다. 표준화 후 숫자는 원래 단위가 아니라 평균에서 떨어진 정도를 나타냅니다.

---

## 10. k=2~8을 모두 시험

```python
for k in range(2, 9):
    model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels = model.fit_predict(scaled_features)
```

- `n_clusters`: 만들 그룹 수
- `n_init=20`: 서로 다른 시작점에서 20번 실행해 가장 좋은 결과 선택
- `fit_predict`: 군집을 만들고 각 고객의 군집 번호 반환

K-Means는 시작 위치에 따라 결과가 달라질 수 있어 여러 번 실행하는 것이 안전합니다.

---

## 11. Elbow Method

```python
'inertia': model.inertia_
```

Inertia는 각 고객과 자신이 속한 군집 중심 사이 거리의 제곱을 모두 더한 값입니다.

- 작을수록 군집 안에서 고객들이 가깝게 모임
- k를 늘리면 거의 항상 작아짐
- 감소 폭이 갑자기 둔해지는 굽은 지점이 효율적인 후보

이 지점이 사람의 팔꿈치처럼 보여 Elbow Method라고 부릅니다. 현재 결과에서 단독 Elbow 기준은 `k=5`입니다.

---

## 12. Elbow 지점 자동 선택

k와 Inertia를 0~1 범위로 바꾼 뒤 첫 점과 마지막 점을 잇는 직선에서 각 점이 얼마나 떨어져 있는지 계산합니다.

```python
k_validation['elbow_distance'] = (1 - normalized_k) - normalized_inertia
k_validation['elbow_strength'] = (
    k_validation['elbow_distance']
    / k_validation['elbow_distance'].max()
)
```

가장 강한 팔꿈치 후보의 강도가 1이 됩니다.

계산이 끝난 `k_validation`은 후보 k마다 한 행인 표입니다.

| k | inertia | smallest_cluster | elbow_distance | elbow_strength |
|---:|---:|---:|---:|---:|
| 2 | 294,673.73 | 20,779 | 0.0000 | 0.0000 |
| 3 | 204,158.20 | 668 | 0.2540 | 0.7098 |
| 4 | 149,300.75 | 668 | 0.3423 | 0.9565 |
| 5 | 110,085.74 | 668 | **0.3578** | **1.0000** |
| 6 | 97,811.70 | 668 | 0.2482 | 0.6937 |
| 7 | 88,228.67 | 668 | 0.1261 | 0.3524 |
| 8 | 79,497.46 | 668 | 0.0000 | 0.0000 |

직선에서 가장 멀리 떨어진 `k=5`가 곡선이 가장 크게 굽는 지점입니다. 따라서 별도의 결합 점수 없이 `elbow_k`를 최종 군집 수로 바로 사용합니다.

```python
elbow_k = int(
    k_validation.loc[k_validation['elbow_distance'].idxmax(), 'k']
)
selected_k = elbow_k
```

`idxmax()`는 `elbow_distance`가 가장 큰 행의 인덱스를 찾고, `.loc`는 그 행의 `k` 값을 가져옵니다. 현재 데이터에서는 `selected_k=5`입니다.

---

## 13. 최종 군집 생성과 프로필

```python
clustered['Cluster'] = fitted_models[selected_k][1]
```

선택한 `k=5` 모델의 군집 번호를 고객에게 붙입니다. 번호는 순위나 등급이 아닙니다. Cluster 4가 Cluster 0보다 우수하다는 뜻이 아니며, 설정을 바꾸면 번호도 달라질 수 있습니다.

```python
cluster_profile = clustered.groupby('Cluster', as_index=False).agg(
    customers=('customer_unique_id', 'nunique'),
    recency_mean=('Recency', 'mean'),
    frequency_mean=('Frequency', 'mean'),
    monetary_mean=('Monetary', 'mean'),
    category_count_mean=('category_count', 'mean'),
    review_score_mean=('review_score', 'mean')
)
```

각 군집의 고객 수와 평균 구매 특성을 계산합니다.

`clustered`에는 기존 고객 특성 옆에 `Cluster` 열이 추가됩니다.

| customer_unique_id | Recency | Monetary | category_count | review_score | Cluster |
|---|---:|---:|---:|---:|---:|
| customer_1 | 25 | 175.50 | 2 | 4.0 | 3 |
| customer_2 | 23 | 42.00 | 1 | 5.0 | 2 |

이를 군집별로 다시 집계한 표가 `cluster_profile`입니다.

| 군집 | 설명형 이름 | 고객 수 | 핵심 특징 |
|---:|---|---:|---|
| 0 | 저액 고평점 장기 미방문군 | 35,030명 | 평균 결제액 66.76헤알, 마지막 구매 후 약 296일 |
| 1 | 저평점 장기 미방문군 | 14,976명 | 평균 평점 1.54점, 마지막 구매 후 약 253일 |
| 2 | 희소 다중 카테고리군 | 668명 | 평균 2.02개 카테고리, 결제액 271.95헤알 |
| 3 | 최근 구매 고평점군 | 15,853명 | 마지막 구매 후 약 49일, 평점 4.60점 |
| 4 | 고액 고평점 장기 미방문군 | 26,191명 | 평균 결제액 303.33헤알, 평점 4.66점 |

최종 보고서에서는 번호만 쓰지 말고 설명형 이름을 함께 사용하는 것이 좋습니다.

---

## 14. 군집별 30일 재구매율

```python
repurchase_profile = (
    clustered[clustered['target_30d_eligible']]
    .groupby('Cluster', as_index=False)
    .agg(
        target_30d_eligible_customers=('customer_unique_id', 'nunique'),
        target_30d_repurchase_customers=('target_30d', 'sum'),
        target_30d_repurchase_rate=('target_30d', 'mean')
    )
)
```

30일의 관찰 기회가 충분한 고객만 사용해 군집별 분석 가능 고객 수, 실제 재구매 고객 수, 재구매율을 계산합니다.

`repurchase_profile`의 실제 결과 형태는 다음과 같습니다.

| Cluster | target_30d_eligible_customers | target_30d_repurchase_customers | target_30d_repurchase_rate |
|---:|---:|---:|---:|
| 0 | 35,030 | 75 | 0.21% |
| 1 | 14,725 | 78 | 0.53% |
| 2 | 615 | 11 | 1.79% |
| 3 | 11,114 | 73 | 0.66% |
| 4 | 26,191 | 344 | 1.31% |

이 표를 `cluster_profile`에 `Cluster` 기준으로 병합하면 고객 특성과 재구매 현황을 한 표에서 함께 볼 수 있습니다.

Cluster 2의 재구매율은 약 1.79%로 상대적으로 높지만 실제 재구매 고객은 11명뿐입니다. 표본이 작으면 몇 명 차이로 비율이 크게 변하므로 일반화하면 안 됩니다.

---

## 15. `target_30d`를 군집 입력에 넣지 않은 이유

군집 입력은 다음 네 변수입니다.

```text
Recency
Monetary
첫 주문 카테고리 수
첫 주문 리뷰 점수
```

`target_30d`는 고객을 나누는 데 쓰지 않고 군집을 만든 후 결과를 비교할 때만 사용합니다. 재구매 여부를 입력에 넣으면 “재구매 고객끼리 묶어 놓고 이 군집은 재구매율이 높다”고 말하는 순환 논리가 생길 수 있기 때문입니다.

따라서 군집별 재구매율은 **사후 기술 지표**이지 예측 결과나 인과효과가 아닙니다.

---

## 16. 결과 해석 시 주의사항

### 군집은 원인을 증명하지 않습니다

저평점 군집의 재구매율이 낮더라도 낮은 평점이 재구매하지 않은 직접 원인이라고 단정할 수 없습니다. 배송, 상품 품질, 구매 주기 등 다른 요인이 함께 작용했을 수 있습니다.

### 군집은 예측 모델이 아닙니다

K-Means는 비슷한 고객을 묶는 도구이며 특정 고객이 앞으로 재구매할 확률을 예측하는 모델이 아닙니다.

### 평균만으로 모든 고객을 설명할 수 없습니다

군집 평균이 200헤알이어도 모든 고객이 200헤알을 결제한 것은 아닙니다. 운영에 사용하기 전에 중앙값과 분포, 고객 수를 함께 확인해야 합니다.

### 정책 효과는 실험으로 검증해야 합니다

Fast Track이나 쿠폰이 재구매율을 높일지는 군집 분석만으로 알 수 없습니다. 처치군과 대조군을 나눈 실험이 필요합니다.

---

## 17. 최종 요약

RFM 이후 추가 코드는 다음 작업을 수행합니다.

1. 주문 단위 선집계로 결제액 중복을 방지합니다.
2. 고객마다 최근성, 구매 횟수, 실제 결제액을 계산합니다.
3. 첫 구매 카테고리 수와 리뷰 점수를 추가합니다.
4. 같은 날짜 주문을 제외한 30일 재구매 타겟을 만듭니다.
5. 로그 변환과 표준화로 변수의 단위 차이를 줄입니다.
6. Elbow Method로 `k=5`를 선택합니다.
7. 다섯 고객군의 규모, 구매 특성, 재구매 현황을 비교합니다.
8. 결과를 인과관계가 아닌 고객 특성의 기술적 비교로 제한합니다.

> 고객을 다섯 그룹으로 나누는 것 자체가 목적이 아니라, 전체 평균에 가려진 서로 다른 고객 특성을 발견하고 다음 조사와 실험의 우선순위를 정하는 것이 목적입니다.

