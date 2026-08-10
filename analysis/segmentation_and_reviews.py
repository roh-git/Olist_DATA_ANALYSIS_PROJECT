"""Corrected Olist segmentation and reproducible review issue analysis.

This is descriptive segmentation, not a repurchase/delivery prediction model.
Every source table is aggregated before joins to prevent many-to-many inflation.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42

ISSUE_PATTERNS = {
    "미수령": [
        r"nao recebi", r"nao chegou", r"nao foi entregue", r"aguardando.*produto",
        r"cad[eê].*produto", r"sem receber",
    ],
    "부분배송_누락": [
        r"faltou", r"faltando", r"so recebi", r"apenas um", r"pedido incompleto",
        r"entrega parcial", r"parcialmente", r"nao chegou tudo", r"nao recebi.*(um|uma)",
    ],
    "배송지연": [
        r"atras", r"fora do prazo", r"prazo.*(venceu|vencido)", r"demorou",
        r"demora.*entrega", r"ainda nao chegou",
    ],
    "배송완료_오표시": [
        r"consta.*entreg", r"como entregue", r"aparece.*entreg", r"status.*entreg",
        r"pedido aparece baixado", r"diz.*entreg",
    ],
    "오배송": [
        r"produto errado", r"produto diferente", r"veio outro", r"cor diferente",
        r"nao foi o que", r"diferente do anunciado",
    ],
    "파손_불량": [
        r"quebrad", r"defeito", r"nao funciona", r"danificad", r"avariad",
        r"estragad",
    ],
    "품질불만": [
        r"pessima qualidade", r"baixa qualidade", r"qualidade ruim", r"material fraco",
        r"produto ruim", r"nao recomendo",
    ],
    "환불_CS": [
        r"reembolso", r"estorno", r"devolu", r"nao respond", r"sem retorno",
        r"nao consigo.*(contato|falar)", r"aguardo.*resposta",
    ],
}


def read_csv(data_dir: Path, name: str, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(data_dir / name, usecols=columns)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value).lower())
    return "".join(character for character in text if not unicodedata.combining(character))


def build_customer_features(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    orders = read_csv(
        data_dir,
        "orders.csv",
        ["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
    )
    customers = read_csv(
        data_dir,
        "customers.csv",
        ["customer_id", "customer_unique_id"],
    )
    payments = read_csv(data_dir, "order_payments.csv", ["order_id", "payment_value"])
    items = read_csv(
        data_dir,
        "order_items.csv",
        ["order_id", "product_id", "price", "freight_value"],
    )
    products = read_csv(
        data_dir,
        "products.csv",
        ["product_id", "product_category_name"],
    )
    reviews = read_csv(
        data_dir,
        "order_reviews.csv",
        ["review_id", "order_id", "review_score", "review_comment_message"],
    ).drop_duplicates("review_id")

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce"
    )
    orders = orders[orders["order_status"].isin(
        ["approved", "processing", "invoiced", "shipped", "delivered"]
    )].copy()
    order_customer = orders.merge(customers, on="customer_id", how="inner")

    order_payment = payments.groupby("order_id", as_index=False).agg(
        order_payment=("payment_value", "sum")
    )
    item_category = items.merge(products, on="product_id", how="left")
    order_items = item_category.groupby("order_id", as_index=False).agg(
        item_count=("product_id", "count"),
        product_count=("product_id", "nunique"),
        category_count=("product_category_name", "nunique"),
        item_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
    )
    order_reviews = reviews.groupby("order_id", as_index=False).agg(
        review_score=("review_score", "mean"),
        review_count=("review_id", "nunique"),
    )
    order_level = (
        order_customer.merge(order_payment, on="order_id", how="left", validate="one_to_one")
        .merge(order_items, on="order_id", how="left", validate="one_to_one")
        .merge(order_reviews, on="order_id", how="left", validate="one_to_one")
    )
    if not order_level["order_id"].is_unique:
        raise ValueError("order_level must contain exactly one row per order")

    snapshot = order_level["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    observation_end = order_level["order_purchase_timestamp"].max()
    purchase_dates = (
        order_level.assign(
            purchase_date=order_level["order_purchase_timestamp"].dt.normalize()
        )
        .dropna(subset=["purchase_date"])
        [["customer_unique_id", "purchase_date"]]
        .drop_duplicates()
        .sort_values(["customer_unique_id", "purchase_date"])
    )
    first_dates = purchase_dates.groupby("customer_unique_id")["purchase_date"].transform("min")
    purchase_dates["days_after_first"] = (purchase_dates["purchase_date"] - first_dates).dt.days
    repurchase_30d = purchase_dates.groupby("customer_unique_id", as_index=False).agg(
        first_purchase_date=("purchase_date", "min"),
        target_30d=("days_after_first", lambda days: bool(days.between(1, 30).any())),
    )
    repurchase_30d["target_30d_eligible"] = (
        repurchase_30d["first_purchase_date"] <= observation_end - pd.Timedelta(days=30)
    )
    customer_rfm = order_level.groupby("customer_unique_id", as_index=False).agg(
        Recency=("order_purchase_timestamp", lambda x: (snapshot - x.max()).days),
        Frequency=("order_id", "nunique"),
        Monetary=("order_payment", "sum"),
    )
    first_orders = (
        order_level.sort_values(["customer_unique_id", "order_purchase_timestamp", "order_id"])
        .drop_duplicates("customer_unique_id", keep="first")
        [["customer_unique_id", "order_id", "category_count", "review_score"]]
        .rename(columns={"order_id": "first_order_id"})
    )
    customer_features = customer_rfm.merge(
        first_orders, on="customer_unique_id", how="inner", validate="one_to_one"
    ).merge(
        repurchase_30d, on="customer_unique_id", how="left", validate="one_to_one"
    )

    quality = {
        "unique_orders": int(order_level["order_id"].nunique()),
        "order_rows": int(len(order_level)),
        "order_id_is_unique": bool(order_level["order_id"].is_unique),
        "customers": int(customer_features["customer_unique_id"].nunique()),
        "customers_with_review": int(customer_features["review_score"].notna().sum()),
        "review_coverage": float(customer_features["review_score"].notna().mean()),
        "repeat_customer_rate": float(customer_features["Frequency"].gt(1).mean()),
        "target_30d_eligible_customers": int(customer_features["target_30d_eligible"].sum()),
        "target_30d_repurchase_rate": float(
            customer_features.loc[
                customer_features["target_30d_eligible"], "target_30d"
            ].mean()
        ),
        "multi_category_rate": float(customer_features["category_count"].gt(1).mean()),
        "monetary_total": float(customer_features["Monetary"].sum()),
        "included_order_payment_total": float(order_level["order_payment"].sum()),
        "raw_payment_total": float(payments["payment_value"].sum()),
    }
    return customer_features, reviews, quality


def select_k_and_cluster(
    customers: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    # Review is an intentional clustering feature, so the modeled population is
    # explicitly limited to customers with an observed first-order review.
    no_review_count = int(customers["review_score"].isna().sum())
    complete = customers.dropna(
        subset=["Recency", "Monetary", "category_count", "review_score"]
    ).copy()
    missing_category_count = int(complete["category_count"].le(0).sum())
    modeled = complete[complete["category_count"].gt(0)].copy()
    modeled["Monetary_for_clustering"] = modeled["Monetary"].clip(
        upper=modeled["Monetary"].quantile(0.99)
    )
    transformed = pd.DataFrame(
        {
            "Recency": np.log1p(modeled["Recency"]),
            "Monetary": np.log1p(modeled["Monetary_for_clustering"]),
            "category_count": np.log1p(modeled["category_count"]),
            "review_score": modeled["review_score"],
        },
        index=modeled.index,
    )
    scaled = StandardScaler().fit_transform(transformed)
    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(15_000, len(modeled))
    sample_positions = rng.choice(len(modeled), size=sample_size, replace=False)

    rows = []
    fitted: dict[int, tuple[KMeans, np.ndarray]] = {}
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(scaled)
        fitted[k] = (model, labels)
        sample_labels = labels[sample_positions]
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(
                    silhouette_score(scaled[sample_positions], sample_labels)
                ),
                "davies_bouldin": float(davies_bouldin_score(scaled, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(scaled, labels)),
                "smallest_cluster": int(pd.Series(labels).value_counts().min()),
            }
        )
    validation = pd.DataFrame(rows)
    # Quantify the elbow as the perpendicular/vertical distance between each
    # normalized inertia point and the straight line joining the first and last
    # candidates. Combine its relative strength with silhouette retention. The
    # geometric mean penalizes a k that is strong on only one of the two criteria.
    normalized_k = (validation["k"] - validation["k"].min()) / (
        validation["k"].max() - validation["k"].min()
    )
    normalized_inertia = (validation["inertia"] - validation["inertia"].min()) / (
        validation["inertia"].max() - validation["inertia"].min()
    )
    validation["inertia_drop_pct"] = validation["inertia"].pct_change().mul(-100)
    validation["elbow_distance"] = (1 - normalized_k) - normalized_inertia
    max_elbow_distance = validation["elbow_distance"].max()
    validation["elbow_strength"] = (
        validation["elbow_distance"] / max_elbow_distance
        if max_elbow_distance > 0
        else 0.0
    )
    validation["silhouette_retention"] = (
        validation["silhouette"] / validation["silhouette"].max()
    )
    validation["elbow_silhouette_score"] = np.sqrt(
        validation["elbow_strength"].clip(lower=0)
        * validation["silhouette_retention"].clip(lower=0)
    )

    # A minimum cluster size avoids selecting a numerically attractive but
    # operationally unusable micro-cluster.
    eligible = validation[validation["smallest_cluster"] >= 300]
    if eligible.empty:
        eligible = validation
    selected_k = int(
        eligible.sort_values(
            ["elbow_silhouette_score", "silhouette"], ascending=[False, False]
        ).iloc[0]["k"]
    )
    modeled["Cluster"] = fitted[selected_k][1]

    profile = modeled.groupby("Cluster", as_index=False).agg(
        customers=("customer_unique_id", "nunique"),
        recency_mean=("Recency", "mean"),
        recency_median=("Recency", "median"),
        frequency_mean=("Frequency", "mean"),
        monetary_mean=("Monetary", "mean"),
        monetary_median=("Monetary", "median"),
        category_count_mean=("category_count", "mean"),
        review_score_mean=("review_score", "mean"),
    )
    repurchase_profile = (
        modeled[modeled["target_30d_eligible"]]
        .groupby("Cluster", as_index=False)
        .agg(
            target_30d_eligible_customers=("customer_unique_id", "nunique"),
            target_30d_repurchase_customers=("target_30d", "sum"),
            target_30d_repurchase_rate=("target_30d", "mean"),
        )
    )
    profile = profile.merge(repurchase_profile, on="Cluster", how="left", validate="one_to_one")
    profile["customer_share"] = profile["customers"] / len(modeled)
    profile = profile.round(4)
    if len(profile) == 4:
        remaining = set(profile["Cluster"].astype(int))
        multi_cluster = int(profile.loc[profile["category_count_mean"].idxmax(), "Cluster"])
        remaining.remove(multi_cluster)
        recent_cluster = int(
            profile[profile["Cluster"].isin(remaining)]
            .sort_values("recency_mean")
            .iloc[0]["Cluster"]
        )
        remaining.remove(recent_cluster)
        low_review_cluster = int(
            profile[profile["Cluster"].isin(remaining)]
            .sort_values("review_score_mean")
            .iloc[0]["Cluster"]
        )
        remaining.remove(low_review_cluster)
        segment_names = {
            multi_cluster: "희소 다중 카테고리군",
            recent_cluster: "최근 구매 고평점군",
            low_review_cluster: "저평점 장기 미방문군",
            remaining.pop(): "고평점 장기 미방문군",
        }
    else:
        segment_names = {
            int(cluster): f"Cluster {int(cluster)}" for cluster in profile["Cluster"]
        }
    profile.insert(1, "Segment", profile["Cluster"].astype(int).map(segment_names))
    modeled["Segment"] = modeled["Cluster"].astype(int).map(segment_names)
    selection = {
        "selected_k": selected_k,
        "selection_rule": (
            "highest geometric mean of normalized elbow strength and silhouette "
            "retention among solutions with min cluster >= 300"
        ),
        "elbow_k": int(validation.loc[validation["elbow_distance"].idxmax(), "k"]),
        "silhouette_k": int(validation.loc[validation["silhouette"].idxmax(), "k"]),
        "modeled_customers": int(len(modeled)),
        "excluded_without_first_review": no_review_count,
        "excluded_missing_first_category": missing_category_count,
        "monetary_cap_p99_for_distance_only": float(customers["Monetary"].quantile(0.99)),
    }
    return modeled, profile, {
        "selection": selection,
        "validation": validation.to_dict(orient="records"),
    }


def classify_review_issues(
    clustered: pd.DataFrame,
    reviews: pd.DataFrame,
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    orders = read_csv(data_dir, "orders.csv", ["order_id", "customer_id"])
    customers = read_csv(
        data_dir, "customers.csv", ["customer_id", "customer_unique_id"]
    )
    review_events = (
        reviews.merge(orders, on="order_id", how="inner", validate="many_to_one")
        .merge(customers, on="customer_id", how="inner", validate="many_to_one")
        .merge(
            clustered[["customer_unique_id", "Cluster", "Segment"]],
            on="customer_unique_id",
            how="inner",
            validate="many_to_one",
        )
    )
    review_events = review_events.drop_duplicates("review_id").copy()
    review_events["normalized_text"] = review_events["review_comment_message"].map(normalize_text)
    for issue, patterns in ISSUE_PATTERNS.items():
        combined = re.compile("|".join(f"(?:{pattern})" for pattern in patterns))
        review_events[issue] = review_events["normalized_text"].map(
            lambda text: bool(combined.search(text))
        )
    issue_columns = list(ISSUE_PATTERNS)
    review_events["any_coded_issue"] = review_events[issue_columns].any(axis=1)

    low = review_events[
        review_events["review_score"].le(2)
        & review_events["review_comment_message"].notna()
    ].copy()
    summary_rows = []
    for (cluster, segment), group in low.groupby(["Cluster", "Segment"]):
        for issue in issue_columns:
            count = int(group[issue].sum())
            summary_rows.append(
                {
                    "Cluster": int(cluster),
                    "Segment": segment,
                    "issue": issue,
                    "low_score_text_reviews": int(len(group)),
                    "issue_reviews": count,
                    "issue_rate": count / len(group) if len(group) else 0.0,
                }
            )
    issue_summary = pd.DataFrame(summary_rows)
    audit = {
        "unique_cluster_reviews": int(review_events["review_id"].nunique()),
        "low_score_text_reviews": int(len(low)),
        "low_score_customers": int(low["customer_unique_id"].nunique()),
        "coded_issue_reviews": int(low["any_coded_issue"].sum()),
        "coded_issue_coverage": float(low["any_coded_issue"].mean()) if len(low) else 0.0,
        "counting_note": "Issue categories are multi-label; category totals may exceed review totals.",
    }
    return review_events, issue_summary, audit


def write_markdown_report(
    path: Path,
    quality: dict,
    clustering: dict,
    profile: pd.DataFrame,
    review_audit: dict,
    issue_summary: pd.DataFrame,
) -> None:
    selected_k = clustering["selection"]["selected_k"]
    lines = [
        "# Olist 고객 세분화 및 리뷰 분석 결과",
        "",
        "## 데이터 품질",
        "",
        f"- 주문 단위 행 수: {quality['order_rows']:,}",
        f"- 주문 ID 유일성: {quality['order_id_is_unique']}",
        f"- 전체 고객 수: {quality['customers']:,}",
        f"- 첫 주문 리뷰 보유 고객: {quality['customers_with_review']:,} ({quality['review_coverage']:.1%})",
        f"- 반복 구매 고객 비율: {quality['repeat_customer_rate']:.2%}",
        f"- 30일 재구매 관찰 가능 고객: {quality['target_30d_eligible_customers']:,}",
        f"- 전체 30일 재구매율: {quality['target_30d_repurchase_rate']:.2%}",
        f"- 다중 카테고리 첫 구매 고객 비율: {quality['multi_category_rate']:.2%}",
        f"- 고객 합산 결제액과 분석 포함 주문 결제액 차이: {quality['monetary_total'] - quality['included_order_payment_total']:,.2f}",
        f"- 취소·불가 주문을 포함한 전체 원 결제액: {quality['raw_payment_total']:,.2f}",
        "",
        "## 군집 선택",
        "",
        f"- 선택된 군집 수: {selected_k}",
        f"- 분석 고객: {clustering['selection']['modeled_customers']:,}",
        f"- 첫 주문 리뷰 부재로 제외: {clustering['selection']['excluded_without_first_review']:,}",
        f"- 첫 주문 카테고리 부재로 제외: {clustering['selection']['excluded_missing_first_category']:,}",
        f"- 단독 엘보우 기준 k: {clustering['selection']['elbow_k']}",
        f"- 단독 Silhouette 기준 k: {clustering['selection']['silhouette_k']}",
        "- 최종 선택 기준: 정규화한 엘보우 강도와 Silhouette 최고점 유지율의 기하평균이 가장 높은 해",
        "- Monetary 상위 1% 값은 고객을 삭제하지 않고 군집 거리 계산에서만 상한 처리",
        "",
        "## 군집 프로필",
        "",
        "`target_30d`는 첫 구매 후 1~30일의 서로 다른 구매일이 있는 경우이며, 데이터 종료일까지 30일 관찰 가능한 고객만 분모에 포함한다.",
        "",
        profile.to_markdown(index=False),
        "",
        "## 리뷰 분류 감사 정보",
        "",
        f"- 중복 제거된 군집 고객 리뷰: {review_audit['unique_cluster_reviews']:,}",
        f"- 텍스트가 있는 1~2점 리뷰: {review_audit['low_score_text_reviews']:,}",
        f"- 해당 저평점 리뷰 고객: {review_audit['low_score_customers']:,}",
        f"- 규칙으로 하나 이상의 이슈가 탐지된 리뷰: {review_audit['coded_issue_reviews']:,} ({review_audit['coded_issue_coverage']:.1%})",
        "- 이슈는 멀티라벨이므로 유형별 건수 합계가 전체 리뷰 수보다 클 수 있음",
        "",
        "## 군집별 저평점 리뷰 이슈",
        "",
        issue_summary.round(4).to_markdown(index=False),
        "",
        "## 해석 제한",
        "",
        "- 군집은 고객 특성의 유사성을 나타내며 이탈 원인이나 액션 효과를 증명하지 않는다.",
        "- 리뷰 키워드 분류는 재현 가능하지만 정답 라벨 기반 모델이 아니므로 표본 검수가 필요하다.",
        "- 리뷰를 남기지 않은 고객은 리뷰 점수를 사용하는 군집 분석에서 제외됐다.",
        "- Fast Track과 쿠폰의 효과 및 ROI는 대조군 실험으로 별도 검증해야 한다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(data_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    customers, reviews, quality = build_customer_features(data_dir)
    clustered, profile, clustering = select_k_and_cluster(customers)
    review_events, issue_summary, review_audit = classify_review_issues(
        clustered, reviews, data_dir
    )

    validation = pd.DataFrame(clustering["validation"])
    profile.to_csv(output_dir / "cluster_profile.csv", index=False)
    validation.to_csv(output_dir / "cluster_validation.csv", index=False)
    clustered[["customer_unique_id", "Cluster", "Segment"]].to_csv(
        output_dir / "customer_clusters.csv", index=False
    )
    issue_summary.to_csv(output_dir / "review_issue_summary.csv", index=False)
    review_events[
        [
            "review_id", "order_id", "customer_unique_id", "Cluster", "Segment", "review_score",
            "review_comment_message", *ISSUE_PATTERNS.keys(), "any_coded_issue",
        ]
    ].to_csv(output_dir / "deduplicated_cluster_reviews.csv", index=False)
    report = {
        "data_quality": quality,
        "clustering": clustering,
        "cluster_profile": profile.to_dict(orient="records"),
        "review_audit": review_audit,
        "review_issue_summary": issue_summary.to_dict(orient="records"),
    }
    (output_dir / "analysis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown_report(
        output_dir / "ANALYSIS_RESULTS.md",
        quality,
        clustering,
        profile,
        review_audit,
        issue_summary,
    )
    print(f"selected_k={clustering['selection']['selected_k']}")
    print(profile.to_string(index=False))
    print(f"results={output_dir.resolve()}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/results"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.data_dir, arguments.output_dir)
