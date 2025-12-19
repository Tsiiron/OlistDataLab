import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_correlation_heatmap(orders_full: pd.DataFrame):
    numeric_cols = [
        "review_score",
        "delivery_time_days",
        "processing_time_days",
        "delivery_delay_days",
        "total_payment",
        "order_item_count"
    ]

    data = orders_full[numeric_cols].dropna()

    corr = data.corr(method="pearson")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Корреляционная матрица числовых признаков")
    plt.tight_layout()
    plt.show()
