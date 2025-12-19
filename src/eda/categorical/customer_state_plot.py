import pandas as pd
import matplotlib.pyplot as plt


def plot_bad_reviews_by_customer_state(
    df: pd.DataFrame,
    min_orders: int = 200,
    top_n: int = 15
):
    data = df[['customer_state', 'review_score']].dropna()
    data['is_bad_review'] = data['review_score'] <= 2

    stats = (
        data
        .groupby('customer_state')
        .agg(
            total_orders=('review_score', 'count'),
            bad_orders=('is_bad_review', 'sum')
        )
        .reset_index()
    )

    stats['bad_review_share'] = stats['bad_orders'] / stats['total_orders']
    stats = stats[stats['total_orders'] >= min_orders]
    stats = stats.sort_values('bad_review_share', ascending=False).head(top_n)

    plt.figure(figsize=(10, 5))
    plt.bar(stats['customer_state'], stats['bad_review_share'])
    plt.ylabel('Доля плохих отзывов (1–2)')
    plt.xlabel('Штат покупателя')
    plt.title('Доля плохих отзывов по штатам покупателей')
    plt.tight_layout()
    plt.show()
