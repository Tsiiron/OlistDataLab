import pandas as pd
import matplotlib.pyplot as plt


def plot_bad_reviews_by_payment_type(df: pd.DataFrame):
    data = df[['payment_type', 'review_score']].dropna()
    data['is_bad_review'] = data['review_score'] <= 2

    stats = (
        data
        .groupby('payment_type')
        .agg(
            total_orders=('review_score', 'count'),
            bad_orders=('is_bad_review', 'sum')
        )
        .reset_index()
    )

    stats['bad_review_share'] = stats['bad_orders'] / stats['total_orders']
    stats = stats.sort_values('bad_review_share', ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(stats['payment_type'], stats['bad_review_share'])
    plt.ylabel('Доля плохих отзывов (1–2)')
    plt.xlabel('Способ оплаты')
    plt.title('Доля плохих отзывов по способам оплаты')
    plt.tight_layout()
    plt.show()
