import pandas as pd
import matplotlib.pyplot as plt


def plot_bad_reviews_by_product_category(
        df: pd.DataFrame,
        min_orders: int = 100,
        top_n: int = 15
):
    """
    Строит график доли плохих отзывов (1–2)
    по категориям товаров.

    min_orders — минимальное число заказов в категории
    top_n — сколько категорий показывать
    """

    # 1. Оставляем нужные столбцы
    data = df[['product_category_name', 'review_score']].dropna()

    # 2. Флаг плохого отзыва
    data['is_bad_review'] = data['review_score'] <= 2

    # 3. Агрегация по категориям
    stats = (
        data
        .groupby('product_category_name')
        .agg(
            total_orders=('review_score', 'count'),
            bad_orders=('is_bad_review', 'sum')
        )
        .reset_index()
    )

    # 4. Доля плохих отзывов
    stats['bad_review_share'] = (
            stats['bad_orders'] / stats['total_orders']
    )

    # 5. Фильтр малых категорий
    stats = stats[stats['total_orders'] >= min_orders]

    # 6. ТОП категорий по доле плохих отзывов
    stats = (
        stats
        .sort_values('bad_review_share', ascending=False)
        .head(top_n)
    )

    # 7. Построение графика
    plt.figure(figsize=(12, 6))
    plt.bar(
        stats['product_category_name'],
        stats['bad_review_share']
    )

    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Доля плохих отзывов (1–2)')
    plt.xlabel('Категория товара')
    plt.title('Доля плохих отзывов по категориям товаров')

    plt.tight_layout()
    plt.show()
