from load_data import load_data
from preprocessing import preprocess_data

# числовые EDA
from eda.delivery_time_plot import plot_delivery_time
from eda.processing_time_plot import plot_processing_time
from eda.delivery_delay_plot import plot_delivery_delay
from eda.total_payment_plot import plot_total_payment
from eda.order_item_count_plot import plot_order_item_count

# категориальные EDA
from eda.categorical.product_category_plot import plot_bad_reviews_by_product_category
from eda.categorical.payment_type_plot import plot_bad_reviews_by_payment_type
from eda.categorical.customer_state_plot import plot_bad_reviews_by_customer_state
from eda.categorical.seller_state_plot import plot_bad_reviews_by_seller_state
from eda.categorical.order_status_plot import plot_bad_reviews_by_order_status

from eda.correlations.correlation_heatmap import plot_correlation_heatmap




def main():
    # Загрузка и предобработка
    dfs = load_data()
    dfs = preprocess_data(dfs)

    # Готовая объединённая таблица
    orders_full = dfs["orders_full"]

    # -------------------------------
    # ЧИСЛОВЫЕ РАСПРЕДЕЛЕНИЯ
    # -------------------------------
    plot_delivery_time(orders_full)
    plot_processing_time(orders_full)
    plot_delivery_delay(orders_full)
    plot_total_payment(orders_full)
    plot_order_item_count(orders_full)

    plot_correlation_heatmap(orders_full)

    # -------------------------------
    # КАТЕГОРИАЛЬНЫЕ РАСПРЕДЕЛЕНИЯ
    # -------------------------------
    plot_bad_reviews_by_product_category(orders_full)
    plot_bad_reviews_by_payment_type(orders_full)
    plot_bad_reviews_by_customer_state(orders_full)
    plot_bad_reviews_by_seller_state(orders_full)
    plot_bad_reviews_by_order_status(orders_full)

if __name__ == "__main__":
    main()

