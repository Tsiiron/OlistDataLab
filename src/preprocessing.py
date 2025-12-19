import pandas as pd

def preprocess_data(dfs):
    """
    Очищаем данные и создаем новые признаки для ML.
    Возвращает dfs с новым ключом 'orders_full'.
    """

    # --- Products ---
    if "olist_products_dataset" in dfs:
        dfs["olist_products_dataset"]["product_category_name"] = (
            dfs["olist_products_dataset"]["product_category_name"]
            .fillna("unknown")
        )
        print("[PREPROCESS] Products: заполнены пропуски в категориях")

    # --- Orders ---
    if "olist_orders_dataset" in dfs:
        date_cols = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
        for col in date_cols:
            dfs["olist_orders_dataset"][col] = pd.to_datetime(
                dfs["olist_orders_dataset"][col], errors="coerce"
            )
        print("[PREPROCESS] Orders: обработаны даты и временные признаки")

    # --- Payments ---
    if "olist_order_payments_dataset" in dfs:
        payments_agg = dfs["olist_order_payments_dataset"].groupby("order_id").sum().reset_index()
        dfs["olist_order_payments_dataset"] = payments_agg
        print("[PREPROCESS] Payments: агрегированы способы оплаты и сумма")

    # --- Order items ---
    if "olist_order_items_dataset" in dfs:
        items_agg = dfs["olist_order_items_dataset"].groupby("order_id").agg({
            "product_id": "first",
            "seller_id": "first",
            "price": "sum"
        }).rename(columns={"price":"order_total_payment"}).reset_index()
        dfs["olist_order_items_dataset"] = items_agg
        print("[PREPROCESS] Order items: агрегированы товары и продавцы")

    # --- Reviews ---
    if "olist_order_reviews_dataset" in dfs:
        dfs["olist_order_reviews_dataset"] = dfs["olist_order_reviews_dataset"][["order_id", "review_score"]]
        print("[PREPROCESS] Reviews: оставлен review_score")

    # --- orders_full ---
    orders_full = (
        dfs["olist_orders_dataset"]
        .merge(dfs["olist_order_reviews_dataset"], on="order_id", how="left")
        .merge(dfs["olist_order_payments_dataset"], on="order_id", how="left")
        .merge(dfs["olist_order_items_dataset"], on="order_id", how="left")
    )

    # --- Новые числовые признаки ---
    orders_full["delivery_time_days"] = (
        (orders_full["order_delivered_customer_date"] - orders_full["order_purchase_timestamp"]).dt.days
    )
    orders_full["processing_time_days"] = (
        (orders_full["order_approved_at"] - orders_full["order_purchase_timestamp"]).dt.days
    )
    orders_full["delivery_delay_days"] = (
        (orders_full["order_delivered_customer_date"] - orders_full["order_estimated_delivery_date"]).dt.days
    )

    # --- order_item_count через merge ---
    order_item_counts = dfs["olist_order_items_dataset"].groupby("order_id")["product_id"].count().reset_index()
    order_item_counts.rename(columns={"product_id": "order_item_count"}, inplace=True)
    orders_full = orders_full.merge(order_item_counts, on="order_id", how="left")
    orders_full["order_item_count"] = orders_full["order_item_count"].fillna(0)

    dfs["orders_full"] = orders_full
    print("[PREPROCESS] orders_full: финальная таблица собрана с новыми признаками")

    return dfs
