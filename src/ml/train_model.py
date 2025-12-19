import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from load_data import load_data
from preprocessing import preprocess_data

def train_ml_model():
    # --- Загрузка и предобработка ---
    dfs = load_data(data_dir=r"\dataset")
    dfs = preprocess_data(dfs)
    orders_full = dfs["orders_full"]

    # --- Целевая переменная: низкий отзыв 1-2 = 1, 3-5 = 0 ---
    orders_full["bad_review"] = orders_full["review_score"].apply(lambda x: 1 if x in [1,2] else 0)

    # --- Отбор признаков ---
    numeric_features = [
        "delivery_time_days",
        "processing_time_days",
        "delivery_delay_days",
        "order_item_count",
        "payment_value"
    ]
    categorical_features = [
        "payment_type",
        "seller_id"
    ]

    # --- Числовые признаки ---
    X_num = orders_full[numeric_features].fillna(0)

    # --- Категориальные признаки ---
    X_cat = orders_full[categorical_features].fillna("unknown")
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_encoded = pd.DataFrame(
        ohe.fit_transform(X_cat),
        columns=ohe.get_feature_names_out(categorical_features),
        index=X_cat.index
    )

    # --- Итоговый датасет ---
    X = pd.concat([X_num, X_cat_encoded], axis=1)
    y = orders_full["bad_review"]

    # --- Разделение на train/test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Обучение модели с балансировкой классов ---
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  #учитываем редкий класс
    )
    model.fit(X_train, y_train)

    # --- Прогноз с порогом 0.3 ---
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > 0.3).astype(int)

    # --- Оценка модели ---
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model, ohe, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    train_ml_model()

