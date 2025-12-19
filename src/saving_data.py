import os
from load_data import load_data
from preprocessing import preprocess_data


def save_dataset_for_datalens(
    data_dir=r"\dataset",
    output_path=r"\data_for_datalens\orders_full.csv"
):
    """
    Загружает данные, выполняет предобработку
    и сохраняет итоговый датасет в CSV для DataLens
    """

    # --- Загрузка данных ---
    dfs = load_data(data_dir=data_dir)

    # --- Предобработка ---
    dfs = preprocess_data(dfs)

    # --- Итоговый датасет ---
    orders_full = dfs["orders_full"]

    # --- Создание папки, если нет ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Сохранение CSV ---
    orders_full.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print("Датасет успешно сохранён")
    print(f"Файл: {output_path}")
    print(f"Строк: {orders_full.shape[0]}")
    print(f"Колонок: {orders_full.shape[1]}")


if __name__ == "__main__":

    save_dataset_for_datalens()
