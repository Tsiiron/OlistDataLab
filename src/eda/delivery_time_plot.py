import matplotlib.pyplot as plt
import seaborn as sns


def plot_delivery_time(orders_full):
    data = orders_full[
        (orders_full["delivery_time_days"].notna()) &
        (orders_full["delivery_time_days"] >= 0)
    ]

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=data,
        x="delivery_time_days",
        bins=30
    )

    plt.xlabel("Время доставки, дней")
    plt.ylabel("Количество заказов")
    plt.title("Распределение времени доставки заказов")
    plt.tight_layout()
    plt.show()
