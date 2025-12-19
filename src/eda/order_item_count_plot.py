import matplotlib.pyplot as plt
import seaborn as sns

def plot_order_item_count(df):
    """
    Гистограмма для order_item_count по типу отзывов
    """
    df['review_label'] = df['review_score'].apply(
        lambda x: 'bad' if x <= 2 else ('good' if x >= 4 else 'neutral')
    )
    df_plot = df[df['review_label'] != 'neutral']

    plt.figure(figsize=(10,5))
    sns.histplot(
        data=df_plot,
        x='order_item_count',
        hue='review_label',
        bins=30,
        palette={'bad':'red', 'good':'green'},
        multiple="dodge"
    )
    plt.title("Distribution of Order Item Count by Review Type", fontsize=14)
    plt.xlabel("Order Item Count", fontsize=12)
    plt.ylabel("Number of Orders", fontsize=12)
    plt.tight_layout()
    plt.show()
