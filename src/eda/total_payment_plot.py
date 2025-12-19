import matplotlib.pyplot as plt
import seaborn as sns

def plot_total_payment(df):
    """
    Гистограмма для total_payment по типу отзывов
    """
    df['review_label'] = df['review_score'].apply(
        lambda x: 'bad' if x <= 2 else ('good' if x >= 4 else 'neutral')
    )
    df_plot = df[df['review_label'] != 'neutral']

    plt.figure(figsize=(10,5))
    sns.histplot(
        data=df_plot,
        x='total_payment',
        hue='review_label',
        bins=30,
        palette={'bad':'red', 'good':'green'},
        multiple="dodge"
    )
    plt.title("Distribution of Total Payment by Review Type", fontsize=14)
    plt.xlabel("Total Payment", fontsize=12)
    plt.ylabel("Number of Orders", fontsize=12)
    plt.tight_layout()
    plt.show()
