import matplotlib.pyplot as plt
import seaborn as sns


def _style_plots():
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlepad"] = 14


def plot_histogram(df, column, bins=50):
    _style_plots()
    sns.histplot(data=df, x=column, kde=True, bins=bins)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()


def plot_box(df, column):
    _style_plots()
    sns.boxplot(data=df, x=column)
    plt.title(f"Box plot of {column}")
    plt.xlabel(column)
    plt.show()


def plot_scatter(df, x, y, hue=None, title=None, log_x=False):
    _style_plots()
    if log_x:
        plt.xscale("log")
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette="tab10", alpha=0.8)
    plt.title(title or f"{y} vs {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_count(df, column):
    _style_plots()
    ax = sns.countplot(data=df, x=column, order=df[column].value_counts().index)
    plt.title(f"Count of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height}", (p.get_x() + p.get_width() / 2.0, height), ha="center", va="bottom")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_pie(df, column, title=None):
    counts = df[column].value_counts()
    _style_plots()
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140, textprops={"fontsize": 10})
    plt.title(title or f"{column} Distribution")
    plt.axis("equal")
    plt.show()


def plot_correlation_matrix(df):
    _style_plots()
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Matrix")
    plt.show()


def visualize_data(df):
    plot_histogram(df, "Rotational speed [rpm]")
    plot_box(df, "Rotational speed [rpm]")
    plot_histogram(df, "Torque [Nm]")
    plot_box(df, "Torque [Nm]")
    plot_scatter(
        df,
        x="Process temperature [K]",
        y="Torque [Nm]",
        hue="Failure Type",
        title="Torque vs Process temperature by Failure Type",
    )
    plot_scatter(
        df,
        x="Rotational speed [rpm]",
        y="Torque [Nm]",
        hue="Failure Type",
        title="Torque vs Rotational speed by Failure Type",
    )
    plot_count(df, "Target")
    plot_count(df, "Type")
    plot_pie(df, "Failure Type", title="Failure Type Distribution")
    plot_correlation_matrix(df)
