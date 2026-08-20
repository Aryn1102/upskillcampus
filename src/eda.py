def dataset_info(self):
    print("=" * 60)
    print("Dataset Shape")
    print("=" * 60)
    print(self.df.shape)

    print("\nColumns")
    print(self.df.columns.tolist())

    print("\nSummary Statistics")
    print(self.df.describe())

def target_distribution(self):

    plt.figure(figsize=(8,5))

    sns.histplot(
        self.df["% Silica Concentrate"],
        bins=40,
        kde=True
    )

    plt.title("Target Variable Distribution")

    plt.tight_layout()

    plt.savefig("figures/target_distribution.png")

    plt.close()

def correlation_heatmap(self):

    plt.figure(figsize=(16,12))

    sns.heatmap(
        self.df.corr(numeric_only=True),
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig("figures/correlation_heatmap.png")

    plt.close()

def boxplots(self):

    numeric = self.df.select_dtypes(include="number")

    for column in numeric.columns:

        plt.figure(figsize=(7,4))

        sns.boxplot(x=numeric[column])

        plt.title(column)

        filename = (
            column.replace("%","")
            .replace(" ","_")
            + "_boxplot.png"
        )

        plt.tight_layout()

        plt.savefig(f"figures/{filename}")

        plt.close()

def feature_distributions(self):

numeric = self.df.select_dtypes(include="number")

for column in numeric.columns:

    plt.figure(figsize=(7,4))

    sns.histplot(
        numeric[column],
        kde=True,
        bins=40
    )

    plt.title(column)

    filename = (
        column.replace("%","")
        .replace(" ","_")
        + "_hist.png"
    )

    plt.tight_layout()

    plt.savefig(f"figures/{filename}")

    plt.close()


def run(self):

    self.dataset_info()

    self.target_distribution()

    self.correlation_heatmap()

    self.boxplots()

    self.feature_distributions()

    print("\nEDA Completed")