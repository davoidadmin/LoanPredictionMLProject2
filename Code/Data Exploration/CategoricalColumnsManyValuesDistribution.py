import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")

# Specify the categorical features
categorical_features = ["Profession", "CITY", "STATE"]

# Set the number of top values to consider
top_n_values = 10

# Create separate plots for each feature
for feature in categorical_features:
    # Get the top N most frequent values
    top_values = df[feature].value_counts().nlargest(top_n_values).index

    # Create a new column with top values and 'Other' for remaining values
    df_top_values = df.copy()
    df_top_values[feature] = df_top_values[feature].where(df_top_values[feature].isin(top_values), 'Other')

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create overlaid histograms
    sns.histplot(data=df_top_values, x=feature, hue="Risk_Flag", multiple="stack", ax=ax, shrink=0.8)

    # Customize the plot
    ax.set_title(f"Distribution of Risk_Flag by {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")

    # Rotate the labels vertically
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

    # Show the plot
    plt.show()

