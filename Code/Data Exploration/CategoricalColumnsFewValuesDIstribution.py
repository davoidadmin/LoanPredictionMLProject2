import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\venv\\Dataset\\loan_data.csv")

# Specify the categorical features
categorical_features = ["Married/Single", "House_Ownership", "Car_Ownership"]

# Create separate plots for each feature
for feature in categorical_features:
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create overlaid histograms
    sns.histplot(data=df, x=feature, hue="Risk_Flag", multiple="stack", ax=ax, shrink=0.8)

    # Customize the plot
    ax.set_title(f"Distribution of Risk_Flag by {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")

    # Show the plot
    plt.show()

