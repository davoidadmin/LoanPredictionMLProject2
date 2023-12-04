import shap
import joblib
import random
import numpy as np

# Load the saved model and data
loaded_data = joblib.load('C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\Code\\Models\\saved_model_data.joblib')
X_val_resampled = loaded_data['X_val_resampled']

# Extract the model from loaded data
model = loaded_data['model']

# Set up SHAP explainer with background summary
background_summary = shap.sample(X_val_resampled, 10)  # Adjust the number based on your available memory
explainer = shap.KernelExplainer(model.predict_proba, background_summary)
shap_values = explainer.shap_values(background_summary)

feature_names = ["Income", "Age", "Experience", "Married/Single", "House_Ownership", "Car_Ownership", "Profession", "CITY", "STATE", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS"]

# 3. Show a Waterfall Plot and a Force Plot for a randomly selected sample
random_sample_index = random.randint(0, len(background_summary) - 1)
shap.waterfall_plot(shap_values[random_sample_index], feature_names=feature_names)
shap.plots.force(explainer.expected_value, shap_values[random_sample_index], feature_names=feature_names)



