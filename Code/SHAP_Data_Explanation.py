import shap
import joblib

# Load the saved model and data
loaded_data = joblib.load('C:\\Users\\dave9\\PycharmProjects\\LoanPredictionMLProject\\Code\\Models\\saved_model_data.joblib')
X_val_resampled = loaded_data['X_val_resampled']

# Extract the model from loaded data
model = loaded_data['model']

# Set up SHAP explainer with background summary
background_summary = shap.sample(X_val_resampled, 1000)  # Adjust the number based on your available memory
explainer = shap.KernelExplainer(model.predict_proba, background_summary)

# Now use the selected samples to calculate SHAP values
shap_values = explainer.shap_values(background_summary)

feature_names = ["Income", "Age", "Experience", "Married/Single", "House_Ownership", "Car_Ownership", "Profession", "CITY", "STATE", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS"]

# Display the summary_plot of the label “0”.
shap.summary_plot(shap_values[0], background_summary, feature_names=feature_names)

# Display the summary_plot of the label “1”.
shap.summary_plot(shap_values[1], background_summary, feature_names=feature_names)

# Display the decision plot for Risk_Flag=0
shap.decision_plot(explainer.expected_value[0], shap_values[0], background_summary, feature_names=feature_names, ignore_warnings=True)

# Display the decision plot for Risk_Flag=1
shap.decision_plot(explainer.expected_value[1], shap_values[1], background_summary, feature_names=feature_names, ignore_warnings=True)






