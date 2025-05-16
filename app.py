
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
import gradio as gr

# Load the dataset
df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

# Drop id column
df = df.drop('id', axis=1)

# Handle missing values
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Convert categorical variables
df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})

# Define features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature groups
numeric_features = ['age', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married',
                       'work_type', 'Residence_type', 'smoking_status']

# Pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_processed, y_train)

# Evaluation
y_pred = model.predict(X_test_processed)
y_pred_proba = model.predict_proba(X_test_processed)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Save model
joblib.dump(model, 'model/stroke_prediction_model.pkl')
joblib.dump(preprocessor, 'model/preprocessor.pkl')

# Load model and preprocessor
model = joblib.load('model/stroke_prediction_model.pkl')
preprocessor = joblib.load('model/preprocessor.pkl')

# Gradio function
def predict_stroke(gender, age, hypertension, heart_disease, ever_married, work_type,
                  Residence_type, avg_glucose_level, bmi, smoking_status):
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [Residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })

    processed_input = preprocessor.transform(input_data)
    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0][1]
    result = "High Risk of Stroke" if prediction == 1 else "Low Risk of Stroke"
    return f"{result} (Probability: {probability:.2%})"

# Gradio UI
inputs = [
    gr.Dropdown(["Male", "Female", "Other"], label="Gender"),
    gr.Slider(0, 100, label="Age"),
    gr.Radio([0, 1], label="Hypertension (0 = No, 1 = Yes)"),
    gr.Radio([0, 1], label="Heart Disease (0 = No, 1 = Yes)"),
    gr.Radio(["Yes", "No"], label="Ever Married"),
    gr.Dropdown(["Private", "Self-employed", "Govt_job", "children", "Never_worked"], label="Work Type"),
    gr.Radio(["Urban", "Rural"], label="Residence Type"),
    gr.Number(label="Average Glucose Level"),
    gr.Number(label="BMI"),
    gr.Dropdown(["formerly smoked", "never smoked", "smokes", "Unknown"], label="Smoking Status")
]

output = gr.Textbox(label="Stroke Risk Prediction")

gr.Interface(fn=predict_stroke,
             inputs=inputs,
             outputs=output,
             title="Stroke Risk Prediction System",
             description="Enter patient details to assess stroke risk").launch()
