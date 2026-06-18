import gradio as gr
import pandas as pd
import pickle
import numpy as np

with open("student_rf_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

def predict_pressure(age_group, gender, current_education_level,
                     how_often_stressed, hours_of_sleep,
                     main_cause_of_stress):
    
    input_df = pd.DataFrame([{
        "Age Group": age_group,
        "Gender": gender,
        "Current Education Level": current_education_level,
        "How often do you feel stressed due to studies?": how_often_stressed,
        "How many hours do you sleep on average per night?": hours_of_sleep,
        "What is the main cause of your academic stress?": main_cause_of_stress
    }])

    prediction = model.predict(input_df)[0]
    return f"Pressure feelings: {np.clip(prediction, 0, 5):.2f}"

# Gradio inputs
inputs = [
    gr.Dropdown(["Under 15", "15–18", "19–22", "23–26", "27+"], label="Age Group"),
    gr.Dropdown(["Male", "Female"], label="Gender"),
    gr.Dropdown(["School", "College", "University", "Other"], label="Current Education Level"),
    gr.Dropdown(["Sometimes", "Often", "Always"], label="How often do you feel stressed due to studies?"),
    gr.Dropdown(["5–6", "7–8", "More than 8"], label="How many hours do you sleep on average per night?"),
    gr.Dropdown([
        "Exams and Grades Pressure",
        "Financial Conditions",
        "Testing Tension",
        "Other",
        "Prefer not to say",
        "Exams Stress",
        "Stress factors"
    ], label="What is the main cause of your academic stress?")
]

app = gr.Interface(
    fn=predict_pressure,
    inputs=inputs,
    outputs="text",
    title="Student Academic Pressure Predictor"
)

app.launch()
