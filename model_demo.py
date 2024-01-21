import gradio as gr

import joblib
import pandas as pd

# Load model
model = joblib.load("models/poissonreg.pkl")

from gradio import Interface, Label

# Define input components
bogo = gr.Checkbox("Bogo Offer?")
paneer = gr.Checkbox("Paneer?")
day = gr.Radio(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], label="Day of the Week")
guest = gr.Checkbox("Guest?")
test = gr.Checkbox("Test?")
max_possible_footfall = gr.Slider(50, 500, value=335, label="How many people on campus?")
meal_type = gr.Radio(["Breakfast", "Lunch", "Dinner"])

# Preprocessing logic
def preprocess(day, meal_type):
    day_Saturday, day_Sunday, meal_type_Breakfast, meal_type_Lunch, meal_type_Dinner = 0, 0, 0, 0, 0
    if day in ["Saturday", "Sunday"]:
        day_Saturday = 1 if day == "Saturday" else 0
        day_Sunday = 1 if day == "Sunday" else 0
    if meal_type == "Breakfast":
        meal_type_Breakfast = 1
    elif meal_type == "Lunch":
        meal_type_Lunch = 1
    elif meal_type == "Dinner":
        meal_type_Dinner = 1
    return day_Saturday, day_Sunday, meal_type_Breakfast, meal_type_Lunch, meal_type_Dinner

def predict_footfall(bogo, paneer, day, guest, test, max_possible_footfall, meal_type):
    day_Saturday, day_Sunday, meal_type_Breakfast, meal_type_Lunch, meal_type_Dinner = preprocess(day, meal_type)
    X = pd.DataFrame({
        "bogo": [bogo],
        "paneer": [paneer],
        "day_Saturday": [day_Saturday],
        "day_Sunday": [day_Sunday],
        "guest": [guest],
        "test": [test],
        "max_possible_footfall": [max_possible_footfall],
        "meal_type_Breakfast": [meal_type_Breakfast],
        "meal_type_Lunch": [meal_type_Lunch],
        "meal_type_Dinner": [meal_type_Dinner]
    })
    X = X.astype({"bogo": "bool", 
                  "paneer": "bool", 
                  "day_Saturday": "bool", 
                  "day_Sunday": "bool", 
                  "guest": "bool", 
                  "test": "bool", 
                  "max_possible_footfall": "int64", 
                  "meal_type_Breakfast": "bool", 
                  "meal_type_Lunch": "bool", 
                  "meal_type_Dinner": "bool"})
    prediction = model.predict(X)[0]

    return int(prediction)

predicted_footfall = Label(label="Predicted Footfall")

interface = Interface(
    fn=predict_footfall,
    inputs=[bogo, paneer, day, guest, test, max_possible_footfall, meal_type],
    outputs=predicted_footfall,
    title="Plaksha University Footfall Prediction"
)

interface.launch(share=True)