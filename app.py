from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load dataset
df = pd.read_csv("heart_data_my.csv")


# Function to predict heart health risk
def predict_risk(age, gender, smoking, alcohol, cholesterol, blood_sugar, physical_activity, family_history, BMI):
    # Find closest match in dataset based on given inputs
    df["diff"] = abs(df["age"] - age) + abs(df["cholesterol"] - cholesterol) + abs(df["BMI"] - BMI)
    closest_row = df.loc[df["diff"].idxmin()]

    risk = round(closest_row["hearthealth_risk_percentage"], 2)
    suggestion = closest_row["suggestions"]
    return risk, suggestion


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():
    age = int(request.form["age"])
    gender = int(request.form["gender"])
    smoking = int(request.form["smoking"])
    alcohol = int(request.form["alcohol"])
    cholesterol = float(request.form["cholesterol"])
    blood_sugar = float(request.form["blood_sugar"])
    physical_activity = int(request.form["physical_activity"])
    family_history = int(request.form["family_history"])
    BMI = float(request.form["BMI"])

    risk, suggestion = predict_risk(age, gender, smoking, alcohol, cholesterol, blood_sugar, physical_activity,
                                    family_history, BMI)

    return render_template("result.html", risk=risk, suggestion=suggestion)


if __name__ == "__main__":
    app.run(debug=True)
