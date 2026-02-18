from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load models
regression_model = joblib.load("models/regression_model.pkl")
classification_model = joblib.load("models/classification_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form["age"])
        traveltime = int(request.form["traveltime"])
        studytime = int(request.form["studytime"])
        failures = int(request.form["failures"])
        freetime = int(request.form["freetime"])
        goout = int(request.form["goout"])
        Dalc = int(request.form["Dalc"])
        Walc = int(request.form["Walc"])
        health = int(request.form["health"])
        absences = int(request.form["absences"])

        # Backend validation
        if not (15 <= age <= 22):
            return "Age must be between 15 and 22"
        if not (1 <= traveltime <= 4):
            return "Travel time must be between 1 and 4"
        if not (1 <= studytime <= 4):
            return "Study time must be between 1 and 4"
        if not (0 <= failures <= 3):
            return "Failures must be between 0 and 3"
        if not (1 <= freetime <= 5):
            return "Free time must be between 1 and 5"
        if not (1 <= goout <= 5):
            return "Go out must be between 1 and 5"
        if not (1 <= Dalc <= 5):
            return "Weekday alcohol must be between 1 and 5"
        if not (1 <= Walc <= 5):
            return "Weekend alcohol must be between 1 and 5"
        if not (1 <= health <= 5):
            return "Health must be between 1 and 5"
        if not (0 <= absences <= 100):
            return "Absences must be between 0 and 100"

        features = np.array([[age, traveltime, studytime, failures,
                              freetime, goout, Dalc, Walc, health, absences]])

        predicted_marks = regression_model.predict(features)[0]
        predicted_pass = classification_model.predict(features)[0]

        result = {
            "marks": round(float(predicted_marks), 2),
            "pass": "Pass" if predicted_pass == 1 else "Fail"
        }

        return render_template("index.html", prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
