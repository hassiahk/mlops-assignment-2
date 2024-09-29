import os
import pickle

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Load the pre-trained model
model = pickle.load(open("best_titanic_model.pkl", "rb"))

scaler = joblib.load("scaler.save")

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predicts passenger survival based on received data.

    Returns:
        JSON: Dictionary containing the predicted survival status.
    """
    try:
        # Get passenger data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing passenger data in request body."}), 400

        # Convert data to pandas DataFrame
        df = pd.DataFrame(data["features"])

        df["Pclass"] = df["Pclass"].astype(int)
        df["Age"] = df["Age"].astype(int)
        df["SibSp"] = df["SibSp"].astype(int)
        df["Parch"] = df["Parch"].astype(int)
        df["Fare"] = df["Fare"].astype(float)

        # Preprocess data (adapt to your specific feature names)
        df["Sex_male"] = np.where(df["Sex"] == "male", 1, 0)

        # Drop the original 'Sex' column
        df = df.drop(columns=["Sex"])

        # Handle Age values
        bins = [0, 18, 35, 60, np.inf]
        labels = ["child", "young_adult", "adult", "elderly"]
        df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

        # Drop the original 'Age' column
        df = df.drop(columns=["Age"])

        # Handle Embarked values
        df["Embarked_Q"] = np.where(df["Embarked"] == "Q", 1, 0)
        df["Embarked_S"] = np.where(df["Embarked"] == "S", 1, 0)

        # Drop the original 'Embarked' column
        df = df.drop(columns=["Embarked"])

        # Handle Family feature
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

        # Drop the original 'Parch' and 'SibSp' columns
        df = df.drop(["Parch", "SibSp"], axis=1)

        # Handle interaction feature
        df["AgeGroup_young_adult"] = np.where(df["AgeGroup"] == "young_adult", 1, 0)
        df["AgeGroup_adult"] = np.where(df["AgeGroup"] == "adult", 1, 0)
        df["AgeGroup_elderly"] = np.where(df["AgeGroup"] == "elderly", 1, 0)

        df = df.drop(columns=["AgeGroup"])

        # Scaling Fare and FamilySize features
        df[["Fare", "FamilySize"]] = scaler.transform(df[["Fare", "FamilySize"]])

        df = df[
            [
                "Pclass",
                "Fare",
                "FamilySize",
                "Sex_male",
                "Embarked_Q",
                "Embarked_S",
                "AgeGroup_young_adult",
                "AgeGroup_adult",
                "AgeGroup_elderly",
            ]
        ]

        # Make prediction using the loaded model
        prediction = model.predict(df)[0]

        # Return prediction as JSON response
        return jsonify({"survival": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
