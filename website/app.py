from flask import Flask, render_template, request, jsonify
from catboost import CatBoostClassifier
import pandas as pd
import os

app = Flask(__name__)


# ============================================================
# LOAD TRAINED CATBOOST MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "catboost_heat_reuse_model.cbm"
)

model = CatBoostClassifier()

try:
    model.load_model(MODEL_PATH)
    print("==============================================")
    print("CatBoost model loaded successfully.")
    print(f"Model path: {MODEL_PATH}")
    print("==============================================")
except Exception as e:
    print("==============================================")
    print("ERROR: Could not load CatBoost model.")
    print(e)
    print("==============================================")


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # RECEIVE DATA FROM WEBSITE
        # ----------------------------------------------------

        data = request.get_json()

        year = int(data["Year"])
        country = str(data["Country"])
        city = str(data["City"])
        facility_type = str(data["Facility_Type"])

        wue = float(data["WUE_L_per_kWh"])
        electricity = float(data["Daily_Electricity_Usage_MWh"])
        water = float(data["Daily_Water_Usage_Gallons"])

        water_stress = str(
            data["Surrounding_Water_Stress_Tier"]
        )

        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # IMPORTANT:
        # Column names MUST match the training dataset
        # ----------------------------------------------------

        input_data = pd.DataFrame([
            {
                "Year": year,
                "Country": country,
                "City": city,
                "Facility_Type": facility_type,
                "WUE_L_per_kWh": wue,
                "Daily_Electricity_Usage_MWh": electricity,
                "Daily_Water_Usage_Gallons": water,
                "Surrounding_Water_Stress_Tier": water_stress
            }
        ])

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_data)

        probabilities = model.predict_proba(input_data)

        # CatBoost returns something like:
        # [['Medium']]

        predicted_class = str(prediction[0][0])

        # Highest probability
        confidence = float(probabilities[0].max()) * 100

        # ----------------------------------------------------
        # GET PROBABILITY FOR EACH CLASS
        # ----------------------------------------------------

        class_names = model.classes_

        probability_dict = {}

        for class_name, probability in zip(
            class_names,
            probabilities[0]
        ):
            probability_dict[str(class_name)] = round(
                float(probability) * 100,
                2
            )

        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return jsonify({
            "success": True,
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "probabilities": probability_dict
        })

    except Exception as e:

        print("Prediction error:")
        print(e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )
