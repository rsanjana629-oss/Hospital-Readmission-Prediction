from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load("model/readmission_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        age = request.form["age"]
        gender = request.form["gender"]

        time_in_hospital = int(request.form["time_in_hospital"])
        num_lab_procedures = int(request.form["num_lab_procedures"])
        num_procedures = int(request.form["num_procedures"])
        num_medications = int(request.form["num_medications"])

        number_inpatient = int(request.form["number_inpatient"])
        number_emergency = int(request.form["number_emergency"])
        number_outpatient = int(request.form["number_outpatient"])
        number_diagnoses = int(request.form["number_diagnoses"])

        patient = pd.DataFrame([{

            "race": "Caucasian",
            "gender": gender,
            "age": age,
            "weight": np.nan,

            "admission_type_id": 1,
            "discharge_disposition_id": 1,
            "admission_source_id": 1,

            "time_in_hospital": time_in_hospital,
            "payer_code": np.nan,
            "medical_specialty": np.nan,

            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,

            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,

            "diag_1": "250.83",
            "diag_2": "276",
            "diag_3": "414",

            "number_diagnoses": number_diagnoses,

            "max_glu_serum": "None",
            "A1Cresult": "None",

            "metformin": "No",
            "repaglinide": "No",
            "nateglinide": "No",
            "chlorpropamide": "No",
            "glimepiride": "No",
            "acetohexamide": "No",
            "glipizide": "No",
            "glyburide": "No",
            "tolbutamide": "No",
            "pioglitazone": "No",
            "rosiglitazone": "No",
            "acarbose": "No",
            "miglitol": "No",
            "troglitazone": "No",
            "tolazamide": "No",
            "examide": "No",
            "citoglipton": "No",
            "insulin": "No",
            "glyburide-metformin": "No",
            "glipizide-metformin": "No",
            "glimepiride-pioglitazone": "No",
            "metformin-rosiglitazone": "No",
            "metformin-pioglitazone": "No",

            "change": "No",
            "diabetesMed": "No",

            "total_previous_visits":
                number_outpatient + number_emergency + number_inpatient,

            "total_treatment":
                num_lab_procedures + num_procedures + num_medications,

            "stay_category":
                "Short" if time_in_hospital <= 3 else
                "Medium" if time_in_hospital <= 7 else
                "Long" if time_in_hospital <= 14 else
                "Very Long",

            "previous_inpatient_visits": number_inpatient,
            "previous_emergency_visits": number_emergency,
            "previous_outpatient_visits": number_outpatient

        }])

        prediction = model.predict(patient)[0]

        probability = model.predict_proba(patient)[0][1] * 100

        if prediction == 1:
            result = "High Risk"
        else:
            result = "Low Risk"

        return render_template(
            "index.html",
            prediction=result,
            probability=round(probability, 2)
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)