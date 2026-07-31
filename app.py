import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from src.logger import logging

app = Flask(__name__)


def get_car_names():
    try:
        csv_path = os.path.join("Dataset", "cardekho_dataset.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return sorted(df["car_name"].unique().tolist())
    except Exception as e:
        logging.error(f"Error loading car names: {e}")
    return [
        "Maruti Swift",
        "Hyundai i20",
        "Honda City",
        "Toyota Fortuner",
        "Mahindra Scorpio",
    ]


@app.route("/", methods=["GET"])
def index():
    car_names = get_car_names()
    return render_template("home.html", car_names=car_names)


@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    car_names = get_car_names()
    if request.method == "GET":
        return render_template("home.html", car_names=car_names)
    else:
        try:
            data = CustomData(
                car_name=request.form.get("car_name"),
                vehicle_age=int(request.form.get("vehicle_age")),
                km_driven=int(request.form.get("km_driven")),
                seller_type=request.form.get("seller_type"),
                fuel_type=request.form.get("fuel_type"),
                transmission_type=request.form.get("transmission_type"),
                mileage=float(request.form.get("mileage")),
                engine=float(request.form.get("engine")),
                max_power=float(request.form.get("max_power")),
                seats=int(request.form.get("seats")),
            )
            pred_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            predicted_price = float(results[0])

            formatted_price = f"₹ {predicted_price:,.2f}"

            return render_template(
                "home.html",
                car_names=car_names,
                prediction_text=formatted_price,
                selected_inputs=request.form,
            )
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return render_template(
                "home.html",
                car_names=car_names,
                prediction_text=f"Error processing prediction: {str(e)}",
            )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
