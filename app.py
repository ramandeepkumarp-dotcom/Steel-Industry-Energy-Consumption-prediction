from flask import Flask, request, jsonify, send_file
import pandas as pd
import joblib

app = Flask(__name__, static_folder='.', static_url_path='')

model = joblib.load("model_pipeline.joblib")

@app.route('/')
def serve_index():
    # Serve your steel-prediction.html file instead of index.html
    return send_file('Templates/steel-prediction.html')
@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True)
    print("Received payload:", payload)

    import math
    hour = payload.get("Hour", 0)
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    # Clean categorical inputs
    load_type = payload.get("Load Type", "").strip()
    if not load_type:
        load_type = "Light Load"  # Replace with a valid default from your training data

    week_status = payload.get("WeekStatus", 1)
    if isinstance(week_status, int):
        week_status = "Weekday" if week_status == 1 else "Weekend"

    day_of_week = payload.get("Day_of_week", 0)
    if isinstance(day_of_week, int):
        day_map = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_of_week = day_map[day_of_week] if 0 <= day_of_week < 7 else "Monday"

    input_row = {
        "Lagging_Current_Reactive.Power_kVarh": payload.get("Lagging_Current_Reactive.Power_kVarh", 0),
        "Leading_Current_Reactive_Power_kVarh": payload.get("Leading_Current_Reactive_Power_kVarh", 0),
        "CO2(tCO2)": payload.get("CO2(tCO2)", 0.0),
        "Lagging_Current_Power_Factor": payload.get("Lagging_Current_Power_Factor", 0.0),
        "Leading_Current_Power_Factor": payload.get("Leading_Current_Power_Factor", 0.0),
        "NSM": payload.get("NSM", 0),
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month": payload.get("month", 1),
        "is_weekend": payload.get("is_weekend", 0),
        "WeekStatus": week_status,
        "Day_of_week": day_of_week,
        "Load Type": load_type
    }

    X_input = pd.DataFrame([input_row])
    print("DataFrame passed to model:\n", X_input)

    prediction = model.predict(X_input)[0]
    return jsonify({"prediction": float(prediction)})

if __name__ == "__main__":
    app.run(debug=True)