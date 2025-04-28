from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os
import logging

app = Flask(__name__)
CORS(app)  # Allow all origins (restrict in production)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'energy_model.pkl')
logger.info("Loading model from: %s", model_path)
logger.info("Model file exists: %s", os.path.exists(model_path))
model = pickle.load(open(model_path, 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received predict request")
    try:
        data = request.get_json()
        logger.info("Request data: %s", data)
        country = data.get('country')
        year = int(data.get('year'))

        if not country or not year:
            logger.error("Missing country or year")
            return jsonify({'error': 'Missing country or year'}), 400

        prediction = predict_energy(country, year)
        if isinstance(prediction, str):
            logger.error("Prediction error: %s", prediction)
            return jsonify({'error': prediction}), 400

        return jsonify({'prediction': prediction})
    except ValueError:
        logger.error("Invalid year format")
        return jsonify({'error': 'Year must be a valid number'}), 400
    except Exception as e:
        logger.error("Error in predict: %s", str(e))
        return jsonify({'error': f'Error: {str(e)}'}), 500

def predict_energy(country, year):
    try:
        # Load dataset for country validation
        data_path = os.path.join(os.path.dirname(__file__),'cleaned_final_data.csv')
        logger.info("Loading data from: %s", data_path)
        logger.info("Data file exists: %s", os.path.exists(data_path))
        data = pd.read_csv(data_path, encoding='utf-8', delimiter=',')

        if country not in data['country'].unique():
            return "No data available for this country."

        # Get sorted list of countries used during model training
        all_countries = sorted(data['country'].unique().tolist())

        # One-hot encode the country
        country_encoded = [1 if c == country else 0 for c in all_countries]

        # Final input: [year, country_encoded...]
        X_input = [year] + country_encoded
        X_input_df = pd.DataFrame([X_input])

        # Predict using the model
        predicted_energy = model.predict(X_input_df)
        return round(predicted_energy[0], 2)

    except Exception as e:
        return f"Prediction error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
