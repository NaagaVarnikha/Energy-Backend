from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model/energy_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        country = data['country']
        year = int(data['year'])
        prediction = predict_energy(country, year)
        return jsonify({'prediction': f"{prediction}"})
    except Exception as e:
        return jsonify({'prediction': f'Error: {str(e)}'}), 400

def predict_energy(country, year):
    try:
        # Load dataset just to get all countries
        data = pd.read_csv('data/cleaned_final_data.csv', encoding='utf-8', delimiter=',')

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
