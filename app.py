from flask import Flask, render_template, request
import numpy as np
import joblib
import requests

app = Flask(__name__)

# Load your ML models
manual_model = joblib.load('crop_recommendation_model.pkl')
weather_model = joblib.load('Crop_recommendation_weather.pkl')

API_KEY = "596445241af79e1036beacb0fa19c585"  # Replace with your OpenWeatherMap API key

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/manual')
def manual_page():
    return render_template('manual.html')

@app.route('/weather')
def weather_page():
    return render_template('weather.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = manual_model.predict(input_data)[0]

          # Optional: show an image of the predicted crop if available
        crop_image_path = f"images/{prediction.lower()}.jpg"
        return render_template('manual.html', prediction=prediction, crop_image=crop_image_path)
    except Exception as e:
        return render_template('manual.html', prediction=f"Error: {str(e)}")

@app.route('/predict_weather', methods=['POST'])
def predict_weather():
    try:
        city = request.form['city']
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if data["cod"] != 200:
            return render_template('weather.html', prediction="City not found")

        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        rainfall = data.get("rain", {}).get("1h", 0)

        input_data = np.array([[temperature, humidity, rainfall]])
        prediction = weather_model.predict(input_data)[0]

         # Try to load a matching image
        crop_image_path = f"images/{prediction.lower()}.jpg"

        return render_template('weather.html', prediction=prediction, city=city,
                               temperature=temperature, humidity=humidity, rainfall=rainfall, crop_image=crop_image_path)
    except Exception as e:
        return render_template('weather.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)