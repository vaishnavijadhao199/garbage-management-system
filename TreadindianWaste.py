from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import random
import time
import threading
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Generate historical waste data for India (2003-2022)
def generate_historical_data():
    years = list(range(2003, 2023))
    # Simulate increasing trend with some randomness
    base_waste = 20  # Million tons in 2003
    growth_rate = 0.08  # 8% annual growth
    
    waste_data = []
    for i, year in enumerate(years):
        # Add some randomness to the trend
        random_factor = random.uniform(0.95, 1.05)
        waste = base_waste * (1 + growth_rate) ** i * random_factor
        waste_data.append(round(waste, 2))
    
    return {'years': years, 'waste': waste_data}

# Generate current year (2023) monthly data
def generate_current_year_data():
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # Simulate monthly data with seasonal variations
    base_monthly = 2.5  # Million tons per month
    seasonal_factors = [0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 
                        1.15, 1.2, 1.1, 1.0, 0.9, 0.85]
    
    monthly_waste = []
    for i, month in enumerate(months):
        random_factor = random.uniform(0.9, 1.1)
        waste = base_monthly * seasonal_factors[i] * random_factor
        monthly_waste.append(round(waste, 2))
    
    return {'months': months, 'waste': monthly_waste}

# AI model for predicting future waste
def predict_future_waste(historical_data):
    # Prepare data for prediction
    X = np.array(historical_data['years']).reshape(-1, 1)
    y = np.array(historical_data['waste'])
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next 5 years
    future_years = [2023, 2024, 2025, 2026, 2027]
    future_X = np.array(future_years).reshape(-1, 1)
    predictions = model.predict(future_X)
    
    # Add some randomness to predictions
    predictions = [round(p * random.uniform(0.98, 1.02), 2) for p in predictions]
    
    return {'years': future_years, 'predictions': predictions}

# Initialize data
historical_data = generate_historical_data()
current_year_data = generate_current_year_data()
future_predictions = predict_future_waste(historical_data)

# Simulate real-time updates for current year
def update_current_year_data():
    global current_year_data
    
    while True:
        # Simulate new monthly data (in production, this would be real data)
        current_month = datetime.now().month - 1  # 0-indexed
        
        # Only update if we haven't reached December yet
        if current_month < 12:
            # Simulate a small change in the current month's data
            change = random.uniform(-0.1, 0.1)
            current_year_data['waste'][current_month] = round(
                max(0, current_year_data['waste'][current_month] + change), 2)
            
            # Recalculate annual total
            annual_total = sum(current_year_data['waste'])
            
            # Send update to frontend
            socketio.emit('waste_scatter_update', {
                'historical': historical_data,
                'current_year': current_year_data,
                'predictions': future_predictions,
                'annual_total': annual_total
            })
        
        time.sleep(5)  # Update every 5 seconds

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    thread = threading.Thread(target=update_current_year_data)
    thread.daemon = True
    thread.start()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)