from flask import Flask, send_file
from flask_socketio import SocketIO
import random

app = Flask(__name__)
# use threading backend to avoid eventlet imports/binding issues
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Serve the frontend HTML file
@app.route('/homepage.html')
def homepage():
    return send_file('homepage.html')  # Make sure homepage.html is in the same directory

@app.route('/wasteTrends.html')
def waste_trends():
    return send_file('wasteTrends.html')  # Make sure wasteTrends.html is in the same directory

# Generate historical data (2003-2022)
def generate_historical_data():
    years = list(range(2003, 2023))
    waste = [random.randint(20, 200) for _ in years]
    return {"years": years, "waste": waste}

# Generate forecast data (2023-2027)
def generate_forecast_data():
    years = list(range(2023, 2028))
    waste = [random.randint(180, 250) for _ in years]
    return {"years": years, "waste": waste}

# Background task to emit data every 5 seconds
def background_task():
    while True:
        data = {
            "historical": generate_historical_data(),
            "forecast": generate_forecast_data()
        }
        socketio.emit('waste_scatter_update', data)
        socketio.sleep(5)

if __name__ == '__main__':
    socketio.start_background_task(background_task)
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

# run putting: python indianWaste.py and open homepage.html and wasteTrends.html in browser at http://