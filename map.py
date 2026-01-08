pfrom flask import Flask, render_template
from flask_socketio import SocketIO, emit
import random
import time
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Simulate bin data and emit updates
def bin_update_simulator():
    while True:
        bin_data = {
            "id": random.randint(1, 5),
            "lat": 40.7128 + random.uniform(-0.01, 0.01),
            "lng": -74.0060 + random.uniform(-0.01, 0.01),
            "detected": True,
            "fill_level": random.randint(10, 100),
            "image": "https://picsum.photos/seed/" + str(random.randint(1,1000)) + "/200/200",
            "predicted_fill": random.randint(10, 100)
        }
        socketio.emit('bin_update', bin_data)
        time.sleep(5)

@app.route('/')
def index():
    return "Bin Monitoring Backend Running"

if __name__ == '__main__':
    # Start the bin update simulator in a background thread
    threading.Thread(target=bin_update_simulator, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000)