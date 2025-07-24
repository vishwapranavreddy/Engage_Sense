from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import base64

app = Flask(__name__)
app.secret_key = "your_secret_key"
db_path = 'user_database.db'

# Function to initialize the database and create the user table if not exists
def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT,
            rollno TEXT,
            email TEXT,
            mobile TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Function to insert a new user into the database
def insert_user(username, password, rollno, email, mobile):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (username, password, rollno, email, mobile) VALUES (?, ?, ?, ?, ?)',
                   (username, password, rollno, email, mobile))
    conn.commit()
    conn.close()

# Function to retrieve a user from the database by username
def get_user(username):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user

# Load the trained model for facial expression detection
model = load_model('video_state_detection_model.h5', compile=False)

# Define emotion labels
EMOTIONS = ["Engaged_Likes_Topic", "Engaged_Confused", "Not_Engaged_Sleepy", "Not_Engaged_Not_Liked"]

# Function to preprocess the image for emotion detection
def preprocess_image(image):
    # Resize the image to match the input size of the model
    resized = cv2.resize(image, (224, 224))
    # Convert image to array
    img_array = img_to_array(resized)
    # Expand dimensions to create a batch of size 1
    processed_image = np.expand_dims(img_array, axis=0)
    # Normalize image pixel values
    processed_image = processed_image / 255.0
    return processed_image

# Route to serve the index.html file
@app.route('/')
def index():
    return render_template('index.html')

# Route for user registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        rollno = request.form['rollno']
        email = request.form['email']
        mobile = request.form['mobile']
        insert_user(username, password, rollno, email, mobile)
        return redirect(url_for('login'))
    return render_template('register.html')

# Route for user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user(username)
        if user and user[2] == password:  # Check if user exists and password matches
            session['username'] = username
            return redirect(url_for('main'))
    return render_template('login.html')

# Route for main page (requires login)
@app.route('/main')
def main():
    if 'username' in session:
        # Check if model is loaded
        model_loaded = True if model is not None else False
        return render_template('main.html', username=session['username'], model_loaded=model_loaded)
    return redirect(url_for('login'))

# Route for logging out
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Route for emotion detection
@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    # Get the base64 encoded image data from the request
    frame_data = request.json['frame']
    
    # Decode base64 data and convert to numpy array
    nparr = np.frombuffer(base64.b64decode(frame_data.split(',')[1]), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Perform emotion detection using the trained model
    prediction = model.predict(processed_image)
    detected_emotion = EMOTIONS[np.argmax(prediction)]
    
    # Return the detected emotion
    return jsonify({'emotion': detected_emotion})

# Route for handling the "Home" button
@app.route('/home', methods=['POST'])
def home():
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
