# app.py

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import base64
import sys
from datetime import datetime
from flask import jsonify

app = Flask(__name__)
app.secret_key = "your_secret_key"
db_path = 'user_database.db'
model_path = 'trained_model.h5'

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
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_logs (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            start_time TEXT,
            Confused REAL DEFAULT 0,
            Engaged_Likes_Topic REAL DEFAULT 0,
            Engaged_Making_Involvement REAL DEFAULT 0,
            Not_Engaged_Chatting REAL DEFAULT 0,
            Not_Engaged_Not_Liked REAL DEFAULT 0,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT,
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

# Function to insert a new admin into the database
def insert_admin(username, password, email, mobile):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO admins (username, password, email, mobile) VALUES (?, ?, ?, ?)',
                   (username, password, email, mobile))
    conn.commit()
    conn.close()

# Function to retrieve an admin from the database by username
def get_admin(username):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM admins WHERE username = ?', (username,))
    admin = cursor.fetchone()
    conn.close()
    return admin

# Load the trained model for facial expression detection
model = None

def load_expression_model():
    global model
    if model is None:
        model = load_model(model_path, compile=False)

# Define emotion labels
EMOTIONS = ["Confused", "Engaged_Likes_Topic", "Engaged_Making_Involvement", "Not_Engaged_Chatting", "Not_Engaged_Not_Liked"]

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
            session['user_id'] = user[0]  # Set user_id in session
            return redirect(url_for('main'))
    return render_template('login.html')

# Route for main page (requires login)
@app.route('/main')
def main():
    if 'username' in session:
        # Check if model is loaded
        load_expression_model()
        model_loaded = True if model is not None else False
        return render_template('main.html', username=session['username'], model_loaded=model_loaded)
    return redirect(url_for('login'))

# Route for logging out
@app.route('/logout')
def logout():
    if 'username' in session:
        # Insert user log for the current session
        insert_user_log(session['user_id'], session['start_time'], session['expression_timings'])
        session.pop('username', None)
        session.pop('user_id', None)
        session.pop('start_time', None)
        session.pop('expression_timings', None)
    return redirect(url_for('login'))

# Route for exiting the server
@app.route('/exit')
def exit_server():
    sys.exit("Server terminated")

# Route for emotion detection
@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if model is not None:
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
        
        # Record expression timings
        record_expression_timings(detected_emotion)
        
        # Return the detected emotion
        return jsonify({'emotion': detected_emotion})
    else:
        return jsonify({'emotion': 'USER NOT FOUND'})

# Function to record expression timings
def record_expression_timings(detected_emotion):
    if 'expression_timings' not in session:
        session['expression_timings'] = {}
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if 'start_time' not in session:
        session['start_time'] = current_time
        
    expression_timings = session['expression_timings']
    
    # Calculate time spent on the previous expression
    previous_time = session.get('last_detection_time', current_time)
    time_difference = (datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S") - datetime.strptime(previous_time, "%Y-%m-%d %H:%M:%S")).seconds
    
    if detected_emotion in expression_timings:
        expression_timings[detected_emotion] += time_difference  # Increment time by time difference
    else:
        expression_timings[detected_emotion] = time_difference  # Initialize time for the expression
        
    # Update the last detection time
    session['last_detection_time'] = current_time

# Function to insert user log into the database
def insert_user_log(user_id, start_time, expression_timings):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO user_logs (user_id, start_time, Confused, Engaged_Likes_Topic, Engaged_Making_Involvement, Not_Engaged_Chatting, Not_Engaged_Not_Liked) VALUES (?, ?, ?, ?, ?, ?, ?)',
                   (user_id, start_time, expression_timings.get('Confused', 0), expression_timings.get('Engaged_Likes_Topic', 0), expression_timings.get('Engaged_Making_Involvement', 0), expression_timings.get('Not_Engaged_Chatting', 0), expression_timings.get('Not_Engaged_Not_Liked', 0)))
    conn.commit()
    conn.close()
# Route for handling the "Home" button
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Handle any POST request processing here
        pass  # Placeholder for any POST request processing
    return redirect(url_for('index'))


# Route for admin page
@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        admin = get_admin(username)
        if admin and admin[2] == password:  # Check if admin exists and password matches
            session['admin_username'] = username
            return redirect(url_for('admin_dashboard'))
    return render_template('admin_login.html')

# Your Flask route to render the admin dashboard
@app.route('/admin_dashboard')
def admin_dashboard():
    # Assuming you have fetched logs from somewhere
    logs = fetch_user_logs()

    # Pass logs, get_username, and get_roll_number functions to the template
    return render_template('admin_dashboard.html', logs=logs, get_username=get_username, get_roll_number=get_roll_number)

    

# Function to fetch all rows from user_logs table
def fetch_user_logs():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM user_logs')
    logs = cursor.fetchall()
    conn.close()
    return logs
    

# Function to retrieve username based on user ID
def get_username(user_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT username FROM users WHERE id = ?', (user_id,))
    username_row = cursor.fetchone()
    username = username_row[0] if username_row is not None else None
    conn.close()
    return username

# Function to retrieve roll number based on user ID
def get_roll_number(user_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT rollno FROM users WHERE id = ?', (user_id,))
    roll_number_row = cursor.fetchone()
    roll_number = roll_number_row[0] if roll_number_row is not None else None
    conn.close()
    return roll_number





# Route for admin registration
@app.route('/admin/register', methods=['GET', 'POST'])
def admin_register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        mobile = request.form['mobile']
        insert_admin(username, password, email, mobile)
        return redirect(url_for('admin'))
    return render_template('admin_register.html')


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
