from flask import Flask, render_template,session, request, jsonify,redirect, url_for
import csv
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash
from bson import ObjectId
import joblib
from model import ChatbotModel
import nltk
from fuzzywuzzy import fuzz

# Initialize the Flask application
app = Flask(__name__)

# Set a random secret key for session management (you can change it to something more secure)
app.secret_key = "e63b1247e22c7b388ff285c7f8a5d8008b68b8b7e7b8cbdcc0a63d95bcb606b5"

app.config["MONGO_URI"] = "mongodb+srv://windowuser23:Ali580943@cluster0.aqhqprh.mongodb.net/windowassistance?retryWrites=true&w=majority&appName=Cluster0"
mongo = PyMongo(app)

# Initialize ChatbotModel with the dataset
model = ChatbotModel(data_path='dataset/windows_assistance.csv')


# Define predefined responses and associated intents
predefined_responses = {
    "name_query": "I am Windows Ease Assistant."
}
name_query_phrases = [
    "what is your name", "who are you", "tell me your name",
    "what are you called", "introduce yourself", "your name please"
]

# Preprocessing utility (to normalize user input)
def preprocess_query(query):
    query = query.lower().strip()  # Convert to lowercase and remove extra spaces
    query = nltk.word_tokenize(query)  # Tokenize the text
    query = " ".join(query)  # Join tokens back into a normalized string
    return query

# Intent matching function
def match_intent(query, phrases):
    query = preprocess_query(query)
    for phrase in phrases:
        # Check similarity using fuzzy matching (threshold 80 for flexibility)
        if fuzz.partial_ratio(query, phrase) > 80:
            return True
    return False


def save_conversation(query, response, user_id):
    try:
        chat_id = str(ObjectId())  # Generate unique chat ID
        
        # Add chat to the "chats" array in the user's conversation
        mongo.db.users.update_one(
            {"_id": ObjectId(user_id)},  # Ensure user_id is ObjectId
            {"$push": {"conversations.chats": {
                "_id": chat_id,
                "query": query,
                "response": response,
                "timestamp": datetime.utcnow()
            }}}
        )
        print(f"Chat saved with chat_id: {chat_id}")
    except Exception as e:
        print(f"Error saving conversation: {e}")



# Save user queries and bot responses
def save_conversationdataset(query, response):
    # Clean the response by stripping extra newlines and unwanted whitespace
    formatted_response = ' '.join(response.splitlines()).strip()

    # Open the CSV file in append mode
    with open('dataset/user_queries_responses.csv', 'a', newline='', encoding='utf-8') as csvfile:
        # Set quoting to always quote all fields
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        # Write the query and the formatted response, both will be enclosed in double quotes
        writer.writerow([query, formatted_response])


# Route for the chatbot interface
@app.route('/')
def index():
    return render_template('index.html')

# Load the trained model and vectorizer
classifier = joblib.load('predication_model/model.pkl')
vectorizer = joblib.load('predication_model/vectorizer.pkl')

# Route for the chatbot interface
@app.route('/predication', methods=['GET'])
def predication():
    return render_template('predict_category.html')

# predication Category
@app.route('/predict', methods=['POST'])
def predict():
    # Get the query from the POST request
    data = request.get_json()
    query = data['query']
    
    # Transform the query using the vectorizer
    query_tfidf = vectorizer.transform([query])
    
    # Predict the category
    predicted_category = classifier.predict(query_tfidf)[0]
    
    # Return the predicted category as a response
    return jsonify({'query': query, 'predicted_category': predicted_category})

# Set your Gemini API key
genai.configure(api_key='AIzaSyCCu3DVz23qRK77Nxi3x71hndblJ11Q4Rg')

# Route for the Sign-In page
@app.route('/signin', methods=['GET'])
def signin():
    return render_template('sign-in.html')

# Route to handle the Sign-In form submission
@app.route('/signin', methods=['POST'])
def handle_signin():
    try:
        email = request.form.get("email")
        password = request.form.get("password")
        
        # Check if the user exists in the database
        user = mongo.db.users.find_one({"email": email})
        
        if user:
            # Compare the hashed password
            if check_password_hash(user['password'], password):
                # Store user data in session
                session["user_id"] = str(user["_id"])
                session["user_name"] = f"{user['first_name']} {user['last_name']}"
                
                # Return success message
                return jsonify({
                    "message": "Login successful",
                    "user_name": f"{user['first_name']} {user['last_name']}"
                })
            else:
                return jsonify({"error": "Incorrect password!"}), 401
        else:
            return jsonify({"error": "User not found!"}), 404
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Route to render the signup page
@app.route('/signup', methods=['GET'])
def show_signup_form():
    return render_template('sign-up.html')

# Route to handle the signup form submission
@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        first_name = data.get("firstName")
        last_name = data.get("lastName")
        mobile = data.get("mobile")
        email = data.get("email")
        password = data.get("password")

        # Check for existing user
        if mongo.db.users.find_one({"email": email}):
            return jsonify({"error": "Email already exists!"}), 400

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Insert user into the database
        user_id = mongo.db.users.insert_one({
            "first_name": first_name,
            "last_name": last_name,
            "mobile": mobile,
            "email": email,
            "password": hashed_password
        }).inserted_id

        # Store user data in session
        session["user_id"] = str(user_id)
        session["user_name"] = f"{first_name} {last_name}"

        return jsonify({"message": "Account created successfully!"}), 201

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/dashboard')
def dashboard():
    if "user_id" in session:
        user_id = ObjectId(session["user_id"])  # Convert to ObjectId
        user = mongo.db.users.find_one(
            {"_id": user_id},
            {"conversations.chats": 1, "first_name": 1, "last_name": 1}
        )

        if user:
            user_chats = user.get("conversations", {}).get("chats", [])
            user_name = f"{user.get('first_name', '')} {user.get('last_name', '')}"
            return render_template('chat.html', user_name=user_name, user_chats=user_chats)
        else:
            print("User not found in database.")
            return redirect(url_for('signin'))
    else:
        print("No user_id in session.")
        return redirect(url_for('signin'))



@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    query = data.get('query', '')
    user_id = session.get("user_id")
    
    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    
     # Check for name-related intent
    if match_intent(query, name_query_phrases):
        return jsonify({
            "response": predefined_responses["name_query"]
        })
    
    # Get the response from the chatbot model
    try:
        model_response = model.get_response(query)
    except Exception as e:
        return jsonify({'error': f"Error fetching response from model: {str(e)}"}), 500
    
    # Generate rewritten response using Gemini API
    try:
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        gemini_response = gemini_model.generate_content(f"Rewrite the following query into a concise 3–4 line response:\n\n{query}")
        rewritten_response = "\n".join([chunk.text for chunk in gemini_response])
        save_conversation(query, rewritten_response, user_id)
    except Exception as e:
        return jsonify({'error': f"Error with Gemini API: {str(e)}"}), 500

    return jsonify({
        "response": rewritten_response  # Only return the rewritten Gemini response
    })


# Route to handle feedback
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.get_json()
    user_query = feedback['query']
    helpful = feedback['helpful']
    
    # Read the existing dataset
    dataset = pd.read_csv('dataset/windows_assistance.csv')

    # Find the index of the corresponding query
    for index, row in dataset.iterrows():
        if row['query'].lower() == user_query.lower():
            if helpful:
                dataset.at[index, 'helpful_count'] += 1
            else:
                dataset.at[index, 'not_helpful_count'] += 1
            break
    else:
        return jsonify({"status": "query not found"}), 404

    # Save the updated dataset back to the CSV
    dataset.to_csv('dataset/windows_assistance.csv', index=False)

    return jsonify({"status": "success"})


@app.route('/logout')
def logout():
    session.clear()  # Clear the session data
    return redirect('/signin')  # Redirect the user to the sign-in page

@app.route('/clear_chats', methods=['POST'])
def clear_chats():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'User not logged in'})

    user_id = session['user_id']
    print(f"Attempting to clear chats for user_id: {user_id}")

    try:
        # Ensure you're using ObjectId for the query
        result = mongo.db.users.update_one(
            {"_id": ObjectId(user_id)},  # Cast user_id into ObjectId if it's not already
            {"$set": {"conversations.chats": []}}
        )

        if result.modified_count > 0:
            message = "All chats cleared successfully."
        else:
            message = "No chats found to clear."

        return jsonify({'status': 'success', 'message': message})
    
    except Exception as e:
        print(f"Error clearing chats: {e}")
        return jsonify({'status': 'error', 'message': f'An error occurred: {str(e)}'})





if __name__ == '__main__':
    app.run(debug=True)
