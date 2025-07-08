from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from graph_config.graph import run_graph

app = Flask(__name__, template_folder='templates')
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message.strip():
        return jsonify({'response': 'Please enter a valid message.'})
    try:
        result = run_graph(user_message)
        ai_response = result.get('final_response', 'Sorry, something went wrong.')
    except Exception as e:
        ai_response = f"Error: {str(e)}"
    return jsonify({'response': ai_response})
