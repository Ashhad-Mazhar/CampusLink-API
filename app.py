from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from inference import get_answer

app = Flask(__name__)

# CORS setup for the new domain
CORS(app, resources={r"/get_answer": {"origins": "https://campus-link-kohl.vercel.app"}})

# Rate limiter: 5 requests per minute per IP
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["10 per minute"]
)

@app.route('/get_answer', methods=['POST'])
@limiter.limit("5 per minute")
def get_answer_from_user():
    data = request.get_json()
    if not data or 'user_input' not in data:
        return jsonify({'error': 'No user input provided'}), 400

    user_input = data['user_input']
    answer = get_answer(user_input)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

