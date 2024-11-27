from flask import Flask, request, jsonify
from flask_cors import CORS
from models.portfolio_optimizer import train_model, optimize_portfolio
import pandas as pd

app = Flask(__name__)
CORS(app)  # Allow requests from Wix frontend

@app.route('/api/optimize', methods=['POST'])
def optimize():
    data = request.json.get("market_data")
    if not data:
        return jsonify({"error": "Market data is required"}), 400

    df = pd.DataFrame(data)
    portfolio = optimize_portfolio(df)
    return jsonify({"optimized_portfolio": portfolio})

@app.route('/api/train', methods=['POST'])
def train():
    data = request.json.get("market_data")
    if not data:
        return jsonify({"error": "Market data is required"}), 400

    df = pd.DataFrame(data)
    model = train_model(df)
    return jsonify({"message": "Model trained successfully"})

if __name__ == '__main__':
    app.run(debug=True)
