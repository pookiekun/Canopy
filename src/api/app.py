# src/api/app.py

import os
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from src.api.routes import api_blueprint

def create_app():
    load_dotenv()
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "templates"))
    app = Flask(__name__, template_folder=template_dir)
    CORS(app) # Enables frontend dashboards to communicate with this API
    
    # Root interactive dashboard
    @app.route('/', methods=['GET'])
    def index():
        return render_template('index.html')

    # Root health check endpoint
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            "status": "healthy",
            "service": "CanopyReason",
            "version": "1.0.0-mvp"
        }), 200

    # Register API routes under /api
    app.register_blueprint(api_blueprint, url_prefix='/api')
    return app

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting CanopyReason API server on port {port}...")
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=port)