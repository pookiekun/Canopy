# run.py
import os
import sys
from dotenv import load_dotenv

# Ensure safe output encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

load_dotenv()

from src.api.app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Launching CanopyReason MVP API server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
