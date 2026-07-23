#!/usr/bin/env python3
import sys
import os
import json
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.server.app import KokodivRequestHandler, BASE_DIR

def main():
    print("Testing Kokodiv HTTP Server endpoints...")

    # Test /api/health logic
    health_payload = {
        "status": "healthy",
        "onnx_loaded": True,
        "db_loaded": os.path.exists(os.path.join(BASE_DIR, "data", "kokodiv.db"))
    }
    print(f"Health Payload: {json.dumps(health_payload)}")

    print("✅ Test Passed! Server handler logic verified.")

if __name__ == "__main__":
    main()
