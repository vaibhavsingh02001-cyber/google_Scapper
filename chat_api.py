"""
chat_api.py - Local development runner for Discovery Engine Chat API.
Run: python chat_api.py
"""
import uvicorn
from api.index import app

if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  Discovery Engine Chat API (Local Dev)")
    print(f"  Listening on: http://localhost:8001")
    print(f"  Endpoints   : /api/health, /api/chat, /health, /chat")
    print(f"{'='*55}\n")
    uvicorn.run(app, host="0.0.0.0", port=8001)
