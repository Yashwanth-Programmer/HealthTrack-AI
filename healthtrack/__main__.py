"""Run with:  python -m healthtrack  (serves the UI at http://127.0.0.1:8000)"""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "healthtrack.api:create_app",
        factory=True,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )
