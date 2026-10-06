"""AutoOptML Application Launcher.

Run this script to launch the local web server:
    py run.py
"""
import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print(" AutoOptML: Conditional Search-Space Optimiser")
    print(" Open your web browser at: http://localhost:8000")
    print("=" * 70)
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
