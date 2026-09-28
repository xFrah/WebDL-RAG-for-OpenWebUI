#!/usr/bin/env python3
import subprocess
import sys
import os

def main():
    print("Starting Open WebUI MCP proxy on port 8766...")
    
    # Run the mcpo proxy, pointing it to start our local server.py
    cmd = [
        "uv", "run", "mcpo",
        "--port", "8766",
        "--",
        "uv", "run", "server.py"
    ]
    
    try:
        # Execute the command and stream output directly to the terminal
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
