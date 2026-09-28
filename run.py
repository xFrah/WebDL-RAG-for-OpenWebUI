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
    
    # Ensure server.py runs in stdio mode, overriding any shell exports
    env = os.environ.copy()
    env["MCP_TRANSPORT"] = "stdio"
    env.pop("MCP_HTTP_PORT", None)
    
    try:
        # Execute the command and stream output directly to the terminal
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
