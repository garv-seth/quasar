import subprocess
import sys

try:
    # Install Playwright dependencies
    print("Installing Playwright dependencies...")
    result = subprocess.run(["playwright", "install-deps"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)
    
    # Also install chromium browser
    print("Installing Chromium browser...")
    result = subprocess.run(["playwright", "install", "chromium"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)
    
    print("Setup completed successfully!")
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)