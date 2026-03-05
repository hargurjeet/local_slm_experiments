import subprocess
import json

# List all models
result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
print(result.stdout)

# Or get the output in a more structured way
models = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
print("Your downloaded models:")
print(models.stdout)