# simple_benchmark.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
import time
import psutil
from typing import Optional, List, Dict, Any
import uvicorn

# Simple test prompts
TEST_PROMPTS = {
    "short": "What is 2+2?",
    "medium": "Explain machine learning in 2 paragraphs.",
    "long": "Write a story about a robot learning to paint.",
    "reasoning": "If a bat and ball cost $1.10, and bat costs $1 more than ball, how much is ball?",
    "coding": "Write a Python function to check if a string is a palindrome."
}

# Request/Response models
class BenchmarkRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 100

class BenchmarkResponse(BaseModel):
    model: str
    prompt: str
    response: str
    time_to_first_token_ms: float
    total_time_seconds: float
    tokens_per_second: float
    cpu_percent: float
    memory_percent: float
    response_length: int

class ModelInfo(BaseModel):
    name: str
    size: str
    modified_at: str = ""

# Initialize FastAPI
app = FastAPI(title="Simple Ollama Benchmark")

def get_system_usage():
    """Get current system usage"""
    return {
        "cpu": psutil.cpu_percent(interval=0.1),
        "memory": psutil.virtual_memory().percent
    }

@app.get("/")
def root():
    return {"message": "Simple Ollama Benchmark API", "endpoints": ["/models", "/benchmark"]}

@app.get("/models")
def list_models():
    """List available models"""
    try:
        models = ollama.list()
        print("Raw Ollama response:", models)  # Debug print
        
        model_list = []
        if 'models' in models:
            for m in models['models']:
                # The model name might be in a different format
                # Try to get the model name correctly
                model_name = m.get('name', '')
                if not model_name and 'model' in m:
                    model_name = m['model']
                
                # Get size - handle different formats
                model_size = m.get('size', '')
                if isinstance(model_size, int):
                    # Convert bytes to readable format
                    size_mb = model_size / (1024 * 1024)
                    model_size = f"{size_mb:.1f} MB"
                
                model_list.append({
                    "name": model_name,
                    "size": str(model_size),
                    "modified_at": m.get('modified_at', '')
                })
        
        if not model_list:
            # Fallback: try to get models directly
            try:
                # Alternative way to list models
                result = ollama.list()
                for model in result.get('models', []):
                    model_list.append({
                        "name": model.get('model', model.get('name', 'unknown')),
                        "size": str(model.get('size', 'unknown')),
                        "modified_at": model.get('modified_at', '')
                    })
            except:
                pass
        
        return model_list
    except Exception as e:
        print(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/benchmark")
def run_benchmark(request: BenchmarkRequest):
    """Run a single benchmark test"""
    try:
        print(f"Running benchmark for model: {request.model}")
        print(f"Prompt: {request.prompt[:50]}...")
        
        # Get system usage before
        usage_before = get_system_usage()
        
        # Run the model
        start_time = time.time()
        first_token_time = None
        response_text = ""
        token_count = 0
        
        # Stream to measure first token
        stream = ollama.chat(
            model=request.model,
            messages=[{'role': 'user', 'content': request.prompt}],
            stream=True,
            options={"num_predict": request.max_tokens}
        )
        
        for chunk in stream:
            if first_token_time is None:
                first_token_time = (time.time() - start_time) * 1000  # Convert to ms
                print(f"First token received at {first_token_time:.2f}ms")
            
            if 'message' in chunk and 'content' in chunk['message']:
                response_text += chunk['message']['content']
                token_count += 1
        
        total_time = time.time() - start_time
        
        # Get system usage after
        usage_after = get_system_usage()
        
        # Calculate metrics
        tokens_per_second = token_count / total_time if total_time > 0 else 0
        
        print(f"Benchmark complete: {token_count} tokens in {total_time:.2f}s")
        
        return {
            "model": request.model,
            "prompt": request.prompt,
            "response": response_text,
            "time_to_first_token_ms": first_token_time or 0,
            "total_time_seconds": total_time,
            "tokens_per_second": tokens_per_second,
            "cpu_percent": (usage_before["cpu"] + usage_after["cpu"]) / 2,
            "memory_percent": usage_after["memory"],
            "response_length": len(response_text)
        }
        
    except Exception as e:
        print(f"Error in benchmark: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")

@app.post("/benchmark/all-tests")
def run_all_tests(model: str, max_tokens: int = 100):
    """Run all test prompts on a model"""
    results = {}
    
    for test_name, prompt in TEST_PROMPTS.items():
        print(f"Running {test_name} test...")
        try:
            request = BenchmarkRequest(model=model, prompt=prompt, max_tokens=max_tokens)
            result = run_benchmark(request)
            results[test_name] = result
        except Exception as e:
            print(f"Error in {test_name} test: {e}")
            results[test_name] = {"error": str(e)}
    
    # Calculate averages (skip failed tests)
    valid_results = [r for r in results.values() if 'error' not in r]
    if valid_results:
        avg_tps = sum(r['tokens_per_second'] for r in valid_results) / len(valid_results)
        avg_latency = sum(r['time_to_first_token_ms'] for r in valid_results) / len(valid_results)
    else:
        avg_tps = 0
        avg_latency = 0
    
    return {
        "model": model,
        "individual_results": results,
        "averages": {
            "avg_tokens_per_second": avg_tps,
            "avg_time_to_first_token_ms": avg_latency
        }
    }

if __name__ == "__main__":
    print("=" * 50)
    print("Starting Ollama Benchmark Server")
    print("=" * 50)
    print("Make sure Ollama is running (ollama serve)")
    print("Server will start at http://localhost:8000")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)