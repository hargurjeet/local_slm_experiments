# simple_benchmark.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
import ollama
import time
import psutil
import json
from typing import Optional, List, Dict, Any
import uvicorn

# JSON Schema definitions for different test types
class MathResponse(BaseModel):
    question: str
    answer: float
    explanation: str
    steps: List[str] = Field(default_factory=list)

class CodeResponse(BaseModel):
    function_name: str
    code: str
    explanation: str
    time_complexity: Optional[str] = None
    space_complexity: Optional[str] = None

class StoryResponse(BaseModel):
    title: str
    characters: List[str]
    plot_summary: str
    story: str
    moral: Optional[str] = None

class ReasoningResponse(BaseModel):
    question: str
    answer: str
    reasoning_steps: List[str]
    final_conclusion: str

class GeneralResponse(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

# Enhanced test prompts with JSON schema instructions
# Ultra-simple prompts for better JSON compliance
TEST_PROMPTS = {
    "short": {
        "prompt": """Return ONLY this JSON object:
{"answer": "4", "confidence": 1.0}

Question: 2+2?""",
        "schema": GeneralResponse
    },
    "medium": {
        "prompt": """Return ONLY this JSON format:
{"answer": "your explanation", "confidence": 0.95}

Explain: What is machine learning?""",
        "schema": GeneralResponse
    },
    "long": {
        "prompt": """Return ONLY this JSON format:
{"title": "title", "characters": ["name1", "name2"], "plot_summary": "summary", "story": "full story", "moral": "moral"}

Write a 2-sentence story about a robot.""",
        "schema": StoryResponse
    },
#     "reasoning": {
#         "prompt": """Return ONLY this JSON format:
# {"question": "question", "answer": "answer", "reasoning_steps": ["step1", "step2"], "final_conclusion": "conclusion"}

# Problem: bat ($1.10) costs $1 more than ball. Find ball price.""",
#         "schema": ReasoningResponse
#     },
#     "coding": {
#         "prompt": """Return ONLY this JSON format:
# {"function_name": "name", "code": "def name():\\n    pass", "explanation": "explanation", "time_complexity": "O(1)", "space_complexity": "O(1)"}

# Write function to check if number is even.""",
#         "schema": CodeResponse
#     }
}

# Request/Response models
class BenchmarkRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 1024
    max_retries: int = 2

class BenchmarkResponse(BaseModel):
    model: str
    prompt: str
    response: str
    parsed_response: Optional[Dict[str, Any]] = None
    validation_success: bool = True
    validation_error: Optional[str] = None
    retry_count: int = 0
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

def validate_json_response(response_text: str, schema_model: BaseModel) -> tuple[bool, Optional[Dict], Optional[str]]:
    """Strictly validate response - must be pure JSON, no extraction"""
    try:
        print(f' Validating JSON response: - {response_text}')  # Print first 100 chars for debugging
        # Try to parse as JSON - if this fails, it's not valid JSON
        parsed = json.loads(response_text)
        
        # Validate against schema
        validated = schema_model(**parsed)
        
        return True, validated.model_dump(), None
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON (must be pure JSON, no markdown or extra text): {str(e)}"
    except ValidationError as e:
        return False, None, f"Schema validation failed: {str(e)}"
    except Exception as e:
        return False, None, f"Unexpected error: {str(e)}"

def run_model_with_retry(model: str, prompt: str, schema_model: BaseModel, max_tokens: int, max_retries: int = 2):
    """Run model with retry mechanism for strict JSON validation"""
    retry_count = 0
    last_error = None
    response_text = ""
    first_token_time = None
    total_time = 0
    token_count = 0
    
    while retry_count <= max_retries:
        try:
            start_time = time.time()
            first_token_time = None
            response_text = ""
            token_count = 0
            
            # Prepare messages based on retry count
            if retry_count == 0:
                messages = [{'role': 'user', 'content': prompt}]
            else:
                # More strict reprompt on failure
                retry_prompt = f"""
                Your previous response was not valid JSON.
                Error: {last_error}

                You MUST respond with ONLY a valid JSON object. No markdown, no backticks, no additional text, no explanations.
                Just the raw JSON object.

                Original instruction:
                {prompt}
                """
                messages = [{'role': 'user', 'content': retry_prompt}]
            
            # Stream to measure first token
            stream = ollama.chat(
                model=model,
                messages=messages,
                stream=True,
                options={
                    "num_predict": max_tokens,
                    "temperature": 0.1,  # Lower temperature for more consistent JSON
                    "stop": ["```", "```json"]  # Try to prevent markdown
                }
            )
            
            for chunk in stream:
                if first_token_time is None:
                    first_token_time = (time.time() - start_time) * 1000
                
                if 'message' in chunk and 'content' in chunk['message']:
                    response_text += chunk['message']['content']
                    token_count += 1
            
            total_time = time.time() - start_time
            
            # Strict validation - must be pure JSON
            # First check if it contains markdown indicators (immediate fail)
            response_text_stripped = response_text.strip()
            if response_text_stripped.startswith('```') or '```json' in response_text_stripped:
                last_error = "Response contains markdown code blocks. Must be pure JSON only."
                retry_count += 1
                print(f"  Retry {retry_count}/{max_retries} - {last_error}")
                continue
            
            # Validate JSON
            is_valid, parsed_response, error_msg = validate_json_response(response_text_stripped, schema_model)
            
            if is_valid:
                return {
                    "response": response_text,
                    "parsed_response": parsed_response,
                    "validation_success": True,
                    "validation_error": None,
                    "retry_count": retry_count,
                    "time_to_first_token_ms": first_token_time or 0,
                    "total_time_seconds": total_time,
                    "tokens_per_second": token_count / total_time if total_time > 0 else 0,
                    "token_count": token_count
                }
            else:
                last_error = error_msg
                retry_count += 1
                print(f"  Retry {retry_count}/{max_retries} for {model} - {error_msg}")
                
        except Exception as e:
            last_error = str(e)
            retry_count += 1
            print(f"  Error on retry {retry_count}: {e}")
    
    # If we get here, all retries failed
    return {
        "response": response_text,
        "parsed_response": None,
        "validation_success": False,
        "validation_error": last_error or "Max retries exceeded",
        "retry_count": retry_count,
        "time_to_first_token_ms": first_token_time if first_token_time else 0,
        "total_time_seconds": total_time if total_time else 0,
        "tokens_per_second": 0,
        "token_count": token_count
    }

@app.get("/")
def root():
    return {"message": "Simple Ollama Benchmark API", "endpoints": ["/models", "/benchmark/all-tests"]}

@app.get("/models")
def list_models():
    """List available models"""
    try:
        models = ollama.list()
        model_list = []
        if 'models' in models:
            for m in models['models']:
                model_name = m.get('name', '') or m.get('model', '')
                model_size = m.get('size', '')
                if isinstance(model_size, int):
                    size_mb = model_size / (1024 * 1024)
                    model_size = f"{size_mb:.1f} MB"
                
                model_list.append({
                    "name": model_name,
                    "size": str(model_size),
                    "modified_at": m.get('modified_at', '')
                })
        
        return model_list
    except Exception as e:
        print(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/benchmark/all-tests")
def run_all_tests(model: str, max_tokens: int = 1024, max_retries: int = 2):
    """Run all test prompts on a model with strict JSON validation"""
    results = {}
    validation_stats = {
        "total_tests": len(TEST_PROMPTS),
        "passed_first_try": 0,
        "passed_after_retry": 0,
        "failed_all_retries": 0,
        "total_retries_used": 0
    }
    
    for test_name, test_data in TEST_PROMPTS.items():
        print(f"Running {test_name} test...")
        
        # Get system usage before
        usage_before = get_system_usage()
        
        # Run with retry mechanism
        result = run_model_with_retry(
            model=model,
            prompt=test_data["prompt"],
            schema_model=test_data["schema"],
            max_tokens=max_tokens,
            max_retries=max_retries
        )
        
        # Get system usage after
        usage_after = get_system_usage()
        
        # Update validation stats
        if result["validation_success"]:
            if result["retry_count"] == 0:
                validation_stats["passed_first_try"] += 1
            else:
                validation_stats["passed_after_retry"] += 1
        else:
            validation_stats["failed_all_retries"] += 1
        
        validation_stats["total_retries_used"] += result["retry_count"]
        
        # Store result with metrics
        results[test_name] = {
            "prompt": test_data["prompt"][:100] + "...",
            "response": result["response"],
            "parsed_response": result["parsed_response"],
            "validation_success": result["validation_success"],
            "validation_error": result["validation_error"],
            "retry_count": result["retry_count"],
            "metrics": {
                "time_to_first_token_ms": result["time_to_first_token_ms"],
                "total_time_seconds": result["total_time_seconds"],
                "tokens_per_second": result["tokens_per_second"],
                "tokens_generated": result["token_count"],
                "cpu_percent": (usage_before["cpu"] + usage_after["cpu"]) / 2,
                "memory_percent": usage_after["memory"],
                "response_length": len(result["response"])
            }
        }
    
    # Calculate averages (only for successful tests)
    successful_results = [r for r in results.values() if r["validation_success"]]
    
    if successful_results:
        avg_tps = sum(r["metrics"]["tokens_per_second"] for r in successful_results) / len(successful_results)
        avg_first_token = sum(r["metrics"]["time_to_first_token_ms"] for r in successful_results) / len(successful_results)
        avg_total = sum(r["metrics"]["total_time_seconds"] for r in successful_results) / len(successful_results)
        avg_length = sum(r["metrics"]["response_length"] for r in successful_results) / len(successful_results)
    else:
        avg_tps = avg_first_token = avg_total = avg_length = 0
    
    return {
        "model": model,
        "individual_results": results,
        "validation_statistics": validation_stats,
        "averages": {
            "avg_tokens_per_second": avg_tps,
            "avg_time_to_first_token_ms": avg_first_token,
            "avg_total_response_time_seconds": avg_total,
            "avg_response_length_chars": avg_length
        },
        "summary": {
            "total_tests_run": len(TEST_PROMPTS),
            "successful_tests": len(successful_results),
            "failed_tests": len(TEST_PROMPTS) - len(successful_results),
            "json_compliance_rate": (len(successful_results) / len(TEST_PROMPTS)) * 100 if TEST_PROMPTS else 0
        }
    }

if __name__ == "__main__":
    print("=" * 50)
    print("Starting Ollama Benchmark Server with STRICT JSON Validation")
    print("=" * 50)
    print("Features:")
    print("  ✅ Strict JSON-only validation (no extraction)")
    print("  ✅ Immediate fail on markdown/backticks")
    print("  ✅ Automatic retry with stricter prompting")
    print("  ✅ Pydantic schema validation")
    print("=" * 50)
    print("Make sure Ollama is running (ollama serve)")
    print("Server will start at http://localhost:8000")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)