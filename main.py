# model_comparison.py
import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def list_models():
    """Get list of available models"""
    try:
        response = requests.get(f"{BASE_URL}/models")
        return response.json()
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

def test_single_model(model_name, max_tokens=150):
    """Run all tests on one model and return results"""
    print(f"  Testing {model_name}...")
    try:
        response = requests.post(
            f"{BASE_URL}/benchmark/all-tests",
            params={"model": model_name, "max_tokens": max_tokens},
            timeout=600
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"    ✅ Done")
            return result
        else:
            print(f"    ❌ Failed: {response.status_code}")
            print(f"    Error: {response.text}")
            return None
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None

def compare_models():
    """Main comparison function"""
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON TOOL")
    print("="*60)
    
    # Get available models
    models = list_models()
    if not models:
        print("❌ No models found or server not running")
        print("Make sure:")
        print("  1. Server is running: python simple_benchmark.py")
        print("  2. You have models pulled: ollama list")
        return
    
    print("\nAvailable models:")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m['name']} ({m['size']})")
    
    # Let user select models
    print("\nEnter model numbers to compare (comma-separated, e.g., 1,2,3):")
    try:
        selections = input("Selection: ").strip().split(',')
        selected_models = []
        for sel in selections:
            idx = int(sel.strip()) - 1
            if 0 <= idx < len(models):
                selected_models.append(models[idx]['name'])
        
        if not selected_models:
            print("❌ No valid models selected")
            return
        
        print(f"\n🔬 Testing {len(selected_models)} models. This will take a few minutes...")
        
        # Test each model
        results = []
        for model in selected_models:
            result = test_single_model(model)
            if result and 'averages' in result:
                results.append({
                    "model": model,
                    "speed": result['averages']['avg_tokens_per_second'],
                    "first_token_ms": result['averages']['avg_time_to_first_token_ms'],
                    "total_time_ms": result['averages']['avg_total_response_time_ms'],
                    "response_length": result['averages']['avg_response_length_chars']
                })
        
        # Display results
        if results:
            print("\n" + "="*100)
            print("📈 COMPARISON RESULTS")
            print("="*100)
            
            # Table format
            print(f"{'Model':<25} {'Speed (t/s)':<15} {'First Token (ms)':<20} {'Total Time (ms)':<20} {'Response Length':<15}")
            print("-"*100)
            
            for r in results:
                print(f"{r['model']:<25} {r['speed']:<15.2f} {r['first_token_ms']:<20.2f} {r['total_time_ms']:<20.2f} {r['response_length']:<15.0f}")
            
            print("="*100)
            
            # Find best in each category
            if results:
                best_speed = max(results, key=lambda x: x['speed'])
                best_latency = min(results, key=lambda x: x['first_token_ms'])
                best_total = min(results, key=lambda x: x['total_time_ms'])
                
                print("\n🏆 AWARDS:")
                print(f"  ⚡ Fastest Generation: {best_speed['model']} ({best_speed['speed']:.2f} tokens/sec)")
                print(f"  🚀 Lowest First Token Latency: {best_latency['model']} ({best_latency['first_token_ms']:.2f}ms)")
                print(f"  ⏱️  Fastest Total Response: {best_total['model']} ({best_total['total_time_ms']:.2f}ms)")
                
                # Save results to file
                filename = f"model_comparison_{len(results)}_models.json"
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n💾 Results saved to: {filename}")
        else:
            print("\n❌ No valid results obtained")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def quick_compare(models_list):
    """Quick comparison without interactive selection"""
    print(f"\n🔬 Comparing: {', '.join(models_list)}")
    
    results = []
    for model in models_list:
        result = test_single_model(model)
        if result and 'averages' in result:
            results.append({
                "model": model,
                "speed": result['averages']['avg_tokens_per_second'],
                "first_token_ms": result['averages']['avg_time_to_first_token_ms'],
                "total_time_ms": result['averages']['avg_total_response_time_ms'],
                "response_length": result['averages']['avg_response_length_chars']
            })
    
    if results:
        # Print simple comparison
        print("\n" + "-" * 80)
        print(f"{'Model':<20} {'Speed (t/s)':<15} {'First Token (ms)':<18} {'Total Time (ms)':<18} {'Length':<10}")
        print("-" * 80)
        for r in results:
            print(f"{r['model']:<20} {r['speed']:<15.2f} {r['first_token_ms']:<18.2f} {r['total_time_ms']:<18.2f} {r['response_length']:<10.0f}")
        print("-" * 80)
    else:
        print("❌ No valid results obtained")
    
    return results

def debug_server_response():
    """Debug function to see what the server returns"""
    print("\n🔍 Debugging server response...")
    try:
        response = requests.get(f"{BASE_URL}/models")
        print(f"Models endpoint: {response.status_code}")
        print(f"Models data: {response.json()}")
        
        # Try a simple benchmark
        response = requests.post(
            f"{BASE_URL}/benchmark",
            json={
                "model": "phi3:mini",
                "prompt": "Hello",
                "max_tokens": 10
            },
            timeout=30
        )
        print(f"\nBenchmark endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"Benchmark response structure: {list(response.json().keys())}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Debug error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--debug":
            debug_server_response()
        else:
            # Command line usage: python model_comparison.py phi3:mini mistral:7b
            models_to_compare = sys.argv[1:]
            quick_compare(models_to_compare)
    else:
        # Interactive mode
        compare_models()