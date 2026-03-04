# test_simple.py
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def list_models():
    """List all available models"""
    try:
        response = requests.get(f"{BASE_URL}/models")
        response.raise_for_status()
        models = response.json()
        
        print("\n📦 Available models:")
        if not models:
            print("  No models found. Make sure Ollama is running and you have pulled models.")
            print("  To pull a model: ollama pull phi3:mini")
        else:
            for i, m in enumerate(models, 1):
                print(f"  {i}. {m['name']} - {m['size']}")
        return models
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure the server is running:")
        print("   python simple_benchmark.py")
        return []
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def test_single(model_name, prompt):
    """Test a single prompt"""
    print(f"\n🔍 Testing {model_name}...")
    print(f"Prompt: {prompt}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/benchmark",
            json={
                "model": model_name,
                "prompt": prompt,
                "max_tokens": 150
            },
            timeout=120  # 2 minute timeout
        )
        
        if response.status_code != 200:
            print(f"❌ Server returned error: {response.status_code}")
            print(f"Error details: {response.text}")
            return None
            
        result = response.json()
        
        print(f"\n✅ Response received:")
        print(f"  {result['response'][:150]}...")
        print(f"\n📊 Metrics:")
        print(f"  ⏱️  Total time: {result['total_time_seconds']:.2f}s")
        print(f"  🚀 First token: {result['time_to_first_token_ms']:.2f}ms")
        print(f"  ⚡ Speed: {result['tokens_per_second']:.2f} tokens/sec")
        print(f"  💾 Memory: {result['memory_percent']:.1f}%")
        print(f"  📝 Response length: {result['response_length']} chars")
        
        return result
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The model might be too slow or not responding.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        return None

def test_all_prompts(model_name):
    """Test all prompts on a model"""
    print(f"\n📊 Running all tests on {model_name}...")
    print("This will take a few minutes...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/benchmark/all-tests",
            params={"model": model_name, "max_tokens": 150},
            timeout=600  # 10 minute timeout
        )
        
        if response.status_code != 200:
            print(f"❌ Server returned error: {response.status_code}")
            print(f"Error details: {response.text}")
            return None
            
        results = response.json()
        
        print(f"\n📈 AVERAGES for {model_name}:")
        print(f"  ⚡ Avg Speed: {results['averages']['avg_tokens_per_second']:.2f} tokens/sec")
        print(f"  🚀 Avg Latency: {results['averages']['avg_time_to_first_token_ms']:.2f}ms")
        
        print(f"\n📋 Individual test results:")
        for test_name, metrics in results['individual_results'].items():
            if 'error' in metrics:
                print(f"  ❌ {test_name}: Failed - {metrics['error']}")
            else:
                print(f"  ✅ {test_name}: {metrics['tokens_per_second']:.2f} tokens/sec")
        
        return results
    except Exception as e:
        print(f"❌ Error running all tests: {e}")
        return None

def quick_test():
    """Quick test with a simple prompt"""
    print("\n🚀 Quick Test")
    print("-" * 30)
    
    # First, make sure we can connect
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Server is running: {response.json()['message']}")
    except:
        print("❌ Server is not running. Start it with: python simple_benchmark.py")
        return
    
    # List models
    models = list_models()
    
    if models:
        # Get the actual model name from the list
        first_model = models[0]['name']
        print(f"\n✅ Testing with: {first_model}")
        
        # Simple test
        test_single(first_model, "What is Python?")
    else:
        print("\n❌ No models found. Pull a model first:")
        print("   ollama pull phi3:mini")
        print("   ollama pull llama2")
        print("   ollama pull mistral")

def interactive_mode():
    """Interactive mode to test different prompts"""
    print("\n🎯 Interactive Test Mode")
    print("-" * 30)
    
    while True:
        print("\nOptions:")
        print("1. List models")
        print("2. Test a prompt")
        print("3. Run all tests")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            list_models()
        elif choice == '2':
            models = list_models()
            if models:
                print("\nAvailable models:")
                for i, m in enumerate(models, 1):
                    print(f"  {i}. {m['name']}")
                
                try:
                    model_idx = int(input("Select model number: ")) - 1
                    if 0 <= model_idx < len(models):
                        model_name = models[model_idx]['name']
                        prompt = input("Enter your prompt: ")
                        test_single(model_name, prompt)
                    else:
                        print("Invalid model number")
                except ValueError:
                    print("Please enter a valid number")
                except Exception as e:
                    print(f"Error: {e}")
        elif choice == '3':
            models = list_models()
            if models:
                print("\nAvailable models:")
                for i, m in enumerate(models, 1):
                    print(f"  {i}. {m['name']}")
                
                try:
                    model_idx = int(input("Select model number: ")) - 1
                    if 0 <= model_idx < len(models):
                        model_name = models[model_idx]['name']
                        test_all_prompts(model_name)
                    else:
                        print("Invalid model number")
                except ValueError:
                    print("Please enter a valid number")
                except Exception as e:
                    print(f"Error: {e}")
        elif choice == '4':
            print("Goodbye!")
            break

if __name__ == "__main__":
    print("=" * 50)
    print("OLLAMA BENCHMARK TEST CLIENT")
    print("=" * 50)
    
    # Run quick test first
    quick_test()
    
    # Ask if user wants interactive mode
    response = input("\n🔧 Enter interactive mode? (y/n): ").strip().lower()
    if response == 'y':
        interactive_mode()