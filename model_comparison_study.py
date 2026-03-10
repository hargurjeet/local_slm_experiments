#!/usr/bin/env python3
"""
Model Comparison Study
Benchmarks llama 3.2B, Phi-4 mini, and mistral 7b on multiple metrics
"""

import requests
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional

BASE_URL = "http://localhost:8000"

# Models to benchmark
MODELS_TO_TEST = [
    "llama3.2:latest",  # Llama 3.2 3B
    "phi3:mini",         # Phi-3 mini
    "mistral:7b"    # Mistral 7B
]

class ModelBenchmark:
    """Class to handle model benchmarking and result collection"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results = []
        
    def check_server(self) -> bool:
        """Check if the benchmark server is running"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Server check failed: {e}")
            return False
    
    def list_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.base_url}/models")
            models = response.json()
            return [m['name'] for m in models]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def benchmark_model(self, model_name: str, max_tokens: int = 1024, max_retries: int = 2) -> Optional[Dict]:
        """Run comprehensive benchmark on a single model"""
        print(f"\n{'='*70}")
        print(f"🔬 Benchmarking: {model_name}")
        print(f"{'='*70}")
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_url}/benchmark/all-tests",
                params={
                    "model": model_name,
                    "max_tokens": max_tokens,
                    "max_retries": max_retries
                },
                timeout=900  # 15 minutes timeout
            )
            
            end_time = time.time()
            total_benchmark_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract key metrics
                metrics = {
                    "model": model_name,
                    "timestamp": datetime.now().isoformat(),
                    "total_benchmark_time_seconds": total_benchmark_time,
                    
                    # Speed metrics
                    "avg_tokens_per_second": result['averages']['avg_tokens_per_second'],
                    "avg_time_to_first_token_ms": result['averages']['avg_time_to_first_token_ms'],
                    "avg_total_response_time_seconds": result['averages']['avg_total_response_time_seconds'],
                    "avg_response_length_chars": result['averages']['avg_response_length_chars'],
                    
                    # Validation metrics
                    "total_tests": result['validation_statistics']['total_tests'],
                    "passed_first_try": result['validation_statistics']['passed_first_try'],
                    "passed_after_retry": result['validation_statistics']['passed_after_retry'],
                    "failed_all_retries": result['validation_statistics']['failed_all_retries'],
                    "total_retries_used": result['validation_statistics']['total_retries_used'],
                    "json_compliance_rate": result['summary']['json_compliance_rate'],
                    
                    # Success rate
                    "successful_tests": result['summary']['successful_tests'],
                    "failed_tests": result['summary']['failed_tests'],
                    "success_rate": (result['summary']['successful_tests'] / result['summary']['total_tests_run'] * 100) if result['summary']['total_tests_run'] > 0 else 0,
                    
                    # Individual test results
                    "individual_results": result['individual_results']
                }
                
                # Calculate average memory and CPU from individual tests
                cpu_values = []
                memory_values = []
                for test_name, test_result in result['individual_results'].items():
                    if 'metrics' in test_result:
                        cpu_values.append(test_result['metrics']['cpu_percent'])
                        memory_values.append(test_result['metrics']['memory_percent'])
                
                metrics['avg_cpu_percent'] = sum(cpu_values) / len(cpu_values) if cpu_values else 0
                metrics['avg_memory_percent'] = sum(memory_values) / len(memory_values) if memory_values else 0
                
                print(f"✅ Benchmark completed successfully!")
                print(f"   Speed: {metrics['avg_tokens_per_second']:.2f} tokens/sec")
                print(f"   First Token: {metrics['avg_time_to_first_token_ms']:.2f} ms")
                print(f"   Success Rate: {metrics['success_rate']:.1f}%")
                print(f"   CPU: {metrics['avg_cpu_percent']:.1f}%")
                print(f"   Memory: {metrics['avg_memory_percent']:.1f}%")
                
                return metrics
            else:
                print(f"❌ Benchmark failed: HTTP {response.status_code}")
                print(f"   Error: {response.text[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"❌ Benchmark timed out after 15 minutes")
            return None
        except Exception as e:
            print(f"❌ Error benchmarking {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compare_models(self, models: List[str]) -> List[Dict]:
        """Run benchmarks on multiple models and compare"""
        print("\n" + "="*70)
        print("📊 MULTI-MODEL COMPARISON STUDY")
        print("="*70)
        print(f"Models to test: {', '.join(models)}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = []
        
        for i, model in enumerate(models, 1):
            print(f"\n[{i}/{len(models)}] Processing {model}...")
            
            result = self.benchmark_model(model)
            if result:
                results.append(result)
            else:
                print(f"⚠️  Skipping {model} due to errors")
            
            # Small delay between models
            if i < len(models):
                print("\n⏳ Waiting 5 seconds before next model...")
                time.sleep(5)
        
        self.results = results
        return results
    
    def print_comparison_table(self):
        """Print a formatted comparison table"""
        if not self.results:
            print("❌ No results to display")
            return
        
        print("\n" + "="*100)
        print("📈 DETAILED COMPARISON RESULTS")
        print("="*100)
        
        # Key metrics table
        print(f"\n{'Model':<20} {'Speed (t/s)':<15} {'First Token (ms)':<18} {'Total Time (s)':<18} {'Success Rate':<15}")
        print("-"*100)
        
        for r in self.results:
            print(f"{r['model']:<20} {r['avg_tokens_per_second']:<15.2f} "
                  f"{r['avg_time_to_first_token_ms']:<18.2f} "
                  f"{r['avg_total_response_time_seconds']:<18.2f} "
                  f"{r['success_rate']:<15.1f}%")
        
        print("\n" + "-"*100)
        
        # Resource usage table
        print(f"\n{'Model':<20} {'Avg CPU %':<15} {'Avg Memory %':<18} {'Response Length':<18} {'JSON Compliance':<15}")
        print("-"*100)
        
        for r in self.results:
            print(f"{r['model']:<20} {r['avg_cpu_percent']:<15.1f} "
                  f"{r['avg_memory_percent']:<18.1f} "
                  f"{r['avg_response_length_chars']:<18.0f} "
                  f"{r['json_compliance_rate']:<15.1f}%")
        
        print("="*100)
        
        # Awards
        if len(self.results) > 1:
            print("\n🏆 PERFORMANCE AWARDS")
            print("-"*100)
            
            fastest_speed = max(self.results, key=lambda x: x['avg_tokens_per_second'])
            fastest_first_token = min(self.results, key=lambda x: x['avg_time_to_first_token_ms'])
            fastest_total = min(self.results, key=lambda x: x['avg_total_response_time_seconds'])
            best_success = max(self.results, key=lambda x: x['success_rate'])
            best_compliance = max(self.results, key=lambda x: x['json_compliance_rate'])
            lowest_cpu = min(self.results, key=lambda x: x['avg_cpu_percent'])
            lowest_memory = min(self.results, key=lambda x: x['avg_memory_percent'])
            
            print(f"⚡ Fastest Generation Speed: {fastest_speed['model']}")
            print(f"   → {fastest_speed['avg_tokens_per_second']:.2f} tokens/second")
            
            print(f"\n🚀 Lowest First Token Latency: {fastest_first_token['model']}")
            print(f"   → {fastest_first_token['avg_time_to_first_token_ms']:.2f} ms")
            
            print(f"\n⏱️  Fastest Total Response Time: {fastest_total['model']}")
            print(f"   → {fastest_total['avg_total_response_time_seconds']:.2f} seconds")
            
            print(f"\n✅ Best Success Rate: {best_success['model']}")
            print(f"   → {best_success['success_rate']:.1f}%")
            
            print(f"\n📋 Best JSON Compliance: {best_compliance['model']}")
            print(f"   → {best_compliance['json_compliance_rate']:.1f}%")
            
            print(f"\n💻 Lowest CPU Usage: {lowest_cpu['model']}")
            print(f"   → {lowest_cpu['avg_cpu_percent']:.1f}%")
            
            print(f"\n🧠 Lowest Memory Usage: {lowest_memory['model']}")
            print(f"   → {lowest_memory['avg_memory_percent']:.1f}%")
            
            print("="*100)
    
    def save_results(self, output_dir: str = "results"):
        """Save results to JSON and CSV files"""
        if not self.results:
            print("❌ No results to save")
            return
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON
        json_file = output_path / f"model_comparison_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Detailed results saved to: {json_file}")
        
        # Save summary CSV
        summary_data = []
        for r in self.results:
            summary_data.append({
                'Model': r['model'],
                'Tokens/Second': r['avg_tokens_per_second'],
                'First Token (ms)': r['avg_time_to_first_token_ms'],
                'Total Time (s)': r['avg_total_response_time_seconds'],
                'Response Length': r['avg_response_length_chars'],
                'Success Rate (%)': r['success_rate'],
                'JSON Compliance (%)': r['json_compliance_rate'],
                'CPU (%)': r['avg_cpu_percent'],
                'Memory (%)': r['avg_memory_percent'],
                'Tests Passed': r['successful_tests'],
                'Tests Failed': r['failed_tests'],
                'Retries Used': r['total_retries_used']
            })
        
        df = pd.DataFrame(summary_data)
        csv_file = output_path / f"model_comparison_summary_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"📊 Summary CSV saved to: {csv_file}")
        
        # Save detailed test results for each model
        for result in self.results:
            model_name = result['model'].replace(':', '_').replace('/', '_')
            detail_file = output_path / f"detailed_{model_name}_{timestamp}.json"
            with open(detail_file, 'w') as f:
                json.dump(result['individual_results'], f, indent=2)
        
        print(f"📁 All results saved in: {output_path.absolute()}")
        
        return {
            'json_file': str(json_file),
            'csv_file': str(csv_file),
            'output_dir': str(output_path.absolute())
        }


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("🎯 MODEL COMPARISON STUDY")
    print("="*70)
    print("Comparing: llama3.2:3b, phi4, mistral:7b")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Initialize benchmark
    benchmark = ModelBenchmark()
    
    # Check if server is running
    if not benchmark.check_server():
        print("\n❌ ERROR: Benchmark server is not running!")
        print("\nPlease start the server first:")
        print("  python benchmark_app.py")
        return
    
    # Check available models
    print("\n🔍 Checking available models...")
    available = benchmark.list_available_models()
    print(f"Found {len(available)} models in Ollama")
    
    # Verify models are available
    missing_models = []
    for model in MODELS_TO_TEST:
        if model not in available:
            missing_models.append(model)
    
    if missing_models:
        print(f"\n⚠️  WARNING: The following models are not available:")
        for model in missing_models:
            print(f"   - {model}")
        print("\nYou may need to pull them first:")
        for model in missing_models:
            print(f"   ollama pull {model}")
        
        response = input("\nContinue with available models only? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
        
        models_to_test = [m for m in MODELS_TO_TEST if m not in missing_models]
    else:
        models_to_test = MODELS_TO_TEST
        print("✅ All models are available")
    
    if not models_to_test:
        print("❌ No models to test. Exiting.")
        return
    
    # Run comparison
    print(f"\n🚀 Starting benchmark of {len(models_to_test)} models...")
    print("⏳ This may take 10-30 minutes depending on your hardware...")
    
    results = benchmark.compare_models(models_to_test)
    
    if results:
        # Print comparison
        benchmark.print_comparison_table()
        
        # Save results
        files = benchmark.save_results()
        
        print("\n" + "="*70)
        print("✅ BENCHMARK STUDY COMPLETE!")
        print("="*70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Models tested: {len(results)}")
        print(f"Results saved to: {files['output_dir']}")
        print("="*70)
    else:
        print("\n❌ No successful benchmarks completed")


if __name__ == "__main__":
    main()