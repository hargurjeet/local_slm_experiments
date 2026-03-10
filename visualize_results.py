#!/usr/bin/env python3
"""
Visualization script for model comparison results
Creates charts and graphs from benchmark data
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_results(json_file: str) -> list:
    """Load results from JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)

def create_visualizations(results: list, output_dir: str = "results/charts"):
    """Create comprehensive visualizations"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    models = [r['model'] for r in results]
    
    # 1. Speed Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Tokens per second
    tps = [r['avg_tokens_per_second'] for r in results]
    axes[0, 0].bar(models, tps, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 0].set_title('Generation Speed (Tokens/Second)', fontweight='bold')
    axes[0, 0].set_ylabel('Tokens/Second')
    axes[0, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(tps):
        axes[0, 0].text(i, v + 1, f'{v:.2f}', ha='center', va='bottom')
    
    # First token latency
    first_token = [r['avg_time_to_first_token_ms'] for r in results]
    axes[0, 1].bar(models, first_token, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 1].set_title('First Token Latency (Lower is Better)', fontweight='bold')
    axes[0, 1].set_ylabel('Milliseconds')
    axes[0, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(first_token):
        axes[0, 1].text(i, v + 5, f'{v:.2f}', ha='center', va='bottom')
    
    # Total response time
    total_time = [r['avg_total_response_time_seconds'] for r in results]
    axes[1, 0].bar(models, total_time, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1, 0].set_title('Total Response Time (Lower is Better)', fontweight='bold')
    axes[1, 0].set_ylabel('Seconds')
    axes[1, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(total_time):
        axes[1, 0].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')
    
    # Success rate
    success_rate = [r['success_rate'] for r in results]
    axes[1, 1].bar(models, success_rate, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1, 1].set_title('Success Rate', fontweight='bold')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].set_ylim([0, 105])
    axes[1, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(success_rate):
        axes[1, 1].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path / 'performance_comparison.png'}")
    plt.close()
    
    # 2. Resource Usage
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Resource Usage Comparison', fontsize=16, fontweight='bold')
    
    # CPU usage
    cpu = [r['avg_cpu_percent'] for r in results]
    axes[0].bar(models, cpu, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0].set_title('Average CPU Usage', fontweight='bold')
    axes[0].set_ylabel('CPU %')
    axes[0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(cpu):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
    
    # Memory usage
    memory = [r['avg_memory_percent'] for r in results]
    axes[1].bar(models, memory, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1].set_title('Average Memory Usage', fontweight='bold')
    axes[1].set_ylabel('Memory %')
    axes[1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(memory):
        axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path / 'resource_usage.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path / 'resource_usage.png'}")
    plt.close()
    
    # 3. Quality Metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Quality Metrics Comparison', fontsize=16, fontweight='bold')
    
    # JSON Compliance
    compliance = [r['json_compliance_rate'] for r in results]
    axes[0].bar(models, compliance, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0].set_title('JSON Compliance Rate', fontweight='bold')
    axes[0].set_ylabel('Percentage (%)')
    axes[0].set_ylim([0, 105])
    axes[0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(compliance):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
    
    # Retry count
    retries = [r['total_retries_used'] for r in results]
    axes[1].bar(models, retries, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1].set_title('Total Retries Used (Lower is Better)', fontweight='bold')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(retries):
        axes[1].text(i, v + 0.1, f'{v}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path / 'quality_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path / 'quality_metrics.png'}")
    plt.close()
    
    # 4. Radar Chart for Overall Comparison
    from math import pi
    
    # Normalize metrics to 0-100 scale
    categories = ['Speed', 'Latency\n(inv)', 'Success\nRate', 'JSON\nCompliance', 'Efficiency\n(inv)']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = [n / len(categories) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, result in enumerate(results):
        # Normalize values to 0-100 scale
        max_tps = max(r['avg_tokens_per_second'] for r in results)
        max_latency = max(r['avg_time_to_first_token_ms'] for r in results)
        max_cpu = max(r['avg_cpu_percent'] for r in results)
        
        values = [
            (result['avg_tokens_per_second'] / max_tps) * 100,  # Speed (higher is better)
            100 - (result['avg_time_to_first_token_ms'] / max_latency) * 100,  # Latency inverted
            result['success_rate'],  # Success rate
            result['json_compliance_rate'],  # JSON compliance
            100 - (result['avg_cpu_percent'] / max_cpu) * 100  # CPU inverted
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=result['model'], color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('Overall Model Comparison (Normalized)', fontweight='bold', size=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path / 'radar_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path / 'radar_comparison.png'}")
    plt.close()
    
    # 5. Create summary statistics
    summary_stats = []
    for result in results:
        summary_stats.append({
            'Model': result['model'],
            'Rank Speed': 0,
            'Rank Latency': 0,
            'Rank Success': 0,
            'Rank Efficiency': 0,
            'Overall Score': 0
        })
    
    # Calculate ranks
    sorted_by_speed = sorted(results, key=lambda x: x['avg_tokens_per_second'], reverse=True)
    sorted_by_latency = sorted(results, key=lambda x: x['avg_time_to_first_token_ms'])
    sorted_by_success = sorted(results, key=lambda x: x['success_rate'], reverse=True)
    sorted_by_cpu = sorted(results, key=lambda x: x['avg_cpu_percent'])
    
    for i, stat in enumerate(summary_stats):
        model_name = stat['Model']
        stat['Rank Speed'] = next(i for i, r in enumerate(sorted_by_speed, 1) if r['model'] == model_name)
        stat['Rank Latency'] = next(i for i, r in enumerate(sorted_by_latency, 1) if r['model'] == model_name)
        stat['Rank Success'] = next(i for i, r in enumerate(sorted_by_success, 1) if r['model'] == model_name)
        stat['Rank Efficiency'] = next(i for i, r in enumerate(sorted_by_cpu, 1) if r['model'] == model_name)
        stat['Overall Score'] = sum([stat['Rank Speed'], stat['Rank Latency'], stat['Rank Success'], stat['Rank Efficiency']])
    
    # Save summary
    summary_df = pd.DataFrame(summary_stats)
    summary_df = summary_df.sort_values('Overall Score')
    summary_file = output_path / 'ranking_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"✅ Saved: {summary_file}")
    
    print(f"\n📊 All visualizations saved to: {output_path.absolute()}")
    
    return output_path

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <results_json_file>")
        print("\nExample:")
        print("  python visualize_results.py results/model_comparison_20250310_143022.json")
        
        # Try to find latest results
        results_dir = Path("results")
        if results_dir.exists():
            json_files = list(results_dir.glob("model_comparison_*.json"))
            if json_files:
                latest = max(json_files, key=lambda p: p.stat().st_mtime)
                print(f"\n💡 Found latest results: {latest}")
                response = input("Use this file? (y/n): ")
                if response.lower() == 'y':
                    results_file = str(latest)
                else:
                    return
            else:
                print("\n❌ No results files found in results/ directory")
                return
        else:
            print("\n❌ No results directory found")
            return
    else:
        results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"❌ File not found: {results_file}")
        return
    
    print(f"\n📊 Loading results from: {results_file}")
    results = load_results(results_file)
    
    print(f"Found {len(results)} model results")
    for r in results:
        print(f"  - {r['model']}")
    
    print("\n🎨 Creating visualizations...")
    output_path = create_visualizations(results)
    
    print("\n✅ Done! Open the PNG files to view the charts.")
    print(f"📁 Location: {output_path.absolute()}")

if __name__ == "__main__":
    main()