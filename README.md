# 🚀 Local SLM Benchmark

A rigorous benchmarking framework for measuring the real-world performance of Small Language Models (SLMs) running locally with OLLAMA. This project provides comprehensive metrics on **Tokens Per Second (TPS)**, **Time to First Token (TTFT)**, and **Total Response Latency** to help you make informed decisions about local AI deployments.

## 📊 What This Project Measures

Unlike synthetic benchmarks, this framework measures what actually matters in production:

- **Tokens Per Second (TPS)**: Raw generation speed - how fast the model produces text
- **Time to First Token (TTFT)**: Perceived responsiveness - how long users wait before seeing output
- **Total Response Latency**: End-to-end user experience - complete request to response time
- **System Resource Usage**: CPU and memory utilization during inference

### ⚡ Quick Performance Overview

Tested on consumer hardware with ~500 character responses:

| Model | Speed (t/s) | First Token (ms) | Total Time (s) | Best For |
|-------|-------------|------------------|----------------|----------|
| **phi3:mini** | 21.71 | 641 ⚡ | 6.0 | Interactive chat, real-time apps |
| **llama3.2:latest** | 19.36 | 977 | 5.4 🏆 | Quick queries, balanced performance |
| **mistral:7b** | 10.31 | 2117 | 11.6 | Batch processing, quality over speed |

🏆 = Fastest total time | ⚡ = Best responsiveness (TTFT)

## ✨ Features

- **FastAPI-based benchmarking server** with streaming support for accurate TTFT measurement
- **Multiple test scenarios** covering different use cases (short queries, medium responses, long generation, reasoning, coding)
- **Automated model comparison** tool for side-by-side performance analysis
- **High-precision timing** using `time.perf_counter()` for nanosecond-level accuracy
- **System resource monitoring** to track CPU and memory impact
- **JSON result export** for further analysis and visualization

## 🏗️ Architecture

```
┌─────────────────────┐
│   main.py           │  ← Comparison tool & CLI interface
│   (Client)          │
└──────────┬──────────┘
           │ HTTP requests
           ▼
┌─────────────────────┐
│  benchmark_app.py   │  ← FastAPI server with performance instrumentation
│  (Server)           │
└──────────┬──────────┘
           │ Python API
           ▼
┌─────────────────────┐
│   OLLAMA Runtime    │  ← Local model inference engine
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   SLM Models        │
│  • phi3:mini        │
│  • mistral:7b       │
│  • llama3.2         │
│  • [your models]    │
└─────────────────────┘
```

## 🎯 Prerequisites

### Required Software

- **Python 3.8+**
- **OLLAMA** - [Installation guide](https://ollama.ai)
- **One or more LLM models** pulled via OLLAMA

### Python Dependencies

```bash
pip install fastapi uvicorn ollama psutil requests
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Start OLLAMA

Make sure OLLAMA is running:

```bash
ollama serve
```

### 2. Pull Models to Test

Pull the models you want to benchmark:

```bash
ollama pull phi3:mini
ollama pull mistral:7b
ollama pull llama3.2
```

Verify models are available:

```bash
ollama list
```

### 3. Start the Benchmark Server

```bash
python benchmark_app.py
```

The server will start at `http://localhost:8000`. You should see:

```
====================================================
Starting Ollama Benchmark Server
====================================================
Make sure Ollama is running (ollama serve)
Server will start at http://localhost:8000
====================================================
```

### 4. Run Model Comparison

In a new terminal, run the comparison tool:

**Interactive Mode** (recommended for first-time use):
```bash
python main.py
```

You'll be prompted to select which models to compare:

```
Available models:
  1. phi3:mini (2.3 GB)
  2. mistral:7b (4.1 GB)
  3. llama3.2 (2.0 GB)

Enter model numbers to compare (comma-separated, e.g., 1,2,3):
Selection: 1,2,3
```

**Command Line Mode** (for automation):
```bash
python main.py phi3:mini mistral:7b llama3.2
```

**Debug Mode** (troubleshooting):
```bash
python main.py --debug
```

## 📈 Understanding the Results

### Sample Output

```
====================================================================================================
📈 COMPARISON RESULTS
====================================================================================================
Model                     Speed (t/s)     First Token (ms)     Total Time (ms)      Response Length
----------------------------------------------------------------------------------------------------
mistral:7b                10.31           2116.76              11594.25             479            
llama3.2:latest           19.36           976.64               5402.23              490            
phi3:mini                 21.71           641.59               6023.84              533            
====================================================================================================

🏆 AWARDS:
  ⚡ Fastest Generation: phi3:mini (21.71 tokens/sec)
  🚀 Lowest First Token Latency: phi3:mini (641.59ms)
  ⏱️  Fastest Total Response: llama3.2:latest (5402.23ms)

💾 Results saved to: results/model_comparison_3_models.json
```

### Key Findings from Real-World Testing

**Performance Rankings:**

1. **phi3:mini** - Best overall balance
   - ⚡ Fastest token generation (21.71 t/s - 2.1x faster than mistral)
   - 🚀 Lowest first token latency (641ms)
   - 📝 Longest responses (533 chars avg)
   
2. **llama3.2:latest** - Best total latency
   - ⏱️ Fastest complete responses (5.4s)
   - Good token speed (19.36 t/s)
   - Balanced first token time (976ms)
   
3. **mistral:7b** - Quality over speed
   - Slowest but potentially higher quality
   - 2.1s first token delay (noticeable)
   - 11.6s total time (use for batch/async tasks)

**Practical Implications:**
- For **interactive chat**: phi3:mini wins (fast + responsive)
- For **quick responses**: llama3.2:latest (best total time)
- For **content quality**: mistral:7b (when speed isn't critical)

### Interpreting the Metrics

#### Tokens Per Second (TPS)
Based on real-world testing:
- **> 20 TPS** (phi3:mini - 21.71): Excellent - smooth, readable generation
- **15-20 TPS** (llama3.2 - 19.36): Good - acceptable for most interactive use
- **< 15 TPS** (mistral:7b - 10.31): Slow - better for async/batch processing

#### Time to First Token (TTFT)
Measured performance:
- **< 700ms** (phi3:mini - 641ms): Excellent - feels instant
- **700-1000ms** (llama3.2 - 976ms): Good - slight delay but acceptable
- **> 2000ms** (mistral:7b - 2116ms): Poor - noticeable wait, users will notice

#### Total Response Latency
For ~500 character responses:
- **5-6s** (phi3:mini/llama3.2): Good for interactive chat
- **11-12s** (mistral:7b): Too slow for chat, better for content generation

**Real-World Recommendation:**
- **Interactive applications**: Use phi3:mini (best TTFT + TPS combo)
- **Batch processing**: mistral:7b acceptable (quality may justify wait)
- **Balanced use**: llama3.2:latest (best total time)

## 🧪 Test Scenarios

The benchmark runs five different test scenarios to cover various use cases:

| Scenario | Example Prompt | Purpose |
|----------|---------------|---------|
| **short** | "What is 2+2?" | Quick factual queries |
| **medium** | "Explain machine learning in 2 paragraphs." | Typical assistant interactions |
| **long** | "Write a story about a robot learning to paint." | Content generation tasks |
| **reasoning** | "If a bat and ball cost $1.10..." | Logical reasoning capability |
| **coding** | "Write a Python function to check if..." | Code generation performance |

Each scenario measures all three core metrics to provide a complete performance profile.

## 📁 Project Structure

```
local-slm-benchmark/
├── benchmark_app.py      # FastAPI server with benchmarking logic
├── main.py               # Model comparison CLI tool
├── requirements.txt      # Python dependencies
├── results/              # Generated comparison results (JSON)
│   └── model_comparison_*.json
├── README.md            # This file
└── .gitignore           # Git ignore patterns
```

## 🔧 API Endpoints

### GET `/`
Health check and API information

### GET `/models`
List all available OLLAMA models

**Response:**
```json
[
  {
    "name": "phi3:mini",
    "size": "2.3 GB",
    "modified_at": "2024-01-15T10:30:00Z"
  }
]
```

### POST `/benchmark`
Run a single benchmark test

**Request:**
```json
{
  "model": "phi3:mini",
  "prompt": "Explain quantum computing",
  "max_tokens": 512
}
```

**Response:**
```json
{
  "model": "phi3:mini",
  "prompt": "Explain quantum computing",
  "response": "Quantum computing is...",
  "time_to_first_token_ms": 185.3,
  "total_time_seconds": 2.08,
  "tokens_per_second": 45.3,
  "cpu_percent": 78.5,
  "memory_percent": 45.2,
  "response_length": 287
}
```

### POST `/benchmark/all-tests`
Run all five test scenarios on a specific model

**Parameters:**
- `model`: Model name (e.g., "phi3:mini")
- `max_tokens`: Maximum tokens to generate (default: 100)

**Response:**
```json
{
  "model": "phi3:mini",
  "individual_results": {
    "short": { /* benchmark result */ },
    "medium": { /* benchmark result */ },
    "long": { /* benchmark result */ },
    "reasoning": { /* benchmark result */ },
    "coding": { /* benchmark result */ }
  },
  "averages": {
    "avg_tokens_per_second": 45.3,
    "avg_time_to_first_token_ms": 185.2,
    "avg_total_response_time_ms": 2080.5,
    "avg_response_length_chars": 287
  },
  "summary": {
    "total_tests_run": 5,
    "failed_tests": 0
  }
}
```

## 🎯 Use Cases

### Choosing the Right Model

Based on actual benchmark results:

**For Interactive Chat Applications:**
- ✅ **Recommended: phi3:mini**
  - Best TTFT (641ms - users barely notice delay)
  - Fast generation (21.71 t/s - smooth streaming)
  - Total time: 6s for ~500 char responses
  
**For Quick One-Off Queries:**
- ✅ **Recommended: llama3.2:latest**
  - Best total latency (5.4s complete response)
  - Decent TTFT (976ms - acceptable)
  - Good speed (19.36 t/s)

**For Content Quality (when speed isn't critical):**
- ⚠️ **Consider: mistral:7b**
  - Slowest but potentially higher quality outputs
  - 2.1s first token delay (not suitable for real-time chat)
  - 11.6s total time (use async/background processing)
  - Better for: email drafts, reports, creative writing

**Real-World Performance Data:**
| Use Case | Best Model | Why |
|----------|-----------|-----|
| Live chat support | phi3:mini | 641ms TTFT feels instant |
| Code completion | phi3:mini | 21.71 t/s enables real-time suggestions |
| Search augmentation | llama3.2 | 5.4s total time for quick results |
| Content generation | mistral:7b | Slower but may produce better quality |
| Resource-constrained | phi3:mini | Smallest, fastest, good quality balance |

### Capacity Planning

Use actual TPS measurements to calculate concurrent user capacity:

```
Concurrent users = (Model TPS) / (Target TPS per user)
```

**Real-world examples from our benchmarks:**

**phi3:mini (21.71 TPS):**
- 2 users @ 10.8 t/s each = excellent UX
- 4 users @ 5.4 t/s each = good UX
- 8 users @ 2.7 t/s each = acceptable but slower
- 10+ users = need horizontal scaling

**llama3.2:latest (19.36 TPS):**
- 2 users @ 9.7 t/s each = excellent UX
- 4 users @ 4.8 t/s each = acceptable UX
- 6+ users = performance degrades noticeably

**mistral:7b (10.31 TPS):**
- 1 user @ 10.3 t/s = good for async tasks
- 2+ concurrent users = not recommended for real-time
- Better for queued/batch processing

**Scaling Thresholds:**
- If you need to serve >5 concurrent users in real-time, consider:
  - Load balancing across multiple instances
  - Using faster models (phi3:mini or llama3.2)
  - Implementing request queuing
  - GPU acceleration (if available)

## 🔬 Advanced Usage

### Custom Test Prompts

Modify `TEST_PROMPTS` in `benchmark_app.py`:

```python
TEST_PROMPTS = {
    "custom_test": "Your custom prompt here",
    # ... other tests
}
```

### Programmatic API Usage

Use the benchmark server in your own code:

```python
import requests

response = requests.post(
    "http://localhost:8000/benchmark",
    json={
        "model": "phi3:mini",
        "prompt": "Your prompt",
        "max_tokens": 200
    }
)

metrics = response.json()
print(f"TPS: {metrics['tokens_per_second']}")
print(f"TTFT: {metrics['time_to_first_token_ms']}ms")
```

### Batch Comparison Script

```bash
# Compare all installed models automatically
for model in $(ollama list | tail -n +2 | awk '{print $1}'); do
    python main.py "$model"
done
```

## 📊 Exporting Results

Results are automatically saved to `results/model_comparison_*.json` with the following structure:

```json
[
  {
    "model": "phi3:mini",
    "speed": 21.71,
    "first_token_ms": 641.59,
    "total_time_ms": 6023.84,
    "response_length": 533
  },
  {
    "model": "llama3.2:latest",
    "speed": 19.36,
    "first_token_ms": 976.64,
    "total_time_ms": 5402.23,
    "response_length": 490
  }
]
```

Use these for:
- Creating visualizations (matplotlib, seaborn)
- Statistical analysis (pandas, numpy)
- Tracking performance over time
- Documentation and blog posts

## 💡 Real-World Insights

Based on comprehensive testing, here are key insights:

### The TTFT Surprise
**Finding:** First token latency varies dramatically (641ms to 2116ms)
- phi3:mini's 641ms TTFT makes it feel 3.3x more responsive than mistral:7b
- Users perceive sub-700ms as "instant", above 2s as "slow"
- **Implication:** For chat apps, TTFT matters more than raw TPS

### The Speed-Size Paradox
**Finding:** Smaller models aren't always faster
- phi3:mini (smallest) → 21.71 t/s
- llama3.2:latest (medium) → 19.36 t/s
- mistral:7b (largest) → 10.31 t/s
- **Implication:** Optimization matters more than parameter count alone

### The Total Time Winner
**Finding:** llama3.2 wins on total latency despite mid-range TPS
- Balanced TTFT (976ms) + decent speed (19.36 t/s) = best 5.4s total time
- **Implication:** For one-shot queries, balanced performance beats extreme optimization in one metric

### Production Recommendations
1. **Start with phi3:mini** - best default choice for most use cases
2. **Use llama3.2** if you need absolute minimum latency per query
3. **Reserve mistral:7b** for batch processing or when quality > speed
4. **Always measure** - your hardware/workload may produce different results

## 🐛 Troubleshooting

### "No models found or server not running"

**Solutions:**
1. Ensure OLLAMA is running: `ollama serve`
2. Check models are pulled: `ollama list`
3. Verify server is accessible: `curl http://localhost:8000`

### "Invalid model name" or "BenchmarkRequest" error

**Cause:** Model name parsing issue

**Solution:**
- Use exact model names from `ollama list`
- Ensure proper format: `model:tag` (e.g., `phi3:mini`, not just `phi3`)

### High variance in results

**Causes:**
- System under load (background processes)
- Thermal throttling
- First run (cold start) vs subsequent runs

**Solutions:**
- Close unnecessary applications
- Run multiple iterations and use median values
- Use dedicated hardware for consistent results

### Connection timeout

**Solution:**
- Increase timeout in `main.py`: `timeout=600` → `timeout=1200`
- Use shorter max_tokens for testing
- Check OLLAMA memory usage (may need restart)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional test scenarios
- Visualization tools
- Statistical analysis features
- Support for batch inference benchmarking
- Multi-threaded performance testing
- Quality metrics (accuracy, coherence)

## 📝 License

MIT License - see LICENSE file for details

## 🔗 Related Projects

- **OLLAMA**: [github.com/ollama/ollama](https://github.com/ollama/ollama)
- **llama.cpp**: [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- **FastAPI**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)

## 📧 Contact

For questions, issues, or contributions, please open an issue on GitHub.

## 🙏 Acknowledgments

This project was built to provide practical, real-world performance data for local LLM deployment decisions. Special thanks to the OLLAMA team for making local LLM inference accessible.

---

**Built with ❤️ for the local AI community**

*Last updated: March 2026*