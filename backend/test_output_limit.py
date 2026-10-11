"""The reply-length cap must be applied the way each provider takes it (no model or network needed).

Run from backend/:  python test_output_limit.py

Binding max_tokens (or num_predict) onto an Ollama model sends it straight to the client's generate(), which rejects both, so a local run died in the
content synthesizer on every attempt. Ollama takes the cap as a model field; Groq takes max_tokens per call.
"""
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM

from app.core.llm import with_output_limit

ollama = OllamaLLM(model="qwen3:14b", num_predict=4096)
capped = with_output_limit(ollama, 123)
assert isinstance(capped, OllamaLLM) and capped.num_predict == 123, "Ollama must carry the cap as num_predict"
assert ollama.num_predict == 4096, "the shared model must not be changed by one call's cap"

groq = ChatGroq(model="openai/gpt-oss-120b", api_key="test-key")
bound = with_output_limit(groq, 321)
assert bound.kwargs == {"max_tokens": 321}, "Groq must carry the cap as a bound max_tokens"

print("ALL PASS (3 checks)")
