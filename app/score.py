import os
import time
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from prometheus_client import Counter, Histogram, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(BASE_DIR, "tokenizer"))
model = GPT2LMHeadModel.from_pretrained(os.path.join(BASE_DIR, "model"))
model.eval()

REQUESTS_TOTAL = Counter(
    "gpt2_requests_total",
    "Total number of GPT-2 requests",
    ["status"]
)
REQUEST_LATENCY = Histogram(
    "gpt2_request_latency_seconds",
    "Latency of GPT-2 inference requests"
)
PROMPT_LENGTH = Histogram(
    "gpt2_prompt_length_chars",
    "Prompt length in characters"
)
RESPONSE_LENGTH = Histogram(
    "gpt2_response_length_chars",
    "Generated response length in characters"
)


@app.route("/joke", methods=["POST"])
def joke():
    start = time.time()
    data = request.get_json() or {}
    user_input = data.get("prompt", "").strip()

    if not user_input:
        REQUESTS_TOTAL.labels(status="error").inc()
        return jsonify({"error": "prompt is required"}), 400

    PROMPT_LENGTH.observe(len(user_input))

    try:
        full_prompt = f"<|startoftext|> Joke setup: {user_input}\nPunchline:"
        inputs = tokenizer(full_prompt, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            max_length=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = result.replace("<|startoftext|>", "").strip()

        if "Punchline:" in result:
            result = result.split("Punchline:", 1)[1].strip()

        # Strip prompt echo if model repeats the input back
        if result.lower().startswith(user_input.lower()):
            result = result[len(user_input):].strip(" :-")

        RESPONSE_LENGTH.observe(len(result))
        REQUESTS_TOTAL.labels(status="success").inc()

        return jsonify({"joke": result})

    except Exception:
        REQUESTS_TOTAL.labels(status="error").inc()
        raise

    finally:
        REQUEST_LATENCY.observe(time.time() - start)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    "/metrics": make_wsgi_app()
})