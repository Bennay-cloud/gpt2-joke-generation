import os
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(BASE_DIR, "tokenizer"))
model = GPT2LMHeadModel.from_pretrained(os.path.join(BASE_DIR, "model"))
model.eval()


@app.route("/joke", methods=["POST"])
def joke():
    data = request.get_json() or {}
    user_input = data.get("prompt", "").strip()

    if not user_input:
        return jsonify({"error": "prompt is required"}), 400

    # Better structured prompt for joke generation
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

    # Clean up unwanted special tokens or prompt echoes
    result = result.replace("<|startoftext|>", "").strip()

    if "Punchline:" in result:
        result = result.split("Punchline:", 1)[1].strip()

    if result.lower().startswith(user_input.lower()):
        result = result[len(user_input):].strip(" :-")

    return jsonify({"joke": result})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})