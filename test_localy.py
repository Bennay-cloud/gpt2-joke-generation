from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_PATH = "/Users/mustaphabennay/OrbStack/ubuntu/home/mustaphabennay/GPT2_AllTrans/app/model"
TOKENIZER_PATH = "/Users/mustaphabennay/OrbStack/ubuntu/home/mustaphabennay/GPT2_AllTrans/app/tokenizer"

tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

def generate_joke(prompt: str, max_length: int = 100) -> str:
    full_prompt = f"<|startoftext|> {prompt}"
    inputs = tokenizer(full_prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generate_joke("Why did the programmer"))
print(generate_joke("A chicken walks into a bar"))