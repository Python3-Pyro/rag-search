from openai import OpenAI
import os
from dotenv import load_dotenv
import tiktoken

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

text = "How to bake a cake?"

# ---- Step 1: Tokenization ----
enc = tiktoken.encoding_for_model("text-embedding-3-small")
tokens = enc.encode(text)

print("\nInput Text:")
print(text)

print("\nTokenizer Output (Token IDs):")
print(tokens)

decoded_tokens = [enc.decode([t]) for t in tokens]
print("\nDecoded Tokens:")
print(decoded_tokens)

# ---- Step 2: Generate Embeddings ----
res = client.embeddings.create(
    model="text-embedding-3-large",
    input=text
)

embedding_vector = res.data[0].embedding

print("\nGenerated Embedding Vector (first 10 values shown):")
print(embedding_vector[:10])

print(f"\nFull vector length: {len(embedding_vector)} dimensions\n")