"""
Exécuter sur le serveur Onyxia pour entraîner et sauvegarder les poids.
Usage: python3 save_weights.py
"""

import sys
import os

sys.path.insert(0, "/home/onyxia/work")
os.chdir("/home/onyxia/work")

import torch
from p5.corpus import load_corpus
from p5.tokenizer import BPETokenizer
from p5.causal_llm import CausalLLM
from p5.causal_train import train
from p5.ner import NERModel, auto_label, train_ner

VOCAB_SIZE = 300  # LLM causal
NER_VOCAB_SIZE = 1000  # NER: vocab mayor captura Rabbit, Queen, Hatter, Caterpillar...
CONTEXT_SIZE = 128
NER_CONTEXT = 64

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# --- Corpus + tokenizer ---
text = load_corpus("alicia")
tokenizer = BPETokenizer(text, vocab_size=VOCAB_SIZE)
tokens = tokenizer.encode(text)
print(f"Vocab: {tokenizer.vocab_size} | Tokens: {len(tokens):,}")


def save_tok(tokenizer):
    return {
        "vocab": tokenizer.vocab,
        "tok2id": tokenizer.tok2id,
        "merges": tokenizer.merges,
    }


# --- Entraîner LLM ---
model = CausalLLM(
    vocab_size=VOCAB_SIZE,
    max_seq_len=CONTEXT_SIZE,
    d_model=128,
    n_heads=4,
    n_layers=4,
    expansion=4,
    dropout=0.1,
).to(device)

train(model, tokens, epochs=5, context_size=CONTEXT_SIZE, batch_size=64)

torch.save({"model_state": model.state_dict(), **save_tok(tokenizer)}, "p5_causal_2609.pth")
print("✓ Saved p5_causal_2609.pth")

# --- Entraîner NER (vocab=1000 pour capturer plus de noms propres) ---
# Con vocab=300 solo emerge "Alice" como token capitalizado multichar.
# Con vocab=1000 emergen: Rabbit, Queen, Hatter, Caterpillar, Duchess, Gryphon...
# El NER se entrena con su propio tokenizador (sin transfer learning, ya que
# los tamaños de embedding difieren del LLM).
ner_tokenizer = BPETokenizer(text, vocab_size=NER_VOCAB_SIZE)
cap_tokens = [v for v in ner_tokenizer.vocab if v and v[0].isupper() and v.isalpha() and len(v) > 1]
print(f"NER vocab={NER_VOCAB_SIZE} | Tokens capitalizados multichar: {len(cap_tokens)}")

ner_text = text[:60_000]
ner_tokens = ner_tokenizer.encode(ner_text)
ner_labels = auto_label(ner_tokens, ner_tokenizer.vocab)
n_ent = sum(1 for l in ner_labels if l > 0)
print(
    f"NER: {len(ner_tokens):,} tokens, {n_ent:,} entidades ({100 * n_ent / len(ner_tokens):.1f}%)"
)

ner_model = NERModel(
    vocab_size=NER_VOCAB_SIZE,
    max_seq_len=NER_CONTEXT,
    d_model=128,
    n_heads=4,
    n_layers=4,
    expansion=4,
    dropout=0.1,
).to(device)

train_ner(ner_model, ner_tokens, ner_labels, epochs=5, context_size=NER_CONTEXT)

torch.save({"model_state": ner_model.state_dict(), **save_tok(ner_tokenizer)}, "p5_ner_2609.pth")
print("✓ Saved p5_ner_2609.pth")

# --- Upload S3 ---
print("\nUpload vers S3...")
import subprocess

bucket = "s3://your-bucket"  # sera remplacé dynamiquement

# Détecter le bucket S3 Onyxia automatiquement
try:
    result = subprocess.run(
        ["mc", "ls", "s3/"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    print("Buckets disponibles:", result.stdout)
except Exception as e:
    print(f"mc non disponible: {e}")
    print("Upload manuel:")
    print("  mc cp p5_causal_2609.pth s3/<votre-bucket>/")
    print("  mc cp p5_ner_2609.pth s3/<votre-bucket>/")

print("\nDone!")
