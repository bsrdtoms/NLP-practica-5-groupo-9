"""
Exploración de hiperparámetros — TP5 PLN 2025/2026
Grupo G09 — Thomas Bossard

Experimentos:
  1. Efecto de d_model (64, 128, 256) sobre la loss y el número de parámetros
  2. Efecto del learning rate (1e-3, 3e-4, 1e-4)
  3. Efecto del número de capas (2, 4, 6)
  4. Análisis de la curva train/val (sobreajuste, underfitting)
  5. Análisis cualitativo de generación de texto

Corpus: primeros 40 000 caracteres de Alice (rápido en CPU).
Duración estimada: ~10-15 min en CPU.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from loguru import logger

from p5.causal_llm import CausalLLM
from p5.causal_train import _make_dataloaders, _run_epoch
from p5.corpus import load_corpus
from p5.tokenizer import BPETokenizer

VOCAB_SIZE = 300
CONTEXT_SIZE = 64  # más pequeño = más rápido
BATCH_SIZE = 32
EPOCHS = 3
CORPUS_CHARS = 40_000  # subconjunto pequeño para comparaciones rápidas
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Device: {DEVICE}")

# ── Corpus ────────────────────────────────────────────────────────────────────
alice_proper = Path(__file__).parent / "alicia" / "alice_proper.txt"
text = alice_proper.read_text(encoding="utf-8") if alice_proper.exists() else load_corpus("alicia")
text_short = text[:CORPUS_CHARS]
logger.info(f"Corpus: {len(text_short):,} chars")

tokenizer = BPETokenizer(text_short, vocab_size=VOCAB_SIZE)
tokens = tokenizer.encode(text_short)
logger.info(f"Tokens: {len(tokens):,}")


def run_config(d_model, n_heads, n_layers, lr, label):
    """Entrena un modelo con la config dada y devuelve las curvas de loss."""
    model = CausalLLM(
        vocab_size=VOCAB_SIZE,
        max_seq_len=CONTEXT_SIZE,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        expansion=4,
        dropout=0.1,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    train_dl, val_dl = _make_dataloaders(tokens, CONTEXT_SIZE, BATCH_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        train_loss = _run_epoch(model, train_dl, optimizer)
        val_loss = _run_epoch(model, val_dl, None)
        history.append(
            {"epoch": epoch + 1, "train": round(train_loss, 4), "val": round(val_loss, 4)}
        )
        logger.info(f"  [{label}] E{epoch + 1} train={train_loss:.4f} val={val_loss:.4f}")

    elapsed = time.time() - t0

    # Generación de muestra con el modelo entrenado
    prompt_ids = tokenizer.encode("Alice looked at")
    with torch.no_grad():
        gen = model.generate(prompt_ids, max_tokens=60, temperature=0.8)
    sample = "Alice looked at" + tokenizer.decode(gen)[:120]

    return {
        "label": label,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "lr": lr,
        "n_params": n_params,
        "history": history,
        "final_train_loss": history[-1]["train"],
        "final_val_loss": history[-1]["val"],
        "elapsed_s": round(elapsed, 1),
        "sample": sample,
    }


# ── Experimento 1: efecto de d_model ─────────────────────────────────────────
logger.info("=== Experimento 1: d_model ===")
exp1 = [
    run_config(d_model=64, n_heads=4, n_layers=4, lr=3e-4, label="d64"),
    run_config(d_model=128, n_heads=4, n_layers=4, lr=3e-4, label="d128"),
    run_config(d_model=256, n_heads=4, n_layers=4, lr=3e-4, label="d256"),
]

# ── Experimento 2: efecto del learning rate ───────────────────────────────────
logger.info("=== Experimento 2: learning rate ===")
exp2 = [
    run_config(d_model=128, n_heads=4, n_layers=4, lr=1e-3, label="lr=1e-3"),
    run_config(d_model=128, n_heads=4, n_layers=4, lr=3e-4, label="lr=3e-4"),
    run_config(d_model=128, n_heads=4, n_layers=4, lr=1e-4, label="lr=1e-4"),
]

# ── Experimento 3: efecto del número de capas ─────────────────────────────────
logger.info("=== Experimento 3: n_layers ===")
exp3 = [
    run_config(d_model=128, n_heads=4, n_layers=2, lr=3e-4, label="L2"),
    run_config(d_model=128, n_heads=4, n_layers=4, lr=3e-4, label="L4"),
    run_config(d_model=128, n_heads=4, n_layers=6, lr=3e-4, label="L6"),
]

# ── Guardar resultados ────────────────────────────────────────────────────────
results = {
    "corpus_chars": CORPUS_CHARS,
    "vocab_size": VOCAB_SIZE,
    "context_size": CONTEXT_SIZE,
    "epochs": EPOCHS,
    "device": DEVICE,
    "exp1_d_model": exp1,
    "exp2_lr": exp2,
    "exp3_n_layers": exp3,
}

out = Path(__file__).parent / "exploration_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
logger.info(f"✓ Resultados guardados en {out}")
