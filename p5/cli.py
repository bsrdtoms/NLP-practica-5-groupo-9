"""CLI para TP5 PLN: LLM causal + NER sobre Alice in Wonderland.

Uso:
    uv run fdi-pln-2609-p5 train-llm   [--corpus PATH] [--out PATH]
    uv run fdi-pln-2609-p5 train-ner   --llm-weights PATH [--corpus PATH] [--out PATH]
    uv run fdi-pln-2609-p5 generate    --weights PATH --prompt TEXT
    uv run fdi-pln-2609-p5 ner         --weights PATH --file PATH
"""

import sys
from pathlib import Path

import torch
import typer
from loguru import logger

app = typer.Typer(help="TP5 PLN 2025/2026 — LLM causal + NER (grupo 09)")

# ---- Hiperparámetros por defecto ----
VOCAB_SIZE = 300
CONTEXT_SIZE = 128
NER_CONTEXT = 64
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
EXPANSION = 4
DROPOUT = 0.1


def _load_tokenizer_and_text(corpus: str):
    from p5.corpus import load_corpus
    from p5.tokenizer import BPETokenizer

    text = load_corpus(corpus)
    logger.info(f"Corpus: {len(text):,} caracteres")
    tokenizer = BPETokenizer(text, vocab_size=VOCAB_SIZE)
    tokens = tokenizer.encode(text)
    logger.info(f"Vocab: {tokenizer.vocab_size} | Tokens: {len(tokens):,}")
    return tokenizer, tokens, text


@app.command()
def train_llm(
    corpus: str = typer.Option("alicia", help="Ruta al corpus (directorio o fichero)"),
    out: Path = typer.Option("p5_causal_2609.pth", help="Fichero de salida para los pesos"),
    epochs: int = typer.Option(5),
    batch_size: int = typer.Option(64),
    lr: float = typer.Option(3e-4),
):
    """Entrena el LLM causal y guarda los pesos."""
    from p5.causal_llm import CausalLLM
    from p5.causal_train import train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Dispositivo: {device}")

    tokenizer, tokens, _ = _load_tokenizer_and_text(corpus)

    model = CausalLLM(
        vocab_size=VOCAB_SIZE,
        max_seq_len=CONTEXT_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        expansion=EXPANSION,
        dropout=DROPOUT,
    ).to(device)

    logger.info(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    train(model, tokens, epochs=epochs, context_size=CONTEXT_SIZE, batch_size=batch_size, lr=lr)

    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": tokenizer.vocab,
            "tok2id": tokenizer.tok2id,
            "merges": tokenizer.merges,
        },
        out,
    )
    logger.info(f"Pesos guardados en {out}")


@app.command()
def train_ner(
    llm_weights: Path = typer.Option(..., help="Pesos del LLM pre-entrenado (.pth)"),
    corpus: str = typer.Option("alicia", help="Ruta al corpus"),
    out: Path = typer.Option("p5_ner_2609.pth", help="Fichero de salida para los pesos NER"),
    epochs: int = typer.Option(3),
):
    """Entrena el modelo NER (con transfer learning del LLM) y guarda los pesos."""
    from p5.causal_llm import CausalLLM
    from p5.corpus import load_corpus
    from p5.ner import NERModel, auto_label, train_ner as _train_ner
    from p5.tokenizer import BPETokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Dispositivo: {device}")

    # Cargar corpus y tokenizador
    text = load_corpus(corpus)
    tokenizer = BPETokenizer(text, vocab_size=VOCAB_SIZE)

    # Cargar pesos LLM para transferencia
    ckpt = torch.load(llm_weights, map_location=device)
    llm_state = ckpt["model_state"]

    ner_text = text[:60_000]
    ner_tokens = tokenizer.encode(ner_text)
    ner_labels = auto_label(ner_tokens, tokenizer.vocab)
    logger.info(
        f"NER: {len(ner_tokens):,} tokens, {sum(1 for l in ner_labels if l > 0):,} entidades"
    )

    ner_model = NERModel(
        vocab_size=VOCAB_SIZE,
        max_seq_len=NER_CONTEXT,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        expansion=EXPANSION,
        dropout=DROPOUT,
    ).to(device)

    pretrained = {
        k: v
        for k, v in llm_state.items()
        if k != "lm_head.weight" and "pos_emb" not in k and "mask" not in k
    }
    backbone_state = {f"transformer.{k}": v for k, v in pretrained.items()}
    missing, _ = ner_model.load_state_dict(backbone_state, strict=False)
    logger.info(f"Transfer learning OK. Capas nuevas: {missing}")

    _train_ner(ner_model, ner_tokens, ner_labels, epochs=epochs, context_size=NER_CONTEXT)

    torch.save(
        {
            "model_state": ner_model.state_dict(),
            "vocab": tokenizer.vocab,
            "tok2id": tokenizer.tok2id,
            "merges": tokenizer.merges,
        },
        out,
    )
    logger.info(f"Pesos NER guardados en {out}")


@app.command()
def generate(
    weights: Path = typer.Option(..., help="Pesos del LLM (.pth)"),
    prompt: str = typer.Option(..., help="Texto de inicio para la generación"),
    max_tokens: int = typer.Option(200),
    temperature: float = typer.Option(0.8),
):
    """Genera texto a partir de un prompt usando el LLM pre-entrenado."""
    from p5.causal_llm import CausalLLM
    from p5.tokenizer import BPETokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(weights, map_location=device)
    tokenizer = BPETokenizer.__new__(BPETokenizer)
    tokenizer.vocab = ckpt["vocab"]
    tokenizer.tok2id = ckpt["tok2id"]
    tokenizer.merges = ckpt["merges"]
    tokenizer.vocab_size = len(tokenizer.vocab)

    model = CausalLLM(
        vocab_size=VOCAB_SIZE,
        max_seq_len=CONTEXT_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        expansion=EXPANSION,
        dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    encoded = tokenizer.encode(prompt)
    generated = model.generate(encoded, max_tokens=max_tokens, temperature=temperature)
    decoded = tokenizer.decode(generated)
    print(f"\n>>> {prompt}{decoded[:500]}")


@app.command()
def ner(
    weights: Path = typer.Option(..., help="Pesos del modelo NER (.pth)"),
    file: Path = typer.Option(..., help="Fichero de texto a analizar"),
):
    """Extrae entidades nombradas de un fichero de texto."""
    from p5.ner import NERModel, extract_entities
    from p5.tokenizer import BPETokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(weights, map_location=device)
    tokenizer = BPETokenizer.__new__(BPETokenizer)
    tokenizer.vocab = ckpt["vocab"]
    tokenizer.tok2id = ckpt["tok2id"]
    tokenizer.merges = ckpt["merges"]
    tokenizer.vocab_size = len(tokenizer.vocab)

    ner_model = NERModel(
        vocab_size=VOCAB_SIZE,
        max_seq_len=NER_CONTEXT,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        expansion=EXPANSION,
        dropout=DROPOUT,
    ).to(device)
    ner_model.load_state_dict(ckpt["model_state"])

    text = Path(file).read_text(encoding="utf-8")

    # Traiter phrase par phrase (split sur les lignes non vides)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        lines = [text]

    for line in lines:
        entities = extract_entities(ner_model, tokenizer, line, context_size=NER_CONTEXT)
        print(f"  '{line[:80]}{'...' if len(line) > 80 else ''}'")
        print(f"  -> {entities}\n")


if __name__ == "__main__":
    app()
