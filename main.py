#!/usr/bin/env python3
"""
TP 5 - LLM sobre Alicia en el país de las maravillas
PLN 2025/2026 (FDI UCM)

Tareas:
  1. Tokenizar el corpus con BPE
  2. Pre-entrenar un CausalLLM en el corpus
  3. Generar texto a partir de prompts
  4. Entrenar un modelo NER y extraer entidades
  5. Explorar hiperparámetros
"""

import torch
from loguru import logger

from p5.corpus import load_corpus
from p5.tokenizer import BPETokenizer
from p5.causal_llm import CausalLLM
from p5.causal_train import train
from p5.ner import NERModel, auto_label, train_ner, extract_entities


# ---- Hiperparámetros ----
CORPUS_PATH = "alicia"
VOCAB_SIZE   = 300
CONTEXT_SIZE = 128
D_MODEL      = 128
N_HEADS      = 4
N_LAYERS     = 4
EXPANSION    = 4
DROPOUT      = 0.1
EPOCHS       = 5
BATCH_SIZE   = 64
LR           = 3e-4


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Dispositivo: {device}")

    # ------------------------------------------------------------------
    # 1. Cargar corpus y tokenizar
    # ------------------------------------------------------------------
    logger.info("Cargando corpus...")
    text = load_corpus(CORPUS_PATH)
    logger.info(f"Corpus cargado: {len(text):,} caracteres")

    logger.info("Entrenando tokenizador BPE...")
    tokenizer = BPETokenizer(text, vocab_size=VOCAB_SIZE)
    tokens = tokenizer.encode(text)
    logger.info(f"Vocabulario: {len(tokenizer.vocab)} tokens | Corpus: {len(tokens):,} tokens")

    # ------------------------------------------------------------------
    # 2. Pre-entrenar el LLM causal
    # ------------------------------------------------------------------
    model = CausalLLM(
        vocab_size=VOCAB_SIZE,
        max_seq_len=CONTEXT_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        expansion=EXPANSION,
        dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Modelo: {n_params:,} parámetros")

    logger.info("Entrenando LLM causal...")
    train(
        model, tokens,
        epochs=EPOCHS,
        context_size=CONTEXT_SIZE,
        batch_size=BATCH_SIZE,
        lr=LR,
    )

    # ------------------------------------------------------------------
    # 3. Generación de texto
    # ------------------------------------------------------------------
    logger.info("=== Generación de texto ===")
    prompts = [
        "Alice was very",
        "The White Rabbit",
        "said the Queen",
        "down the rabbit hole",
    ]
    for prompt in prompts:
        encoded = tokenizer.encode(prompt)
        generated = model.generate(encoded, max_tokens=150, temperature=0.8)
        decoded = tokenizer.decode(generated)
        print(f"\n>>> Prompt: '{prompt}'")
        print(decoded[:400])
        print("---")

    # ------------------------------------------------------------------
    # 4. NER - Reconocimiento de entidades nombradas
    # ------------------------------------------------------------------
    logger.info("=== NER ===")

    # Usamos solo un fragmento del corpus para ser más rápidos
    ner_text = text[:60_000]
    ner_tokens = tokenizer.encode(ner_text)
    ner_labels = auto_label(ner_tokens, tokenizer.vocab)

    n_entities = sum(1 for l in ner_labels if l > 0)
    logger.info(f"Datos NER: {len(ner_tokens):,} tokens, {n_entities:,} tokens de entidad")

    # Creamos el modelo NER (transformer bidireccional + cabeza clasificadora)
    NER_CONTEXT = 64
    ner_model = NERModel(
        vocab_size=VOCAB_SIZE,
        max_seq_len=NER_CONTEXT,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        expansion=EXPANSION,
        dropout=DROPOUT,
    ).to(device)

    # Transferimos los pesos pre-entrenados del LLM al transformer del NER
    # (solo los pesos del backbone, no la cabeza lm_head)
    pretrained = {k: v for k, v in model.state_dict().items()
                  if k != "lm_head.weight" and "pos_emb" not in k and "mask" not in k}
    ner_backbone_state = {f"transformer.{k}": v for k, v in pretrained.items()}
    missing, unexpected = ner_model.load_state_dict(ner_backbone_state, strict=False)
    logger.info(f"Pesos transferidos. Capas nuevas (a entrenar): {missing}")

    logger.info("Entrenando modelo NER...")
    train_ner(ner_model, ner_tokens, ner_labels, epochs=3, context_size=NER_CONTEXT)

    # Extraemos entidades en frases de prueba
    test_sentences = [
        "Alice looked at the White Rabbit nervously.",
        "The Queen of Hearts was shouting at everyone.",
        "Humpty Dumpty sat on a wall near Wonderland.",
        "Alice and the Cheshire Cat had a strange conversation.",
    ]
    logger.info("Entidades encontradas:")
    for sent in test_sentences:
        entities = extract_entities(ner_model, tokenizer, sent, context_size=NER_CONTEXT)
        print(f"  '{sent}'")
        print(f"  -> {entities}")

    # ------------------------------------------------------------------
    # 5. Exploración de hiperparámetros (pequeño experimento)
    # ------------------------------------------------------------------
    logger.info("=== Experimento: efecto de d_model ===")
    for d in [64, 128, 256]:
        m = CausalLLM(
            vocab_size=VOCAB_SIZE,
            max_seq_len=CONTEXT_SIZE,
            d_model=d,
            n_heads=4,
            n_layers=4,
            expansion=4,
            dropout=0.1,
        ).to(device)
        n = sum(p.numel() for p in m.parameters())
        logger.info(f"  d_model={d:4d} -> {n:,} parámetros")


if __name__ == "__main__":
    main()
