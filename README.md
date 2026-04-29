# TP5 PLN — LLM causal + NER sobre Alice in Wonderland

**Grupo 09 — PLN 2025/2026, FDI UCM**

## Integrantes del equipo

- Thomas Bossard

## Descripción

Implementación de un LLM causal basado en Transformer para modelado de lenguaje sobre el corpus de *Alicia en el país de las maravillas* de Lewis Carroll, con un módulo adicional de reconocimiento de entidades nombradas (NER) con transfer learning.

### Componentes

| Módulo | Descripción |
|---|---|
| `p5/tokenizer.py` | Tokenizador BPE entrenado sobre el corpus |
| `p5/transformer.py` | Bloque Transformer (atención multi-cabeza + FFN) |
| `p5/causal_llm.py` | LLM causal (predicción del siguiente token) |
| `p5/causal_train.py` | Bucle de entrenamiento con AdamW |
| `p5/ner.py` | Modelo NER bidireccional + etiquetado automático BIO |
| `p5/cli.py` | CLI con typer: train-llm, train-ner, generate, ner |

## Instalación

```bash
uv pip install fdi_pln_2609_p5-1.0-py3-none-any.whl
```

O en modo desarrollo:

```bash
uv pip install -e .
```

## Uso

### Entrenar el LLM causal

```bash
uv run fdi-pln-2609-p5 train-llm --corpus alicia --out p5_causal_2609.pth
```

### Entrenar el NER (requiere pesos LLM pre-entrenados)

```bash
uv run fdi-pln-2609-p5 train-ner --llm-weights p5_causal_2609.pth --out p5_ner_2609.pth
```

### Generar texto

```bash
uv run fdi-pln-2609-p5 generate --weights p5_causal_2609.pth --prompt "Alice was very"
```

### Extraer entidades nombradas de un fichero

```bash
uv run fdi-pln-2609-p5 ner --weights p5_ner_2609.pth --file mi_texto.txt
```

## Resultados (entrenamiento en GPU NVIDIA A2, CUDA 12.9)

### LLM causal (5 épocas, vocab=300, d_model=128, 4 capas, 4 cabezas)

| Época | Train loss | Val loss |
|---|---|---|
| 1 | 5.11 | 3.82 |
| 2 | 3.72 | 3.45 |
| 3 | 3.18 | 3.28 |
| 4 | 2.72 | 3.21 |
| 5 | 2.38 | 3.18 |

### Exploración de hiperparámetros (d_model)

| d_model | Parámetros |
|---|---|
| 64 | ~430 K |
| 128 | ~1.1 M |
| 256 | ~3.5 M |

## Notas técnicas

- El corpus es el texto completo de *Alice's Adventures in Wonderland* y *Through the Looking-Glass*.
- El NER usa etiquetado heurístico automático (mayúsculas fuera de inicio de frase → entidad).
- Transfer learning: los pesos del backbone Transformer se transfieren del LLM al NER (excepto `pos_emb` y máscaras, que tienen tamaños distintos).
