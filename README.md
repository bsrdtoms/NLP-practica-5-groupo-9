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

### Exploración de hiperparámetros

Ver `informe_2609.html` para la exploración completa (3 experimentos: d_model, lr, n_layers).

| d_model | Parámetros | Descripción |
|---|---|---|
| 64 | ~430 K | Infraparamétrico para este corpus |
| 128 | ~1.1 M | **Óptimo** — mejor val_loss |
| 256 | ~3.5 M | Sobreajuste en corpus pequeño |

### NER — Vocabulario ampliado (vocab=1000)

El modelo NER usa un tokenizador BPE separado con `vocab_size=1000`, que captura
43 tokens multicarácter capitalizados vs. solo 2 con vocab=300:

| vocab | Entidades reconocibles |
|---|---|
| 300 | `Alice`, `Alic` (2 tokens) |
| **1000** | `Alice`, `Rabbit`, `Queen`, `Hatter`, `Caterpillar`, `Duchess`, `Gryphon`, `King`… (43 tokens) |

## Pre-entrega: corpus anotado y kappa de Cohen

El directorio `pre_entrega/` contiene el corpus etiquetado generado con dos heurísticas automáticas independientes, tratadas como dos anotadores distintos para calcular la kappa de Cohen:

| Heurística | Criterio | Entidades encontradas |
|---|---|---|
| **H1 (amplia)** | Token en mayúscula + no inicio de frase + alfabético (len ≥ 1) | 1 089 tokens (3.7 %) |
| **H2 (estricta)** | H1 + len ≥ 4 + frecuencia ≥ 2 en el corpus | 58 tokens (0.2 %) |

**Kappa de Cohen H1 vs H2: κ = 0.098** (acuerdo débil) — refleja la diferencia real de criterios: H1 incluye letras sueltas capitalizadas (fragmentos BPE como `I`, `W`, `A`…), H2 solo acepta nombres propios recurrentes (`Alice`, `Alic`).

### Relación entre la pre-entrega y el entrenamiento

> **El modelo NER no usa el TSV para entrenarse.** Los labels se generan en tiempo de ejecución mediante `auto_label()` en `p5/ner.py` (heurística equivalente a H1). El archivo `corpus_etiquetado.tsv` es un entregable académico que documenta y justifica el proceso de anotación.

### Contenido del zip pre-entrega

```
pre_entrega_G09.zip
├── corpus_etiquetado.tsv     # token_id | token | etiqueta_H1 | etiqueta_H2
├── metadatos.json            # estadísticas, distribución, kappa
└── anotacion_NER.md          # documentación completa del proceso
```

## Notas técnicas

- El corpus usa *Alice's Adventures in Wonderland* (Project Gutenberg, fichero 11) con mayúsculas originales preservadas.
- El NER usa etiquetado heurístico automático (`auto_label()`): token en mayúscula fuera de inicio de frase → entidad.
- Transfer learning: los pesos del backbone Transformer se transfieren del LLM al NER (excepto `pos_emb` y máscaras, que tienen tamaños distintos).
- Con BPE vocab_size=300 solo emergen 2 tokens multicarácter capitalizados: `Alic` y `Alice`.
