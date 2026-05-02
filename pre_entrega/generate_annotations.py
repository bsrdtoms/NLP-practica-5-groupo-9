"""
Genera el corpus etiquetado para la Pre-entrega P5.

Dos heurísticas de anotación automática:
  H1 (amplia) : token en mayúscula + no inicio de frase + alfabético (len >= 1)
                → incluye iniciales sueltas ("I", "A", "M"…) que son falsos positivos
  H2 (estricta): token en mayúscula + no inicio de frase + alfabético + len >= 4
                  + aparece ≥ 2 veces en mayúscula en el corpus
                  (= nombres propios recurrentes — sólo "Alic", "Alice" con BPE 300)

El kappa de Cohen entre H1 y H2 mide el acuerdo real entre dos criterios
de anotación distintos: H1 es generosa (más falsos positivos), H2 es conservadora.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from p5.corpus import load_corpus
from p5.tokenizer import BPETokenizer

LABELS = ["O", "B-ENT", "I-ENT"]
O_LABEL, B_LABEL, I_LABEL = 0, 1, 2
VOCAB_SIZE = 300


# ---------------------------------------------------------------------------
# Heurística H1 — amplia
# Criterio: token en mayúscula + no inicio de frase + alfabético (len >= 1)
# Nota: con BPE vocab_size=300 los tokens de una sola letra ("I", "A", "T"...)
#       aparecen en mitad de palabras capitalizadas → muchos falsos positivos,
#       pero es la heurística más laxa posible.
# ---------------------------------------------------------------------------
def label_h1(tokens, vocab):
    labels = []
    prev_entity = False
    for i, tok_id in enumerate(tokens):
        tok = vocab[tok_id] if tok_id < len(vocab) else ""
        after_punct = True
        if i > 0:
            prev_tok = vocab[tokens[i - 1]] if tokens[i - 1] < len(vocab) else ""
            after_punct = any(prev_tok.endswith(c) for c in (".", "!", "?", "\n"))
        is_entity = tok and tok[0].isupper() and len(tok) >= 1 and tok.isalpha() and not after_punct
        if is_entity:
            labels.append(I_LABEL if prev_entity else B_LABEL)
            prev_entity = True
        else:
            labels.append(O_LABEL)
            prev_entity = False
    return labels


# ---------------------------------------------------------------------------
# Heurística H2 — estricta
# Criterio: igual que H1 + len > 3 + el token aparece ≥ 2 veces en mayúscula
# en el corpus completo (= nombre propio recurrente, no palabra común)
# ---------------------------------------------------------------------------
def label_h2(tokens, vocab, min_len=4, min_freq=2):
    # Contar cuántas veces aparece cada token en mayúscula en el corpus completo
    cap_freq = Counter(
        vocab[t] for t in tokens if t < len(vocab) and vocab[t] and vocab[t][0].isupper()
    )

    labels = []
    prev_entity = False
    for i, tok_id in enumerate(tokens):
        tok = vocab[tok_id] if tok_id < len(vocab) else ""
        after_punct = True
        if i > 0:
            prev_tok = vocab[tokens[i - 1]] if tokens[i - 1] < len(vocab) else ""
            after_punct = any(prev_tok.endswith(c) for c in (".", "!", "?", "\n"))
        is_entity = (
            tok
            and tok[0].isupper()
            and len(tok) >= min_len
            and tok.isalpha()
            and not after_punct
            and cap_freq[tok] >= min_freq
        )
        if is_entity:
            labels.append(I_LABEL if prev_entity else B_LABEL)
            prev_entity = True
        else:
            labels.append(O_LABEL)
            prev_entity = False
    return labels


# ---------------------------------------------------------------------------
# Kappa de Cohen
# ---------------------------------------------------------------------------
def cohen_kappa(y1, y2, n_classes=3):
    n = len(y1)
    po = sum(a == b for a, b in zip(y1, y2)) / n
    c1 = Counter(y1)
    c2 = Counter(y2)
    pe = sum((c1[c] / n) * (c2[c] / n) for c in range(n_classes))
    if abs(1 - pe) < 1e-10:
        return 1.0
    return (po - pe) / (1 - pe)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
out_dir = Path(__file__).parent

print("Cargando corpus y tokenizador...")
# Usar la versión con mayúsculas correctas descargada de Project Gutenberg
alice_proper = Path(__file__).parent.parent / "alicia" / "alice_proper.txt"
if alice_proper.exists():
    text = alice_proper.read_text(encoding="utf-8")
    print(f"  → usando {alice_proper.name} ({len(text):,} caracteres)")
else:
    text = load_corpus("alicia")
tokenizer = BPETokenizer(text, vocab_size=VOCAB_SIZE)

ner_text = text[:60_000]
ner_tokens = tokenizer.encode(ner_text)
vocab = tokenizer.vocab

print(f"Corpus: {len(ner_tokens):,} tokens")

# Anotar con las dos heurísticas
labels_h1 = label_h1(ner_tokens, vocab)
labels_h2 = label_h2(ner_tokens, vocab)

n_ent_h1 = sum(1 for l in labels_h1 if l > 0)
n_ent_h2 = sum(1 for l in labels_h2 if l > 0)
print(f"H1 (amplia)  : {n_ent_h1:,} tokens de entidad ({100*n_ent_h1/len(ner_tokens):.1f}%)")
print(f"H2 (estricta): {n_ent_h2:,} tokens de entidad ({100*n_ent_h2/len(ner_tokens):.1f}%)")

# Kappa H1 vs H2
kappa = cohen_kappa(labels_h1, labels_h2)
print(f"\nKappa de Cohen H1 vs H2: {kappa:.4f}")
if kappa > 0.80:
    interpretation = "excelente"
elif kappa > 0.60:
    interpretation = "bueno"
elif kappa > 0.40:
    interpretation = "moderado"
else:
    interpretation = "débil"
print(f"Interpretación: acuerdo {interpretation}")

# Análisis del desacuerdo
disagreements = sum(1 for a, b in zip(labels_h1, labels_h2) if a != b)
print(f"Tokens en desacuerdo: {disagreements:,} ({100*disagreements/len(ner_tokens):.1f}%)")

# Ejemplos de desacuerdo (tokens que H1 marca como entidad pero H2 no)
h1_not_h2 = [
    vocab[ner_tokens[i]]
    for i in range(len(ner_tokens))
    if labels_h1[i] == B_LABEL and labels_h2[i] == O_LABEL
]
counter_disagree = Counter(h1_not_h2).most_common(10)
print(f"\nTokens marcados por H1 pero no H2 (más frecuentes):")
for tok, cnt in counter_disagree:
    print(f"  '{tok}': {cnt}x")

# ---- Exportar TSV con ambas anotaciones ----
with open(out_dir / "corpus_etiquetado.tsv", "w", encoding="utf-8") as f:
    f.write("token_id\ttoken\tetiqueta_H1\tetiqueta_H2\n")
    for i, tok_id in enumerate(ner_tokens):
        tok = vocab[tok_id].replace("\n", "\\n").replace("\t", "\\t")
        f.write(f"{i}\t{tok}\t{LABELS[labels_h1[i]]}\t{LABELS[labels_h2[i]]}\n")
print("\n✓ corpus_etiquetado.tsv (con H1 y H2)")

# ---- Guardar metadatos ----
metadata = {
    "proyecto": "TP5 PLN 2025/2026 - FDI UCM",
    "grupo": "G09",
    "anotadores": [
        {
            "id": "H1",
            "nombre": "Heurística amplia",
            "descripcion": "Token en mayúscula + no inicio de frase + alfabético (len >= 1, incluye iniciales)",
            "fecha": "2026-04-29",
            "responsable": "Thomas Bossard",
        },
        {
            "id": "H2",
            "nombre": "Heurística estricta",
            "descripcion": "Igual que H1 + len >= 4 + frecuencia >= 2 en corpus (nombre propio recurrente)",
            "fecha": "2026-04-29",
            "responsable": "Thomas Bossard",
        },
    ],
    "corpus": {
        "fuente": "Alice's Adventures in Wonderland + Through the Looking-Glass (Lewis Carroll)",
        "tokens_totales": len(ner_tokens),
        "H1": {
            "tokens_entidad": n_ent_h1,
            "distribucion": {LABELS[i]: sum(1 for l in labels_h1 if l == i) for i in range(3)},
        },
        "H2": {
            "tokens_entidad": n_ent_h2,
            "distribucion": {LABELS[i]: sum(1 for l in labels_h2 if l == i) for i in range(3)},
        },
    },
    "esquema_etiquetas": LABELS,
    "acuerdo_inter_anotador": {
        "kappa_cohen": round(kappa, 4),
        "interpretacion": f"acuerdo {interpretation}",
        "tokens_en_desacuerdo": disagreements,
        "porcentaje_desacuerdo": round(100 * disagreements / len(ner_tokens), 2),
    },
}

with open(out_dir / "metadatos.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print("✓ metadatos.json")
