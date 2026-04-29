# Reconocimiento de entidades nombradas (NER) con Transformer bidireccional
#
# PLN 2025/2026 (FDI UCM)

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset

from p5.transformer import Transformer

# Usamos un esquema simple: O (fuera), B (inicio entidad), I (interior entidad)
LABELS = ["O", "B-ENT", "I-ENT"]
O_LABEL, B_LABEL, I_LABEL = 0, 1, 2


class NERModel(nn.Module):
    """Transformer bidireccional con cabeza de clasificación para NER.

    A diferencia del LLM causal, aquí usamos causal=False para que cada
    token vea todo el contexto (izquierda y derecha), lo que es mejor
    para entender entidades en contexto.
    """

    def __init__(self, vocab_size, max_seq_len, d_model, n_heads, n_layers, expansion, dropout):
        super().__init__()
        self.transformer = Transformer(
            vocab_size, max_seq_len, d_model, n_heads, n_layers, expansion, dropout
        )
        # Capa lineal que proyecta cada hidden state a las etiquetas NER
        self.classifier = nn.Linear(d_model, len(LABELS))

    def forward(self, idx, labels=None):
        # causal=False -> atención bidireccional, cada token ve todo el contexto
        hidden = self.transformer(idx, causal=False)
        logits = self.classifier(hidden)  # (batch, n_tokens, n_labels)

        if labels is None:
            return logits, None

        loss = cross_entropy(logits.flatten(0, 1), labels.flatten())
        return logits, loss

    def predict(self, idx):
        self.eval()
        with torch.no_grad():
            logits, _ = self(idx)
        return logits.argmax(dim=-1)


def auto_label(tokens, vocab):
    """Genera etiquetas NER automáticas usando heurísticas de capitalización.

    Heurística: token que empieza en mayúscula y no es inicio de frase
    -> probable entidad nombrada (persona, lugar...).

    Es una aproximación muy simple, pero suficiente para demostrar el concepto.
    """
    labels = []
    prev_entity = False

    for i, tok_id in enumerate(tokens):
        tok = vocab[tok_id] if tok_id < len(vocab) else ""

        # Miramos si el token anterior es puntuación final o salto de línea
        if i > 0:
            prev_tok = vocab[tokens[i - 1]] if tokens[i - 1] < len(vocab) else ""
            after_punct = any(prev_tok.endswith(c) for c in (".", "!", "?", "\n"))
        else:
            after_punct = True  # inicio del texto = no es entidad

        # Entidad candidata: empieza en mayúscula, tiene más de 1 char, no es inicio de frase
        is_entity = tok and tok[0].isupper() and len(tok) > 1 and tok.isalpha() and not after_punct

        if is_entity:
            if prev_entity:
                labels.append(I_LABEL)
            else:
                labels.append(B_LABEL)
            prev_entity = True
        else:
            labels.append(O_LABEL)
            prev_entity = False

    return labels


class NERDataset(Dataset):
    def __init__(self, tokens, labels, seq_len):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        return (
            self.tokens[idx : idx + self.seq_len],
            self.labels[idx : idx + self.seq_len],
        )


def train_ner(model, tokens, labels, epochs=3, context_size=64, batch_size=32, lr=3e-4):
    """Entrena el modelo NER sobre los tokens y etiquetas dados."""
    dataset = NERDataset(tokens, labels, context_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    device = next(model.parameters()).device

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0, 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        print(f"  Época NER {epoch + 1}/{epochs} | loss={total_loss / n:.4f}")


def extract_entities(model, tokenizer, text, context_size=64):
    """Extrae entidades nombradas de un texto usando el modelo NER."""
    model.eval()
    device = next(model.parameters()).device

    tokens = tokenizer.encode(text)

    # Procesamos el texto en trozos de context_size
    all_preds = []
    for i in range(0, len(tokens), context_size):
        chunk = tokens[i : i + context_size]
        x = torch.tensor([chunk], dtype=torch.long, device=device)
        preds = model.predict(x)[0].tolist()
        all_preds.extend(preds)

    # Reconstruimos las entidades a partir de las etiquetas BIO
    entities = []
    current = []
    for tok_id, label in zip(tokens, all_preds):
        tok = tokenizer.vocab[tok_id] if tok_id < len(tokenizer.vocab) else ""
        if label == B_LABEL:
            if current:
                entities.append("".join(current))
            current = [tok]
        elif label == I_LABEL and current:
            current.append(tok)
        else:
            if current:
                entities.append("".join(current))
                current = []
    if current:
        entities.append("".join(current))

    # Quitamos duplicados manteniendo el orden
    seen = set()
    unique = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    return unique
