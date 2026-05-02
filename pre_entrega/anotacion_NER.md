# Documentación del proceso de anotación NER — TP5 PLN

**Grupo G09 | Thomas Bossard | 2026-04-29**

---

## 1. Corpus

- **Fuente**: *Alice's Adventures in Wonderland* + *Through the Looking-Glass* (Lewis Carroll, dominio público)
- **Fragmento anotado**: primeros 60 000 caracteres del corpus combinado
- **Tokenización**: BPE (Byte Pair Encoding) entrenado sobre el corpus completo, vocab_size=300

## 2. Esquema de etiquetas (BIO)

| Etiqueta | Significado |
|---|---|
| `O` | Fuera de entidad |
| `B-ENT` | Inicio de entidad nombrada |
| `I-ENT` | Interior de entidad nombrada (continuación) |

Se usa un esquema simplificado con una única clase de entidad (`ENT`) que agrupa personas, lugares y organizaciones del mundo de Alicia.

## 3. Metodología de anotación — dos heurísticas

Se definieron **dos heurísticas automáticas** con criterios distintos, tratadas como dos anotadores independientes:

### H1 — Heurística amplia

1. Token que empieza en **mayúscula** (cualquier longitud ≥ 1).
2. No está al **inicio de frase** (no sigue a `.`, `!`, `?`, `\n`).
3. Token **alfabético** (sin dígitos ni puntuación).
4. Tokens consecutivos candidatos forman una secuencia `B-ENT … I-ENT`.

> Incluye letras sueltas capitalizadas (`I`, `A`, `M`…) que son fragmentos BPE de palabras mayúsculas — muchos falsos positivos, criterio muy generoso.

### H2 — Heurística estricta

Igual que H1, más dos condiciones adicionales:

5. Longitud del token **≥ 4 caracteres** (excluye letras sueltas y abreviaturas).
6. El token aparece **≥ 2 veces en mayúscula** en el corpus completo (nombre propio recurrente, no hapax).

> Con BPE vocab_size=300 sólo emergen como tokens multicarácter capitalizados `Alic` y `Alice`, que son el nombre de la protagonista.

**Ventajas de ambas**: reproducibles, consistentes, sin anotación manual.  
**Limitaciones**: H1 produce exceso de falsos positivos; H2 es tan conservadora que casi solo detecta "Alice".

## 4. Estadísticas del corpus anotado

| Métrica | H1 (amplia) | H2 (estricta) |
|---|---|---|
| Tokens totales | 29 263 | 29 263 |
| Tokens O | 28 174 (96.3%) | 29 205 (99.8%) |
| Tokens B-ENT + I-ENT | 1 089 (3.7%) | 58 (0.2%) |

## 5. Kappa de Cohen inter-anotador (H1 vs H2)

Se calculó la **kappa de Cohen** entre las dos heurísticas sobre los 29 263 tokens:

- **Anotador 1**: H1 (amplia)
- **Anotador 2**: H2 (estricta)
- **Tokens en desacuerdo**: 1 031 (3.5 %)
- **Kappa calculada**: **κ = 0.0980** — *acuerdo débil*

> Interpretación estándar: κ < 0.40 débil, 0.40–0.60 moderado, 0.60–0.80 bueno, > 0.80 excelente.

El kappa bajo refleja la diferencia real entre los criterios: H1 marca como entidades todas las letras capitales en mitad de secuencia (fragmentos BPE como `I`, `W`, `A`…), mientras H2 sólo considera entidades los tokens de longitud ≥ 4 con frecuencia ≥ 2, que con este vocabulario son únicamente `Alice` y `Alic`.

**Tokens marcados por H1 pero no H2 (top 10)**: `I` (129×), `W` (77×), `A` (67×), `C` (65×), `M` (52×)…

## 6. Revisiones y decisiones

- **Decisión 1**: No se distinguen subtipos de entidades (PER/LOC/ORG) por limitación del método automático y del modelo BPE (tokens que pueden fragmentar nombres propios).
- **Decisión 2**: Se usan dos heurísticas automáticas reales (H1 vs H2) para el cálculo de kappa inter-anotador, en lugar de ruido artificial: el desacuerdo es genuino y refleja ambigüedad real del criterio de anotación.
- **Decisión 3**: El corpus usa *Alice's Adventures in Wonderland* (Project Gutenberg, fichero 11) con mayúsculas originales preservadas. BPE vocab_size=300 fragmenta la mayoría de palabras en subpalabras, produciendo un vocabulario mayúsculo muy reducido.
- **Decisión 4**: El fragmento de 60 000 caracteres (~primeros 8 capítulos) es suficiente para el entrenamiento NER demo con este vocabulario.

## 7. Contenido del zip

```
pre_entrega_G09.zip
├── corpus_etiquetado.tsv   # token_id \t token \t etiqueta
├── metadatos.json          # estadísticas, distribución, kappa
└── anotacion_NER.md        # este fichero
```
