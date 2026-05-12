# Chapter 4 GPT Diagrams

- 관련 문서: [../Chapter_4_Excercise_GPT.md](../Chapter_4_Excercise_GPT.md)
- 관련 노트북: [../../Chapter_4_Excercise_GPT.ipynb](../../Chapter_4_Excercise_GPT.ipynb)

## 전체 구조

```mermaid
flowchart TD
    A["Raw text"] --> B["Tokenizer<br/>text -> token ids"]
    B --> C["Input ids<br/>[batch, seq_len]"]
    C --> D["Token embedding<br/>[batch, seq_len, emb_dim]"]
    C --> E["Position embedding<br/>[seq_len, emb_dim]"]
    D --> F["Embedding sum"]
    E --> F
    F --> G["Dropout"]
    G --> H["TransformerBlock x n_layers"]
    H --> I["Final LayerNorm"]
    I --> J["Output head<br/>Linear emb_dim -> vocab_size"]
    J --> K["Logits<br/>[batch, seq_len, vocab_size]"]
```

## TransformerBlock

```mermaid
flowchart TD
    X["Input x"] --> S1["shortcut"]
    X --> LN1["LayerNorm 1"]
    LN1 --> MHA["Masked Multi-Head Attention"]
    MHA --> Drop1["Dropout"]
    Drop1 --> Add1["Residual add"]
    S1 --> Add1
    Add1 --> S2["shortcut"]
    Add1 --> LN2["LayerNorm 2"]
    LN2 --> FFN["FeedForward<br/>Linear -> GELU -> Linear"]
    FFN --> Drop2["Dropout"]
    Drop2 --> Add2["Residual add"]
    S2 --> Add2
    Add2 --> Out["Block output"]
```

## FeedForward

```mermaid
flowchart LR
    X["x<br/>emb_dim"] --> A["Linear<br/>emb_dim -> 4 * emb_dim"]
    A --> G["GELU"]
    G --> B["Linear<br/>4 * emb_dim -> emb_dim"]
    B --> Y["output<br/>emb_dim"]
```

## Text Generation Loop

```mermaid
flowchart TD
    A["Current ids<br/>idx"] --> B["Crop context<br/>idx[:, -context_size:]"]
    B --> C["model(idx_cond)"]
    C --> D["logits<br/>[batch, seq_len, vocab_size]"]
    D --> E["Take last position<br/>logits[:, -1, :]"]
    E --> F["Select next token<br/>argmax or sampling"]
    F --> G["Append to idx<br/>torch.cat"]
    G --> H{"목표 길이 도달"}
    H -- "No" --> B
    H -- "Yes" --> I["Decode ids -> text"]
```

## Shape Summary

```mermaid
flowchart LR
    A["encoded ids<br/>[1, seq_len]"] --> B["token + position embedding<br/>[1, seq_len, 768]"]
    B --> C["Transformer blocks<br/>[1, seq_len, 768]"]
    C --> D["logits<br/>[1, seq_len, 50257]"]
    D --> E["last logits<br/>[1, 50257]"]
    E --> F["next token id<br/>[1, 1]"]
```
