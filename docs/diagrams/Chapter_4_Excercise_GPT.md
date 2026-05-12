# Chapter 4 GPT Diagrams

- 관련 문서: [../Chapter_4_Excercise_GPT.md](../Chapter_4_Excercise_GPT.md)
- 관련 노트북: [../../Chapter_4_Excercise_GPT.ipynb](../../Chapter_4_Excercise_GPT.ipynb)

## Chapter 4 전체 구조

```mermaid
flowchart TD
    A["Raw text<br/>Hello, I am"] --> B["Tokenizer<br/>text -> token ids"]
    B --> C["Input ids<br/>[batch, seq_len]"]

    C --> D["GPTModel.forward"]
    D --> E["Token embedding<br/>[batch, seq_len, emb_dim]"]
    D --> F["Position embedding<br/>[seq_len, emb_dim]"]
    E --> G["Embedding sum<br/>token + position"]
    F --> G
    G --> H["Dropout"]
    H --> I["TransformerBlock x n_layers"]
    I --> J["Final LayerNorm"]
    J --> K["Output head<br/>Linear emb_dim -> vocab_size"]
    K --> L["Logits<br/>[batch, seq_len, vocab_size]"]
    L --> M["Next-token selection<br/>argmax last position"]
    M --> N["Append token id"]
    N --> C
```

## GPTModel 내부

```mermaid
flowchart TD
    IDs["in_idx<br/>[batch, seq_len]"] --> Tok["tok_emb<br/>nn.Embedding(vocab_size, emb_dim)"]
    IDs --> PosIndex["torch.arange(seq_len)"]
    PosIndex --> Pos["pos_emb<br/>nn.Embedding(context_length, emb_dim)"]

    Tok --> Sum["tok_embeds + pos_embeds"]
    Pos --> Sum
    Sum --> Drop["drop_emb"]
    Drop --> Blocks["trf_blocks<br/>TransformerBlock repeated n_layers"]
    Blocks --> Norm["final_norm"]
    Norm --> Head["out_head<br/>nn.Linear(emb_dim, vocab_size)"]
    Head --> Logits["logits<br/>[batch, seq_len, vocab_size]"]
```

## TransformerBlock 구조

```mermaid
flowchart TD
    X["Input x<br/>[batch, seq_len, emb_dim]"] --> S1["shortcut = x"]
    X --> LN1["LayerNorm 1<br/>pre-norm"]
    LN1 --> ATT["Masked Multi-Head Attention"]
    ATT --> Drop1["Dropout"]
    Drop1 --> Add1["Add residual<br/>x + shortcut"]
    S1 --> Add1

    Add1 --> S2["shortcut = x"]
    Add1 --> LN2["LayerNorm 2<br/>pre-norm"]
    LN2 --> FF["FeedForward<br/>Linear -> GELU -> Linear"]
    FF --> Drop2["Dropout"]
    Drop2 --> Add2["Add residual<br/>x + shortcut"]
    S2 --> Add2
    Add2 --> Out["Block output<br/>[batch, seq_len, emb_dim]"]
```

## FeedForward 구조

```mermaid
flowchart LR
    X["x<br/>emb_dim"] --> A["Linear<br/>emb_dim -> 4 * emb_dim"]
    A --> G["GELU"]
    G --> B["Linear<br/>4 * emb_dim -> emb_dim"]
    B --> Y["output<br/>emb_dim"]
```

FeedForward는 토큰끼리 섞는 역할이 아니라, attention이 만든 각 토큰의 표현을 개별적으로 더 복잡하게 가공하는 역할입니다. 보통 내부 차원을 4배로 키웠다가 다시 줄입니다.

## Text Generation Loop

```mermaid
flowchart TD
    A["Current ids<br/>idx"] --> B["Crop context<br/>idx[:, -context_size:]"]
    B --> C["model(idx_cond)"]
    C --> D["logits<br/>[batch, seq_len, vocab_size]"]
    D --> E["Take last position<br/>logits[:, -1, :]"]
    E --> F["argmax<br/>next token id"]
    F --> G["torch.cat<br/>append to idx"]
    G --> H{"Generated enough<br/>tokens?"}
    H -- "No" --> B
    H -- "Yes" --> I["Decode ids -> text"]
```

## Shape Summary

```mermaid
flowchart LR
    A["Text"] --> B["tokenizer.encode"]
    B --> C["encoded_tensor<br/>[1, seq_len]"]
    C --> D["tok_embeds<br/>[1, seq_len, 768]"]
    D --> E["Transformer blocks<br/>[1, seq_len, 768]"]
    E --> F["logits<br/>[1, seq_len, 50257]"]
    F --> G["last logits<br/>[1, 50257]"]
    G --> H["next token id<br/>[1, 1]"]
```
