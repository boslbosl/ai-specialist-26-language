# Chapter 3 Attention Diagrams

- 관련 문서: [../Chapter_3_Excercise_Attention.md](../Chapter_3_Excercise_Attention.md)
- 관련 노트북: [../../Chapter_3_Excercise_Attention.ipynb](../../Chapter_3_Excercise_Attention.ipynb)

## Notebook Flow

```mermaid
flowchart TD
    A["Cell 0<br/>CausalAttention class"] --> B["Cell 1<br/>Sample input batch"]
    B --> C["Run CausalAttention"]
    C --> D["context_vecs<br/>shape: [2, 6, 2]"]

    D --> E["Cell 2<br/>MultiHeadAttentionWrapper"]
    E --> F["Run 2 independent CausalAttention heads"]
    F --> G["Concatenate head outputs"]
    G --> H["context_vecs<br/>shape: [2, 6, 4]"]

    H --> I["Cell 3<br/>MultiHeadAttention"]
    I --> J["Single Q/K/V projection<br/>split into heads"]
    J --> K["Scaled dot-product attention per head"]
    K --> L["Merge heads + output projection"]
    L --> M["context_vecs<br/>shape: [2, 6, 2]"]
```

## CausalAttention

```mermaid
flowchart TD
    X["Input x<br/>[batch, num_tokens, d_in]"] --> Q["W_query(x)<br/>queries"]
    X --> K["W_key(x)<br/>keys"]
    X --> V["W_value(x)<br/>values"]

    Q --> S["Attention scores<br/>queries @ keys.T"]
    K --> S

    S --> Mask["Causal mask<br/>future positions = -inf"]
    Mask --> Scale["Scale<br/>divide by sqrt(d_out)"]
    Scale --> Softmax["Softmax<br/>attention weights"]
    Softmax --> Dropout["Dropout"]
    Dropout --> Context["Context vectors<br/>attn_weights @ values"]
    V --> Context

    Context --> Out["Output<br/>[batch, num_tokens, d_out]"]
```

## Causal Mask

```mermaid
flowchart LR
    T0["Token 0"] --> P0["볼 수 있음:<br/>Token 0"]
    T1["Token 1"] --> P1["볼 수 있음:<br/>Token 0, Token 1"]
    T2["Token 2"] --> P2["볼 수 있음:<br/>Token 0, Token 1, Token 2"]
    Tn["Token n"] --> Pn["볼 수 있음:<br/>Tokens 0..n"]

    F["미래 토큰"] -. "masked" .-> Z["softmax 후 확률 0"]
```

## MultiHeadAttentionWrapper

```mermaid
flowchart TD
    X["Input batch<br/>[2, 6, 3]"] --> H1["CausalAttention head 1<br/>d_out = 2"]
    X --> H2["CausalAttention head 2<br/>d_out = 2"]

    H1 --> O1["Head 1 output<br/>[2, 6, 2]"]
    H2 --> O2["Head 2 output<br/>[2, 6, 2]"]

    O1 --> Cat["torch.cat(..., dim=-1)"]
    O2 --> Cat

    Cat --> Out["Wrapper output<br/>[2, 6, 4]"]
```

## MultiHeadAttention

```mermaid
flowchart TD
    X["Input x<br/>[batch, num_tokens, d_in]"] --> Linear["One Q/K/V projection each<br/>d_in -> d_out"]

    Linear --> View["view<br/>[batch, num_tokens, num_heads, head_dim]"]
    View --> Transpose["transpose<br/>[batch, num_heads, num_tokens, head_dim]"]

    Transpose --> Scores["Per-head scores<br/>queries @ keys.transpose"]
    Scores --> Mask["Apply causal mask"]
    Mask --> Weights["softmax(scores / sqrt(head_dim))"]
    Weights --> Dropout["Dropout"]
    Dropout --> Context["attn_weights @ values"]

    Context --> TransposeBack["transpose back<br/>[batch, num_tokens, num_heads, head_dim]"]
    TransposeBack --> Merge["contiguous().view<br/>[batch, num_tokens, d_out]"]
    Merge --> Proj["out_proj"]
    Proj --> Out["Output<br/>[batch, num_tokens, d_out]"]
```

## Shape Summary

```mermaid
flowchart LR
    A["Sample inputs<br/>[6, 3]"] --> B["Batch<br/>torch.stack<br/>[2, 6, 3]"]
    B --> C["CausalAttention<br/>d_out=2"]
    C --> D["[2, 6, 2]"]
    B --> E["Wrapper<br/>2 heads x d_out=2"]
    E --> F["[2, 6, 4]"]
    B --> G["Integrated MHA<br/>d_out=2, num_heads=2"]
    G --> H["[2, 6, 2]"]
```
