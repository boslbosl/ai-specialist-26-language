# Chapter 3 Attention Diagrams

- 관련 문서: [../Chapter_3_Excercise_Attention.md](../Chapter_3_Excercise_Attention.md)
- 관련 노트북: [../../Chapter_3_Excercise_Attention.ipynb](../../Chapter_3_Excercise_Attention.ipynb)

## 노트북 흐름

```mermaid
flowchart TD
    A["CausalAttention class"] --> B["Sample input batch"]
    B --> C["Run CausalAttention"]
    C --> D["context_vecs<br/>[2, 6, 2]"]
    D --> E["MultiHeadAttentionWrapper"]
    E --> F["Run independent heads"]
    F --> G["torch.cat along last dim"]
    G --> H["wrapper output<br/>[2, 6, 4]"]
    H --> I["Integrated MultiHeadAttention"]
    I --> J["Q/K/V projection"]
    J --> K["split into heads"]
    K --> L["scaled dot-product attention"]
    L --> M["merge heads + out_proj"]
```

## CausalAttention 내부

```mermaid
flowchart TD
    X["Input x<br/>[batch, tokens, d_in]"] --> Q["W_query(x)<br/>queries"]
    X --> K["W_key(x)<br/>keys"]
    X --> V["W_value(x)<br/>values"]
    Q --> S["Attention scores<br/>queries @ keys.T"]
    K --> S
    S --> Scale["Scale<br/>divide by sqrt(d_k)"]
    Scale --> Mask["Causal mask<br/>future positions = -inf"]
    Mask --> Softmax["Softmax<br/>attention weights"]
    Softmax --> Dropout["Dropout"]
    Dropout --> Context["Context vectors<br/>weights @ values"]
    V --> Context
```

## Causal Mask

```mermaid
flowchart LR
    T0["Token 0"] --> V0["볼 수 있음<br/>Token 0"]
    T1["Token 1"] --> V1["볼 수 있음<br/>Token 0, Token 1"]
    T2["Token 2"] --> V2["볼 수 있음<br/>Token 0, Token 1, Token 2"]
    Tn["Token n"] --> Vn["볼 수 있음<br/>Token 0..n"]
    Future["미래 token"] -. "masked" .-> Zero["softmax 후 weight 0"]
```

## Multi-Head 비교

```mermaid
flowchart TD
    X["Input batch<br/>[2, 6, 3]"] --> H1["Head 1<br/>[2, 6, 2]"]
    X --> H2["Head 2<br/>[2, 6, 2]"]
    H1 --> Cat["Wrapper 방식<br/>torch.cat"]
    H2 --> Cat
    Cat --> WOut["[2, 6, 4]"]

    X --> Proj["Integrated 방식<br/>single module projections"]
    Proj --> Split["view + transpose<br/>head dimension 분리"]
    Split --> Attn["attention per head"]
    Attn --> Merge["transpose + contiguous + view"]
    Merge --> IOut["[2, 6, 2]"]
```

## Shape Summary

```mermaid
flowchart LR
    A["inputs<br/>[6, 3]"] --> B["batch<br/>[2, 6, 3]"]
    B --> C["Q/K/V<br/>[2, 6, 2]"]
    C --> D["scores<br/>[2, 6, 6]"]
    D --> E["weights<br/>[2, 6, 6]"]
    E --> F["context<br/>[2, 6, 2]"]
```

