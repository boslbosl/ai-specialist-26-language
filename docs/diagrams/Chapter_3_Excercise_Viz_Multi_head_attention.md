# Chapter 3 Multi-Head Attention Visualization Diagrams

- 관련 문서: [../Chapter_3_Excercise_Viz_Multi_head_attention.md](../Chapter_3_Excercise_Viz_Multi_head_attention.md)
- 관련 노트북: [../../Chapter_3_Excercise_Viz_Multi_head_attention.ipynb](../../Chapter_3_Excercise_Viz_Multi_head_attention.ipynb)

## 시각화 목표

```mermaid
flowchart TD
    A["Attention tensor"] --> B["Layer 선택"]
    B --> C["Head 선택"]
    C --> D["Token-token matrix"]
    D --> E["Heatmap"]
    E --> F["어떤 token을 강하게 참고하는지 해석"]
```

## 축 해석

```mermaid
flowchart LR
    A["attention<br/>[layers, heads, query_tokens, key_tokens]"] --> B["layer index"]
    A --> C["head index"]
    A --> D["row: query token"]
    A --> E["column: key token"]
    D --> F["현재 token이"]
    E --> G["어떤 과거 token을 보는가"]
    F --> H["attention weight"]
    G --> H
```

## Head별 패턴 비교

```mermaid
flowchart TD
    P["같은 prompt"] --> H1["Head 1<br/>가까운 이전 token 집중"]
    P --> H2["Head 2<br/>구문 관계 집중"]
    P --> H3["Head 3<br/>특정 구분자 집중"]
    H1 --> C["Head마다 다른 관계를 학습할 수 있음"]
    H2 --> C
    H3 --> C
```

## 읽는 순서

```mermaid
flowchart LR
    A["prompt 확인"] --> B["token label 확인"]
    B --> C["layer/head 선택"]
    C --> D["heatmap row 하나 선택"]
    D --> E["가장 진한 column 해석"]
    E --> F["causal mask로 미래 영역이 비어 있는지 확인"]
```

