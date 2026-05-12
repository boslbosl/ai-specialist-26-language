# Chapter 2 Dataset Diagrams

- 관련 문서: [../Chapter_2_Exercise_Dataset.md](../Chapter_2_Exercise_Dataset.md)
- 관련 노트북: [../../Chapter_2_Exercise_Dataset.ipynb](../../Chapter_2_Exercise_Dataset.ipynb)

## 전체 흐름

```mermaid
flowchart TD
    A["원문 텍스트<br/>datas/the-verdict.txt"] --> B["Tokenizer"]
    B --> C["Token IDs<br/>[total_tokens]"]
    C --> D["Sliding window"]
    D --> E["Input x<br/>[context_length]"]
    D --> F["Target y<br/>x를 한 칸 shift"]
    E --> G["GPTDatasetV1.__getitem__"]
    F --> G
    G --> H["DataLoader"]
    H --> I["Batch x, y<br/>[batch_size, context_length]"]
```

## Next-Token Dataset

```mermaid
flowchart LR
    T["Token sequence<br/>t0 t1 t2 t3 t4"] --> X["input x<br/>t0 t1 t2 t3"]
    T --> Y["target y<br/>t1 t2 t3 t4"]
    X --> L["모델 입력"]
    Y --> L
    L --> O["각 위치에서 다음 token 예측"]
```

## DataLoader 역할

```mermaid
flowchart TD
    A["Dataset sample 0<br/>x0, y0"] --> D["DataLoader collate"]
    B["Dataset sample 1<br/>x1, y1"] --> D
    C["Dataset sample n<br/>xn, yn"] --> D
    D --> E["x_batch<br/>[batch, context_length]"]
    D --> F["y_batch<br/>[batch, context_length]"]
```

## Tokenizer 비교

```mermaid
flowchart LR
    A["같은 문장"] --> B["Tokenizer A"]
    A --> C["Tokenizer B"]
    B --> D["긴 token sequence"]
    C --> E["짧은 token sequence"]
    D --> F["언어/도메인에 따라 효율 차이"]
    E --> F
```

