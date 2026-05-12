# Chapter 6 LoRA Fine-Tuning Diagrams

- 관련 문서: [../Chapter_6_Excercise_Finetuning_Classification_LoRA.md](../Chapter_6_Excercise_Finetuning_Classification_LoRA.md)
- 관련 노트북: [../../Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb](../../Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb)

## 전체 흐름

```mermaid
flowchart TD
    A["Pretrained GPT"] --> B["Freeze base weights"]
    B --> C["Replace target Linear layers"]
    C --> D["Add LoRA A/B matrices"]
    D --> E["Train only LoRA parameters"]
    E --> F["Classification evaluation"]
```

## LoRA Linear

```mermaid
flowchart LR
    X["input x"] --> Base["Frozen base Linear<br/>xW"]
    X --> A["LoRA A<br/>in_dim -> rank"]
    A --> B["LoRA B<br/>rank -> out_dim"]
    B --> Scale["scale<br/>alpha / rank"]
    Base --> Sum["base output + LoRA update"]
    Scale --> Sum
    Sum --> Y["output"]
```

## 학습 가능한 파라미터

```mermaid
flowchart TD
    M["Model parameters"] --> Frozen["Frozen<br/>requires_grad=False"]
    M --> Trainable["Trainable<br/>requires_grad=True"]
    Frozen --> F1["embedding"]
    Frozen --> F2["attention/FFN base weights"]
    Trainable --> T1["LoRA A/B"]
    Trainable --> T2["classification head if configured"]
    T1 --> Opt["optimizer parameter list"]
    T2 --> Opt
```

## Full Fine-Tuning vs LoRA

```mermaid
flowchart LR
    A["Full fine-tuning"] --> B["모든 weight 업데이트"]
    B --> C["메모리와 저장 비용 큼"]
    D["LoRA"] --> E["작은 low-rank update만 학습"]
    E --> F["task별 adapter 저장 가능"]
```

