# Chapter 7 Instruction Fine-Tuning Diagrams

- 관련 문서: [../Chapter_7_Exercise_Follow_Instructions.md](../Chapter_7_Exercise_Follow_Instructions.md)
- 관련 노트북: [../../Chapter_7_Exercise_Follow_Instructions.ipynb](../../Chapter_7_Exercise_Follow_Instructions.ipynb)

## 전체 흐름

```mermaid
flowchart TD
    A["Instruction data<br/>instruction/input/output"] --> B["Prompt template"]
    B --> C["Formatted prompt + response"]
    C --> D["Tokenizer"]
    D --> E["input_ids"]
    D --> F["labels"]
    F --> G["Mask prompt/pad labels<br/>ignore_index = -100"]
    E --> H["Causal LM forward"]
    G --> I["Response-only loss"]
    H --> I
    I --> J["Supervised fine-tuning"]
```

## Prompt Formatting

```mermaid
flowchart LR
    A["instruction"] --> T["Template"]
    B["optional input"] --> T
    C["expected response"] --> T
    T --> D["### Instruction<br/>...<br/>### Input<br/>...<br/>### Response<br/>..."]
```

## Label Masking

```mermaid
flowchart TD
    A["Full token sequence"] --> B["Prompt tokens"]
    A --> C["Response tokens"]
    A --> D["Padding tokens"]
    B --> E["labels = -100<br/>loss 제외"]
    D --> E
    C --> F["labels = response token id<br/>loss 계산"]
    E --> G["Causal LM loss"]
    F --> G
```

## 생성 평가

```mermaid
flowchart LR
    A["새 instruction"] --> B["same prompt template"]
    B --> C["model.generate"]
    C --> D["decode response"]
    D --> E["지시 준수 여부 확인"]
```

