# Chapter 7 DPO Diagrams

- 관련 문서: [../Chapter_7_Exercise_Follow_Instructions_dpo.md](../Chapter_7_Exercise_Follow_Instructions_dpo.md)
- 관련 노트북: [../../Chapter_7_Exercise_Follow_Instructions_dpo.ipynb](../../Chapter_7_Exercise_Follow_Instructions_dpo.ipynb)

## 전체 흐름

```mermaid
flowchart TD
    A["Preference data<br/>prompt, chosen, rejected"] --> B["Format prompt"]
    B --> C["Tokenize chosen sequence"]
    B --> D["Tokenize rejected sequence"]
    C --> E["Policy model logprobs"]
    D --> E
    C --> F["Reference model logprobs"]
    D --> F
    E --> G["DPO loss"]
    F --> G
    G --> H["Update policy model only"]
```

## Policy와 Reference

```mermaid
flowchart LR
    A["SFT model checkpoint"] --> P["Policy model<br/>trainable"]
    A --> R["Reference model<br/>frozen"]
    P --> L["chosen/rejected logprob 비교"]
    R --> L
    L --> D["DPO objective"]
```

## Chosen / Rejected Pair

```mermaid
flowchart TD
    Prompt["Prompt"] --> Chosen["Chosen response<br/>선호 응답"]
    Prompt --> Rejected["Rejected response<br/>비선호 응답"]
    Chosen --> CLog["chosen logprob"]
    Rejected --> RLog["rejected logprob"]
    CLog --> Loss["chosen이 rejected보다 높아지도록 학습"]
    RLog --> Loss
```

## Masking과 Log Probability

```mermaid
flowchart TD
    A["prompt + response tokens"] --> B["prompt mask = False"]
    A --> C["response mask = True"]
    B --> D["질문 token은 loss 비교에서 제외"]
    C --> E["응답 token logprob만 합산"]
    E --> F["DPO loss 계산"]
```

## 학습 루프

```mermaid
flowchart LR
    A["batch"] --> B["compute_dpo_loss_batch"]
    B --> C["loss.backward"]
    C --> D["optimizer.step"]
    D --> E["evaluate_dpo_loss_loader"]
    E --> F["sample generation 비교"]
```

