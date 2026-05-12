# Chapter 6 Classification Fine-Tuning Diagrams

- 관련 문서: [../Chapter_6_Excercise_Finetuning_Classification.md](../Chapter_6_Excercise_Finetuning_Classification.md)
- 관련 노트북: [../../Chapter_6_Excercise_Finetuning_Classification.ipynb](../../Chapter_6_Excercise_Finetuning_Classification.ipynb)

## 전체 흐름

```mermaid
flowchart TD
    A["CSV dataset<br/>text + label"] --> B["Tokenizer"]
    B --> C["Padding"]
    C --> D["attention mask"]
    D --> E["DataLoader batch"]
    E --> F["GPT backbone"]
    F --> G["Last valid token hidden state"]
    G --> H["Classification head"]
    H --> I["Class logits<br/>[batch, num_classes]"]
    I --> J["Cross entropy loss"]
    J --> K["optimizer step"]
```

## Padding과 마지막 유효 Token

```mermaid
flowchart LR
    A["sample A<br/>real real real real"] --> C["batch"]
    B["sample B<br/>real real PAD PAD"] --> C
    C --> D["attention mask"]
    D --> E["마지막 유효 token 위치 찾기"]
    E --> F["문장 대표 hidden state"]
```

## 분류 Head

```mermaid
flowchart TD
    A["hidden states<br/>[batch, seq_len, emb_dim]"] --> B["select last valid state"]
    B --> C["sentence vector<br/>[batch, emb_dim]"]
    C --> D["Linear<br/>emb_dim -> num_classes"]
    D --> E["class logits<br/>[batch, num_classes]"]
    E --> F["argmax"]
    F --> G["predicted label"]
```

## 평가 흐름

```mermaid
flowchart LR
    A["validation/test loader"] --> B["model.eval"]
    B --> C["no_grad"]
    C --> D["logits"]
    D --> E["pred = argmax"]
    E --> F["compare with labels"]
    F --> G["accuracy"]
```

