# Chapter 5 Pretraining Diagrams

- 관련 문서: [../Chapter_5_Excercise_Pretraining.md](../Chapter_5_Excercise_Pretraining.md)
- 관련 노트북: [../../Chapter_5_Excercise_Pretraining.ipynb](../../Chapter_5_Excercise_Pretraining.ipynb)

## 전체 흐름

```mermaid
flowchart TD
    A["Text corpus"] --> B["Tokenizer"]
    B --> C["GPTDatasetV1"]
    C --> D["Train DataLoader"]
    C --> E["Validation DataLoader"]
    D --> F["GPTModel forward"]
    F --> G["Logits"]
    G --> H["Cross entropy loss"]
    H --> I["loss.backward"]
    I --> J["optimizer.step"]
    E --> K["evaluate_model"]
    K --> L["validation loss"]
    J --> M["sample generation"]
```

## Training Step

```mermaid
flowchart LR
    A["input_batch<br/>[batch, seq_len]"] --> B["model(input_batch)"]
    B --> C["logits<br/>[batch, seq_len, vocab_size]"]
    D["target_batch<br/>[batch, seq_len]"] --> E["calc_loss_batch"]
    C --> E
    E --> F["loss scalar"]
    F --> G["zero_grad"]
    G --> H["backward"]
    H --> I["optimizer.step"]
```

## Evaluation Step

```mermaid
flowchart TD
    A["model.eval"] --> B["torch.no_grad"]
    B --> C["loop over eval batches"]
    C --> D["calculate loss only"]
    D --> E["average loss"]
    E --> F["model.train"]
```

## Loss와 생성의 관계

```mermaid
flowchart LR
    A["낮은 training loss"] --> B["훈련 데이터 다음 token 예측 개선"]
    C["낮은 validation loss"] --> D["보지 않은 데이터 일반화 개선"]
    B --> E["sample generation 확인"]
    D --> E
    E --> F["숫자 loss와 실제 문장 품질 함께 관찰"]
```

