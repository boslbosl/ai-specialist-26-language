# Chap5. 언어 모델 사전학습

참고 자료: `../pdf/AI-Application-Specialist-LLM-20260420_part2.pdf` 중 슬라이드 32-50 전후

## 이 장을 왜 배우는가

Chap4까지는 GPT 구조와 생성 방법을 봤습니다. Chap5는 그 구조가 어떻게 학습되는지 다룹니다.
LLM의 사전학습은 대량의 텍스트에서 다음 토큰을 예측하는 문제를 반복하며,
모델 파라미터가 언어 패턴, 사실적 연관, 문법, 문체를 흡수하도록 만드는 과정입니다.

이 장은 loss, optimizer, batch, epoch, train/validation split 같은 학습 루프의 기본 단위를 정리합니다.

## 한눈에 보는 흐름

```mermaid
flowchart LR
    Text["대량 텍스트"] --> Dataset["Input/Target dataset"]
    Dataset --> Batch["Mini-batch"]
    Batch --> Model["GPT forward"]
    Model --> Logits["Logits"]
    Logits --> Loss["Cross entropy loss"]
    Loss --> Backward["Backward"]
    Backward --> Optim["Optimizer step"]
    Optim --> Eval["Validation loss 확인"]
```

## 학습 목표

- 사전학습이 다음 토큰 예측 문제임을 설명할 수 있다.
- train/validation split이 왜 필요한지 이해한다.
- cross entropy loss가 language modeling에서 어떻게 계산되는지 이해한다.
- batch size, step, epoch의 차이를 설명할 수 있다.
- gradient descent와 Adam optimizer의 역할을 이해한다.
- inference와 training의 차이를 구분할 수 있다.

## 1. 사전학습은 "다음 토큰 맞히기"를 대규모로 반복한다

언어 모델의 사전학습 데이터는 보통 다음처럼 만들어집니다.

```text
input : "The cat sat on"
target: "cat sat on the"
```

모델은 각 위치에서 다음 토큰을 예측합니다.
정답 토큰의 확률이 높아지도록 loss를 줄이면,
모델은 자연스럽게 문장 구조와 단어 관계를 학습합니다.

중요한 점은 사람이 각 문장에 별도 label을 붙이지 않아도 된다는 것입니다.
원문 텍스트에서 input/target을 자동으로 만들 수 있기 때문에 self-supervised learning입니다.

## 2. Train/validation split은 외운 것과 일반화한 것을 구분한다

학습 데이터의 loss만 보면 모델이 진짜 잘 배우는지 알기 어렵습니다.
모델이 훈련 데이터 문장을 외우고 있을 수도 있기 때문입니다.

그래서 데이터를 보통 두 부분으로 나눕니다.

| 구분 | 용도 |
|---|---|
| Training set | 파라미터 업데이트에 사용 |
| Validation set | 학습 중 성능 점검에 사용 |

Validation loss가 계속 나빠지는데 training loss만 좋아진다면 overfitting을 의심할 수 있습니다.

## 3. Cross entropy loss는 정답 토큰의 확률을 높인다

모델 출력 shape는 보통 다음과 같습니다.

```text
logits:  [batch, seq_len, vocab_size]
targets: [batch, seq_len]
```

각 위치마다 vocabulary 전체에 대한 점수가 있고,
target에는 정답 token ID가 들어 있습니다.

PyTorch의 `F.cross_entropy`는 일반적으로 class 차원이 두 번째에 오기를 기대하므로,
언어 모델에서는 flatten해서 계산하는 경우가 많습니다.

```text
logits_flat:  [batch * seq_len, vocab_size]
targets_flat: [batch * seq_len]
loss: scalar
```

이 loss 하나가 batch 전체의 평균적인 예측 오류를 나타냅니다.

## 4. Cost function과 loss의 차이

실무에서는 loss와 cost를 자주 섞어 말하지만, 구분하면 다음과 같습니다.

- loss: 하나의 샘플 또는 batch에서의 오류
- cost/objective: 전체 데이터나 학습 목표 관점에서 최적화하려는 함수

코드에서는 보통 `loss`라는 scalar를 만들고, 그 값을 기준으로 gradient를 계산합니다.

## 5. Gradient descent는 loss를 줄이는 방향으로 파라미터를 움직인다

모델 파라미터는 처음에는 무작위에 가깝습니다.
학습은 다음을 반복합니다.

```text
1. 현재 파라미터로 예측한다.
2. loss를 계산한다.
3. loss가 각 파라미터에 대해 어떻게 변하는지 gradient를 계산한다.
4. gradient 반대 방향으로 파라미터를 조금 이동한다.
```

이 "조금"의 크기를 learning rate라고 합니다.
너무 크면 발산할 수 있고, 너무 작으면 학습이 지나치게 느립니다.

## 6. Adam optimizer는 gradient descent를 더 똑똑하게 만든다

기본 gradient descent는 모든 파라미터를 같은 방식으로 업데이트합니다.
Adam은 gradient의 이동 평균과 제곱 평균을 사용해 파라미터별 업데이트 크기를 조절합니다.

직관적으로는 다음과 같습니다.

- 최근 gradient 방향을 어느 정도 기억한다.
- gradient 크기의 변동성을 고려한다.
- sparse하거나 noisy한 gradient에서도 비교적 안정적으로 작동한다.

LLM 학습과 fine-tuning에서 AdamW 계열 optimizer가 자주 사용됩니다.

## 7. Epoch, step, batch size

| 용어 | 의미 |
|---|---|
| Batch size | 한 번의 forward/backward에 넣는 샘플 수 |
| Step | optimizer가 한 번 업데이트되는 단위 |
| Epoch | training dataset 전체를 한 번 훑은 단위 |

대규모 LLM에서는 dataset이 매우 크기 때문에 epoch보다 step 수를 기준으로 학습량을 말하는 경우도 많습니다.

## 8. Batch size trade-off

Batch size가 크면 gradient 추정이 안정적이고 GPU를 효율적으로 사용할 수 있습니다.
하지만 메모리를 많이 쓰고, 너무 큰 batch는 일반화에 영향을 줄 수 있습니다.

Batch size가 작으면 메모리 부담이 적고 업데이트가 자주 일어나지만,
gradient가 noisy해질 수 있습니다.

메모리가 부족할 때는 gradient accumulation으로 여러 작은 batch의 gradient를 모아
큰 batch처럼 업데이트할 수 있습니다.

## 9. Training과 inference의 차이

| 구분 | Training | Inference |
|---|---|---|
| 목적 | 파라미터 업데이트 | 예측 또는 생성 |
| gradient | 필요 | 보통 불필요 |
| dropout | 켜짐 | 꺼짐 |
| loss | 계산 | 선택적 |
| 속도/메모리 | 무거움 | 상대적으로 가벼움 |

PyTorch에서는 inference 때 보통 다음을 사용합니다.

```python
model.eval()
with torch.no_grad():
    logits = model(input_ids)
```

## 코드 관점에서 꼭 볼 것

사전학습 루프는 다음 구조로 읽으면 됩니다.

```python
for epoch in range(num_epochs):
    model.train()
    for input_batch, target_batch in train_loader:
        logits = model(input_batch)
        loss = calc_loss(logits, target_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = evaluate(model, val_loader)
```

확인 포인트:

- `input_batch`와 `target_batch` shape가 같은가?
- `logits`의 마지막 차원이 `vocab_size`인가?
- loss가 scalar인가?
- validation은 `torch.no_grad()`에서 계산되는가?
- optimizer step 전에 `zero_grad()`가 호출되는가?

## 자주 헷갈리는 지점

1. Training loss가 낮다고 항상 좋은 모델은 아닙니다. Validation loss도 함께 봐야 합니다.
2. Epoch는 데이터셋 크기에 의존합니다. 대규모 말뭉치에서는 step 기준이 더 현실적일 수 있습니다.
3. Inference에서는 gradient가 필요 없으므로 `torch.no_grad()`를 사용해야 메모리를 아낄 수 있습니다.
4. Cross entropy는 softmax를 내부적으로 포함하는 구현이 많으므로, logits에 softmax를 먼저 적용하면 안 되는 경우가 많습니다.

## 수업 중 질문

- validation set을 training에 사용하면 어떤 문제가 생길까?
- batch size를 두 배로 키우면 learning rate도 반드시 두 배로 키워야 할까?
- loss가 감소하지 않을 때 의심할 수 있는 원인은 무엇일까?
- `model.eval()`을 하지 않고 생성하면 어떤 차이가 생길 수 있을까?

## 이 장을 배우고 나면

GPT가 어떻게 사전학습되는지, 학습 루프에서 loss와 optimizer가 어떤 역할을 하는지 이해할 수 있습니다.
또한 이후 Chap6의 fine-tuning이 사전학습과 무엇이 같고 무엇이 다른지 비교할 준비가 됩니다.

## 관련 FAQ

- [Cross entropy는 무엇이고 왜 쓰나?](../docs/FAQ.md#cross-entropy는-무엇이고-왜-쓰나)
- [`calc_loss_batch`에서 왜 `flatten(0, 1)`을 하나?](../docs/FAQ.md#calc_loss_batch에서-왜-flatten0-1을-하나)
- [`.pth` 파일은 무엇인가?](../docs/FAQ.md#pth-파일은-무엇인가)
- [`with torch.no_grad():`를 쓰는 이유는?](../docs/FAQ.md#with-torchnograd를-쓰는-이유는)
