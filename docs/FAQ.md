# LLM Hands-on FAQ

이 문서는 프로젝트 대화에서 반복적으로 나온 질문을 강의 노트와 연결한 FAQ입니다.
짧은 답을 먼저 보고, 더 자세한 맥락은 관련 lecture 문서를 따라가면 됩니다.

## Chapter 2. Tokenizer와 Dataset

### `tiktoken`에서 `tik`은 무슨 뜻인가?

`tiktoken`은 OpenAI 계열 tokenizer 라이브러리 이름입니다. 공개 문서상 `tik`이 특정 약자의 의미라고 명확히 설명된 것은 아니므로, 실무적으로는 `token` 앞에 붙은 고유 패키지 이름으로 보면 됩니다.

관련 문서: [Chap2 tokenizer/dataset](../lectures/chap2_tokenizer_dataset.md)

### `stride`는 무슨 뜻인가?

`stride`는 긴 token sequence에서 다음 학습 샘플을 만들 때 몇 token만큼 이동할지를 뜻합니다.

```text
tokens = [0, 1, 2, 3, 4, 5, 6, 7]
max_length = 4
stride = 4

sample 1 = [0, 1, 2, 3]
sample 2 = [4, 5, 6, 7]
```

`stride = max_length`이면 겹치지 않고, `stride < max_length`이면 샘플끼리 일부 token이 겹칩니다.

관련 문서: [Chap2 tokenizer/dataset](../lectures/chap2_tokenizer_dataset.md)

### `drop_last`는 왜 켜거나 끄나?

`drop_last=True`는 마지막 batch가 `batch_size`보다 작을 때 버립니다. 학습에서는 batch shape를 일정하게 유지하려고 자주 씁니다.

`drop_last=False`는 마지막 작은 batch도 사용합니다. 검증/평가에서는 데이터를 버리지 않는 것이 중요하므로 보통 `False`를 씁니다.

관련 문서: [Chap5 pretraining](../lectures/chap5_pretraining.md)

## Chapter 4. GPT Model

### `model.eval()`은 무엇을 하나?

`model.eval()`은 모델을 평가/추론 모드로 바꿉니다. 대표적으로 `Dropout`이 꺼지고, `BatchNorm`은 학습 중 저장한 통계를 사용합니다. GPT 예제에서는 주로 dropout을 끄는 의미가 큽니다.

```python
model.train()  # 학습 모드
model.eval()   # 평가/추론 모드
```

단, `model.eval()`은 gradient 계산을 끄지 않습니다.

관련 문서: [Chap4 GPT model](../lectures/chap4_gpt_model.md)

### `with torch.no_grad():`를 쓰는 이유는?

`torch.no_grad()`는 해당 블록 안에서 계산 그래프를 만들지 않게 합니다. 추론에서는 파라미터 업데이트가 필요 없으므로 메모리와 계산을 줄일 수 있습니다.

```python
model.eval()
with torch.no_grad():
    logits = model(input_ids)
```

`with` 블록을 쓰는 이유는 블록이 끝나면 gradient 설정이 자동으로 원래 상태로 돌아오기 때문입니다.

관련 문서: [Chap4 GPT model](../lectures/chap4_gpt_model.md), [Chap5 pretraining](../lectures/chap5_pretraining.md)

### 추론 때도 gradient가 필요한 경우가 있나?

일반 생성에서는 필요 없습니다. 하지만 saliency 분석, adversarial example 생성, 입력 임베딩 최적화, gradient 기반 해석 실험에서는 추론 형태의 forward를 하면서도 gradient가 필요할 수 있습니다.

관련 문서: [Chap5 pretraining](../lectures/chap5_pretraining.md)

### 예시에서 dropout을 끄고 직접 생성 loop를 수행하는 이유는?

Chapter 4의 생성 예시는 좋은 문장을 만들기보다 GPT의 autoregressive loop를 보여주는 목적입니다.

```text
input token IDs
=> model
=> logits
=> 마지막 위치 logits 선택
=> 다음 token 선택
=> 입력 뒤에 붙이고 반복
```

dropout이 켜져 있으면 같은 입력에서도 일부 activation이 랜덤하게 꺼져 결과가 흔들립니다. 그래서 `model.eval()`로 dropout을 끄고 구조의 흐름을 확인합니다.

관련 문서: [Chap4 GPT model](../lectures/chap4_gpt_model.md)

### `torch.arange`는 무엇인가?

`torch.arange(seq_len)`은 `0`부터 `seq_len - 1`까지의 정수 tensor를 만듭니다.

```python
torch.arange(4)
# tensor([0, 1, 2, 3])
```

GPT에서는 위치 embedding을 만들 때 자주 씁니다.

```python
pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
```

`device=in_idx.device`는 CPU/GPU 위치를 입력 tensor와 맞추기 위한 옵션입니다.

관련 문서: [Chap4 GPT model](../lectures/chap4_gpt_model.md)

### `idx_cond`에서 `cond`는 무슨 뜻인가?

`cond`는 보통 `condition` 또는 `conditioning`의 줄임말입니다.

```python
idx_cond = idx[:, -context_size:]
```

뜻은 "다음 token 예측의 조건으로 사용할 token indices"입니다. 생성 중 문맥이 너무 길어지면 모델의 context length에 맞게 최근 token만 잘라서 넣습니다.

관련 문서: [Chap4 GPT model](../lectures/chap4_gpt_model.md)

### Transformer block 안의 `emb_dim -> 4 * emb_dim -> GELU -> emb_dim`은 원래 구조인가?

네. 학습용 임시 구조가 아니라 GPT/Transformer block의 일반적인 FFN/MLP 구조입니다.

```text
[batch, seq_len, emb_dim]
=> Linear: emb_dim -> 4 * emb_dim
=> GELU
=> Linear: 4 * emb_dim -> emb_dim
=> [batch, seq_len, emb_dim]
```

내부에서만 넓어졌다가 다시 원래 차원으로 돌아옵니다. 그래야 residual connection에서 원래 입력과 더할 수 있습니다.

관련 문서: [Chap4 GPT model](../lectures/chap4_gpt_model.md)

## Chapter 5. Pretraining

### Cross entropy는 무엇이고 왜 쓰나?

Cross entropy는 모델이 정답 class/token에 얼마나 낮은 확률을 줬는지 벌점으로 계산하는 loss입니다.

```text
loss = -log(정답 token에 준 확률)
```

정답 token 확률이 높으면 loss가 작고, 낮으면 loss가 큽니다. GPT pretraining에서는 각 위치마다 다음 token을 맞히도록 cross entropy를 씁니다.

관련 문서: [Chap5 pretraining](../lectures/chap5_pretraining.md)

### `calc_loss_batch`에서 왜 `flatten(0, 1)`을 하나?

GPT 출력은 보통 다음 shape입니다.

```text
logits:  [batch, seq_len, vocab_size]
targets: [batch, seq_len]
```

PyTorch `cross_entropy`는 예측을 `[N, C]`, 정답을 `[N]` 형태로 받는 경우가 많습니다. 그래서 batch 차원과 sequence 차원을 합칩니다.

```python
loss = torch.nn.functional.cross_entropy(
    logits.flatten(0, 1),
    target_batch.flatten()
)
```

결과 shape는 다음과 같습니다.

```text
logits:  [batch * seq_len, vocab_size]
targets: [batch * seq_len]
```

관련 문서: [Chap5 pretraining](../lectures/chap5_pretraining.md)

### `.pth` 파일은 무엇인가?

`.pth`는 PyTorch 모델 가중치나 checkpoint 저장 파일로 자주 쓰이는 확장자입니다.

```python
torch.save(model.state_dict(), "model.pth")
```

실무적으로는 "PyTorch checkpoint 파일"이라고 이해하면 됩니다. `.pt`도 비슷하게 쓰입니다.

관련 문서: [Chap5 pretraining](../lectures/chap5_pretraining.md), [Chap6 fine-tuning/LoRA](../lectures/chap6_finetuning_lora.md)

## Chapter 6. Classification Fine-tuning과 LoRA

### 언더샘플링은 무엇이고 왜 쓰나?

언더샘플링은 많은 쪽 class 데이터를 일부러 줄여서 class 균형을 맞추는 방법입니다. 스팸 분류 예제에서는 보통 정상 메일 `ham`이 `spam`보다 많으므로, `ham`을 `spam` 개수만큼만 샘플링합니다.

```python
num_spam = df[df["Label"] == "spam"].shape[0]
ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
```

이렇게 하면 모델이 다수 class만 찍어서 높은 accuracy를 얻는 문제를 줄일 수 있습니다.

관련 문서: [Chap6 fine-tuning/LoRA](../lectures/chap6_finetuning_lora.md)

### LoRA의 A/B 행렬은 왜 보통 `A=random`, `B=zero`로 초기화하나?

LoRA는 원래 weight `W`에 작은 low-rank update를 더합니다.

```text
W' = W + delta_W
delta_W = B @ A
```

`B=0`이면 처음에는 `delta_W=0`이므로 pretrained model의 출력을 바꾸지 않습니다. 하지만 `A`가 random이면 `B` 쪽에는 gradient가 흐를 수 있고, 첫 update 이후에는 `A`도 함께 조정됩니다.

관련 문서: [Chap6 fine-tuning/LoRA](../lectures/chap6_finetuning_lora.md)

### LoRA 적용과 classification `out_head` 교체 순서는 어떻게 잡아야 하나?

의도에 따라 다릅니다.

분류 head까지 LoRA로 감싸고 싶다면:

```python
model.out_head = torch.nn.Linear(768, 2)
freeze_all_params(model)
replace_linear_with_lora(model, rank, alpha)
```

분류 head는 일반 Linear로 직접 학습하고, backbone만 LoRA로 학습하고 싶다면:

```python
freeze_all_params(model)
replace_linear_with_lora(model, rank, alpha)
model.out_head = torch.nn.Linear(768, 2)
```

분류 fine-tuning에서는 두 번째 방식이 더 직관적인 경우가 많습니다.

관련 문서: [Chap6 fine-tuning/LoRA](../lectures/chap6_finetuning_lora.md)

## Chapter 7. Instruction Tuning과 DPO

### Chapter 7에서 JSON의 `input`과 학습 tensor의 `inputs`는 어떻게 다른가?

JSON 데이터의 `input`은 instruction을 수행하는 데 필요한 추가 문맥입니다.

```text
instruction = 해야 할 일
input       = 필요한 문장/자료
output      = 기대 답변
```

학습 tensor의 `inputs`는 모델에 들어가는 token sequence입니다. `targets`는 `inputs`를 한 칸 오른쪽으로 민 정답 sequence입니다.

```text
tokens:  [10, 20, 30, 40, 50]
inputs:  [10, 20, 30, 40]
targets: [20, 30, 40, 50]
```

관련 문서: [Chap7 instruction/DPO](../lectures/chap7_instruction_dpo.md)

### SFT에서 `labels = [-100, -100, ...]`는 어디에 넣나?

`targets`를 만든 직후, loss 계산 전에 prompt 영역을 `-100`으로 바꿉니다.

```python
inputs = torch.tensor(padded[:-1])
targets = torch.tensor(padded[1:])

targets[:prompt_len - 1] = ignore_index
```

`-100`은 PyTorch cross entropy에서 해당 위치를 loss 계산에서 무시하라는 뜻입니다. `prompt_len - 1`을 쓰는 이유는 `targets`가 한 칸 shift되어 있기 때문입니다.

관련 문서: [Chap7 instruction/DPO](../lectures/chap7_instruction_dpo.md)

### DPO에서도 `-100` labels를 직접 넣어야 하나?

보통 DPO에서는 `-100` labels 대신 `chosen_mask`, `rejected_mask` 같은 selection mask를 사용합니다. prompt와 padding 위치를 mask에서 `False`로 만들고, response token log probability만 평균/합산합니다.

```text
SFT: labels에 -100을 넣어 loss에서 제외
DPO: mask=False로 logprob 계산에서 제외
```

관련 문서: [Chap7 instruction/DPO](../lectures/chap7_instruction_dpo.md)

### Chapter 7 SFT 예시에서 주의할 로직은?

Instruction tuning 설명은 보통 "prompt는 조건, response는 학습 대상"입니다. 따라서 response token에만 loss를 거는 것이 자연스럽습니다.

만약 collate 함수가 padding만 `-100`으로 바꾸고 prompt는 그대로 둔다면, prompt token도 다음 token 예측 대상으로 학습됩니다. 실행은 되지만 "response만 학습"이라는 설명과는 다릅니다.

관련 문서: [Chap7 instruction/DPO](../lectures/chap7_instruction_dpo.md)

## Architecture Optimization

### MLA에서 RoPE는 무엇이고 왜 붙나?

RoPE는 Rotary Positional Embedding입니다. 위치 정보를 token embedding에 더하는 대신, Q/K 벡터를 위치에 따라 회전시켜 attention score 안에 위치 관계를 넣습니다.

```text
Q_pos = rotate(Q, position)
K_pos = rotate(K, position)
score = Q_pos @ K_pos
```

MLA는 KV cache를 줄이기 위해 K/V 정보를 latent로 압축합니다. 이때 위치 정보까지 부정확하게 압축하면 attention이 순서를 제대로 보기 어려우므로, RoPE 위치 정보는 Q/K attention 계산에 명확히 남겨야 합니다.

관련 문서: [Chap3 attention](../lectures/chap3_attention.md)

### 앞으로 더 나올 수 있는 LLM 최적화 방향은?

핵심 방향은 "모든 token을 모든 layer에서 같은 방식으로 처리하지 않는 것"입니다.

가능한 연구 방향:

- 쉬운 token은 얕게, 어려운 token은 깊게 처리하는 동적 깊이
- 오래된 token은 압축하고 중요한 token만 보존하는 동적 attention/cache
- token 단위가 아니라 의미 chunk 단위로 KV cache를 압축하는 방식
- 문제 난이도에 따라 계산 예산을 자동 조절하는 routing
- SSM, attention, MoE, memory module을 역할별로 섞는 hybrid architecture
- 여러 token을 한 번에 draft/verify하는 parallel 또는 speculative decoding

관련 문서: [Chap3 attention](../lectures/chap3_attention.md), [Chap6 fine-tuning/LoRA](../lectures/chap6_finetuning_lora.md), [Chap7 instruction/DPO](../lectures/chap7_instruction_dpo.md)
