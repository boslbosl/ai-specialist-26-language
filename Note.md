# 실습 IPYNB 공부노트

범위: `ai-specialist-2026-quiz/`, `llm_hands_on/llm_from_scratch/` 제외.

정리 방식:
- `answer` 노트북과 빈칸 실습 노트북은 같은 주제로 묶었다.
- `Chapter_2_Exercise_Dataset.org.ipynb`, `Task_1-Copy1.ipynb`, `11_SegFormer_torch.ipynb`처럼 내용이 겹치는 파일은 중복 파일로 표시했다.
- RAG 노트북은 직접 NN을 구현하지 않고, 임베딩 모델/LLM API/프레임워크를 쓰는 실습이므로 NN 관점에서 별도 설명했다.

## 0. 전체 맵

| 구분 | 노트북 | 핵심 NN/모델 |
|---|---|---|
| LLM 기초 | Chapter 1 | GloVe word vector |
| LLM 데이터 | Chapter 2 | GPT next-token dataset |
| LLM Attention | Chapter 3 | Causal Self-Attention, Multi-Head Attention |
| LLM GPT | Chapter 4 | Decoder-only Transformer/GPT |
| LLM Pretraining | Chapter 5 | Next-token LM training loop |
| LLM Fine-tuning | Chapter 6 | GPT classifier, LoRA |
| LLM Alignment | Chapter 7 | Instruction tuning, DPO |
| Vision | 01 | ResNet18 |
| Vision | 02 | ViT |
| Vision | 03 | DETR |
| Vision | 04 | U-Net |
| Vision | 05 | DDPM U-Net |
| Vision | 06 | Stable Diffusion pipeline |
| Vision | 11 | SegFormer |
| On-device | 1 | CNN pruning |
| On-device | 2 | CNN quantization |
| On-device | 3 | Knowledge Distillation |
| On-device | 4 | LLM pruning, Wanda |
| On-device | 5 | LLM quantization, AWQ/SmoothQuant |
| RAG | 1일차/2일차 | Embedding, Retriever, Reader, KG, Evaluation |
| Time Series | 3_4 practice | LSTM, Conv1D, RNN, Encoder-Decoder forecasting |
| Recommender | 7_8 practice | NCF, NGCF/GCF, BPR, Recall/Precision/NDCG |

---

# 1. `llm_hands_on`

## Chapter 1. 단어 벡터와 의미 공간

대상: `llm_hands_on/Chapter_1_Exercise_Vector Space.ipynb`

### 1) NN 설명

GloVe word vector를 사용한다. 직접 NN을 학습하지는 않지만, 단어를 고정 길이 dense vector로 표현해 단어 의미를 벡터 공간에서 비교한다.

용도:
- 단어 간 의미 유사도 계산
- `king - man + woman ≈ queen` 같은 벡터 연산
- 임베딩이 왜 LLM 입력 표현의 기초인지 이해

특징:
- 단어 하나가 하나의 벡터다.
- 문맥별 의미가 바뀌는 contextual embedding은 아니다.
- 유사도는 주로 cosine similarity로 본다.

### 2) 주요 코드

```python
import gensim.downloader as api

model = api.load("glove-wiki-gigaword-50")

model.similarity("cat", "dog")
model.similarity("cat", "car")
model.most_similar("king", topn=5)
```

### 3) 간략 설명

단어를 숫자 벡터로 바꾸면 의미가 비슷한 단어끼리 가까운 위치에 놓인다. 이후 Transformer의 token embedding도 같은 발상에서 출발한다.

---

## Chapter 2. 토큰화와 GPT 데이터셋

대상:
- `llm_hands_on/Chapter_2_Exercise_Dataset.ipynb`
- `llm_hands_on/Chapter_2_Exercise_Dataset.org.ipynb` 중복/원본성 파일

### 1) NN 설명

모델 자체보다 GPT 학습 데이터를 만드는 실습이다. GPT는 입력 토큰 시퀀스를 보고 다음 토큰을 예측한다.

용도:
- 긴 텍스트를 고정 길이 학습 샘플로 나누기
- 입력과 타깃을 한 칸 shifted pair로 만들기
- DataLoader로 batch 구성하기

특징:
- `input = tokens[i : i + max_length]`
- `target = tokens[i + 1 : i + max_length + 1]`
- `stride`가 작으면 샘플이 많이 겹쳐 데이터 수가 늘어난다.

### 2) 주요 코드

```python
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```

```python
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
```

### 3) 간략 설명

GPT 학습에서 정답은 입력의 다음 토큰이다. 그래서 input과 target 길이는 같지만 target은 한 칸 뒤로 밀려 있다.

---

## Chapter 3. Self-Attention과 Multi-Head Attention

대상: `llm_hands_on/Chapter_3_Excercise_Attention.ipynb`

### 1) NN 설명

Causal Self-Attention은 현재 토큰이 이전 토큰들만 참고하도록 만든 Attention이다. GPT 같은 decoder-only 모델의 핵심이다.

용도:
- 문맥 토큰 간 중요도 계산
- 미래 토큰 누출 방지
- 각 토큰을 문맥 정보가 섞인 벡터로 변환

특징:
- Q, K, V를 각각 Linear layer로 만든다.
- attention score는 `Q @ K.T`.
- causal mask로 미래 위치를 `-inf` 처리한다.
- Multi-head는 여러 관점의 attention을 병렬 계산한 뒤 concat/projection한다.

### 2) 주요 코드

```python
self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

self.register_buffer(
    "mask",
    torch.triu(torch.ones(context_length, context_length), diagonal=1)
)
```

```python
keys = self.W_key(x)
queries = self.W_query(x)
values = self.W_value(x)

attn_scores = queries @ keys.transpose(1, 2)
attn_scores.masked_fill_(
    self.mask.bool()[:num_tokens, :num_tokens],
    -torch.inf
)

attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
context_vec = attn_weights @ values
```

Multi-head 핵심:

```python
keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
values = values.view(b, num_tokens, self.num_heads, self.head_dim)

keys = keys.transpose(1, 2)
queries = queries.transpose(1, 2)
values = values.transpose(1, 2)

attn_scores = queries @ keys.transpose(2, 3)
attn_scores.masked_fill_(mask_bool, -torch.inf)
context_vec = (attn_weights @ values).transpose(1, 2)
context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
context_vec = self.out_proj(context_vec)
```

### 3) 간략 설명

시험 포인트는 mask다. `torch.triu(..., diagonal=1)`은 대각선 위쪽, 즉 미래 토큰 영역을 1로 만든다. 그 위치를 `-torch.inf`로 채우면 softmax 후 확률이 0이 된다.

---

## Chapter 3 보충. Multi-Head Attention 시각화

대상: `llm_hands_on/Chapter_3_Excercise_Viz_Multi_head_attention.ipynb`

### 1) NN 설명

학습된 GPT 내부 attention weight를 hook으로 꺼내 시각화하는 실습이다.

용도:
- layer/head별 attention pattern 관찰
- causal mask가 실제로 반영되는지 확인
- 어떤 토큰이 어떤 토큰을 보는지 분석

특징:
- forward hook으로 module 입력/출력을 잡는다.
- attention score를 다시 계산해 `attn_weights`를 저장한다.
- layer/head 선택 위젯으로 시각화한다.

### 2) 주요 코드

```python
mask_bool = module.mask.bool()[:num_tokens, :num_tokens]
attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
all_attentions.append(attn_weights.detach())
```

### 3) 간략 설명

Attention은 모델의 판단 근거 그 자체는 아니지만, 토큰 간 정보 흐름을 이해하는 데 매우 유용하다.

---

## Chapter 4. GPT 모델 구조와 생성 루프

대상: `llm_hands_on/Chapter_4_Excercise_GPT.ipynb`

### 1) NN 설명

Decoder-only Transformer인 GPT를 직접 구성한다.

구조:
- Token Embedding
- Position Embedding
- N개의 TransformerBlock
- Final LayerNorm
- Vocabulary Linear Head

특징:
- Pre-LayerNorm 구조
- Attention residual + FFN residual
- Output shape는 `(batch, seq_len, vocab_size)`
- 생성 시 마지막 위치 logits만 사용한다.

### 2) 주요 코드

Transformer block:

```python
class TransformerBlock(nn.Module):
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
```

GPT forward:

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

생성:

```python
idx_cond = idx[:, -context_size:]
with torch.no_grad():
    logits = model(idx_cond)
logits = logits[:, -1, :]
idx_next = torch.argmax(logits, dim=-1, keepdim=True)
idx = torch.cat((idx, idx_next), dim=1)
```

### 3) 간략 설명

GPT는 모든 위치에서 다음 토큰을 예측하지만, 생성할 때는 현재 문맥의 마지막 위치 logits만 사용해 다음 토큰 하나를 뽑는다.

---

## Chapter 5. GPT 사전학습 루프

대상: `llm_hands_on/Chapter_5_Excercise_Pretraining.ipynb`

### 1) NN 설명

GPT를 next-token prediction으로 사전학습하는 루프다.

용도:
- logits와 target으로 CrossEntropy 계산
- train/eval 모드 전환
- 주기적 validation loss 측정
- 학습 중 샘플 생성

특징:
- logits: `(B, T, V)`
- target: `(B, T)`
- CrossEntropy 입력은 `(B*T, V)`와 `(B*T)`로 펴야 한다.

### 2) 주요 코드

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    return loss
```

```python
model.train()
optimizer.zero_grad()
loss = calc_loss_batch(input_batch, target_batch, model, device)
loss.backward()
optimizer.step()
```

```python
model.eval()
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
model.train()
```

### 3) 간략 설명

사전학습의 본질은 "모든 위치에서 다음 토큰 맞히기"다. flatten은 PyTorch loss가 class 차원을 두 번째 축으로 기대하기 때문에 필요하다.

---

## Chapter 6. 텍스트 분류 Fine-Tuning

대상: `llm_hands_on/Chapter_6_Excercise_Finetuning_Classification.ipynb`

### 1) NN 설명

GPT를 문장 분류기로 fine-tuning한다. 시퀀스 전체를 읽은 뒤 마지막 토큰 위치의 representation/logits로 label을 예측한다.

용도:
- spam/ham 분류
- 사전학습된 LM을 downstream task에 맞게 변환

특징:
- 입력 text는 tokenization 후 padding/truncation한다.
- loss는 class label에 대한 CrossEntropy다.
- 마지막 토큰 위치 `[:, -1, :]`를 분류 결과로 사용한다.

### 2) 주요 코드

```python
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        self.encoded_texts = [
            encoded_text[:self.max_length]
            for encoded_text in self.encoded_texts
        ]
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
```

```python
logits = model(input_batch)[:, -1, :]
loss = torch.nn.functional.cross_entropy(logits, target_batch)
predicted_labels = torch.argmax(logits, dim=-1)
```

### 3) 간략 설명

언어모델은 원래 vocab token을 예측하지만, 분류 fine-tuning에서는 마지막 위치 출력만 뽑아 class label을 맞히도록 학습한다.

---

## Chapter 6 보충. LoRA 기반 분류 Fine-Tuning

대상: `llm_hands_on/Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb`

### 1) NN 설명

LoRA는 기존 Linear weight를 직접 크게 업데이트하지 않고, 저랭크 행렬 A/B만 학습하는 parameter-efficient fine-tuning 기법이다.

용도:
- LLM fine-tuning 메모리 절감
- 적은 학습 파라미터로 task adaptation

특징:
- 기존 Linear는 유지한다.
- 추가 경로 `x @ A @ B`만 학습한다.
- scaling은 `alpha / rank`.

### 2) 주요 코드

```python
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.rank = rank

    def forward(self, x):
        return (self.alpha / self.rank) * (x @ self.A @ self.B)
```

```python
class LinearWithLoRA(nn.Module):
    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

```python
def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)
```

### 3) 간략 설명

LoRA는 `W + ΔW`에서 `ΔW = A @ B`로 두고, `A/B`만 학습한다. rank가 작을수록 파라미터 수가 줄어든다.

---

## Chapter 7. Instruction Fine-Tuning

대상: `llm_hands_on/Chapter_7_Exercise_Follow_Instructions.ipynb`

### 1) NN 설명

Instruction-response 형식의 데이터로 GPT를 supervised fine-tuning한다.

용도:
- 명령어를 따르는 답변 생성
- Alpaca 스타일 prompt-response 학습

특징:
- 입력은 instruction/input/response를 한 문자열로 합친다.
- batch마다 길이가 다르므로 custom collate로 padding한다.
- loss 계산에서 padding target은 `ignore_index=-100`으로 무시한다.

### 2) 주요 코드

```python
def format_input(entry):
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text
```

```python
batch_max_length = max(len(item) + 1 for item in batch)
new_item = item.copy()
new_item += [pad_token_id]
padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

inputs = torch.tensor(padded[:-1])
targets = torch.tensor(padded[1:])

mask = targets == pad_token_id
indices = torch.nonzero(mask).squeeze()
if indices.numel() > 1:
    targets[indices[1:]] = ignore_index
```

### 3) 간략 설명

`ignore_index=-100`은 시험 단골이다. CrossEntropyLoss가 해당 target 위치를 손실 계산에서 제외한다.

---

## Chapter 7 보충. DPO 선호도 학습

대상: `llm_hands_on/Chapter_7_Exercise_Follow_Instructions_dpo.ipynb`

### 1) NN 설명

DPO(Direct Preference Optimization)는 chosen/rejected 답변 쌍으로 policy model이 더 선호되는 답변의 로그확률을 높이도록 학습한다.

용도:
- RLHF보다 단순한 preference alignment
- reference model 대비 policy model의 선호 차이 학습

특징:
- chosen/rejected 각각에 대해 log probability를 계산한다.
- prompt와 padding 부분은 mask로 loss에서 제외한다.
- reference model은 gradient 없이 비교 기준으로 사용한다.

### 2) 주요 코드

노트북 빈칸 기준 정답 형태:

```python
chosen_full_tokens = tokenizer.encode(f"{prompt}\n\n### Response:\n{chosen_response}")
rejected_full_tokens = tokenizer.encode(f"{prompt}\n\n### Response:\n{rejected_response}")
```

```python
mask = torch.ones(len(padded)).bool()
mask[len(sequence):] = False

if mask_prompt_tokens:
    mask[:prompt.shape[0] + 2] = False
```

```python
def compute_logprobs(logits, labels, selection_mask=None):
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    if selection_mask is not None:
        mask = selection_mask[:, 1:].clone()
        selected_log_probs = selected_log_probs * mask
        return selected_log_probs.sum(-1) / mask.sum(-1)
    return selected_log_probs.mean(-1)
```

```python
model_logratios = model_chosen_logprobs - model_rejected_logprobs
reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
logits = model_logratios - reference_logratios
losses = -F.logsigmoid(beta * logits)
```

### 3) 간략 설명

DPO의 핵심은 "policy가 reference보다 chosen을 rejected보다 더 높게 평가하게 만들기"다. 그래서 `chosen - rejected` log ratio가 중심이다.

---

# 2. `ai-specialist-2026-vision`

## 01. ResNet18 CIFAR-10

대상: `ai-specialist-2026-vision/01_ResNet18_CIFAR10.ipynb`

### 1) NN 설명

ResNet18은 residual connection을 쓰는 CNN 분류 모델이다.

용도:
- CIFAR-10 이미지 분류
- activation map, kernel, Grad-CAM, receptive field 분석

특징:
- residual block이 입력을 출력에 더해 gradient 흐름을 안정화한다.
- CIFAR-10은 32x32라 ImageNet용 큰 stem을 작게 바꾼다.
- 최종 `fc`를 class 수 10으로 교체한다.

### 2) 주요 코드

```python
def build_resnet18_for_cifar10(num_classes: int = 10):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
```

```python
criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=cfg.lr,
    momentum=cfg.momentum,
    weight_decay=cfg.weight_decay
)
```

Grad-CAM 핵심:

```python
self.h1 = target_layer.register_forward_hook(self._forward_hook)
self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

weights = self.grad.mean(dim=(2, 3), keepdim=True)
cam = (weights * self.activ).sum(dim=1, keepdim=True)
cam = F.relu(cam)
```

### 3) 간략 설명

ResNet18은 CNN feature extractor + classifier다. CIFAR에서는 `7x7 stride2 conv + maxpool`이 너무 공격적이라 `3x3 stride1 conv`와 `Identity maxpool`로 바꾸는 것이 중요하다.

---

## 02. ViT CIFAR-10

대상: `ai-specialist-2026-vision/02_ViT_CIFAR10.ipynb`

### 1) NN 설명

ViT(Vision Transformer)는 이미지를 patch token sequence로 바꾸고 Transformer encoder로 분류한다.

용도:
- 이미지 분류
- attention map으로 어느 patch를 보는지 시각화

특징:
- 이미지를 patch로 자른다.
- 각 patch를 Linear projection해 token embedding으로 만든다.
- `[CLS]` token과 position embedding을 더한다.
- Transformer encoder의 self-attention이 patch 간 전역 관계를 학습한다.

### 2) 주요 코드

```python
self.to_patch_embedding = nn.Sequential(
    Rearrange(
        "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
        p1=patch_height,
        p2=patch_width,
    ),
    nn.Linear(patch_dim, cfg.dim),
)

self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.dim))
self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, cfg.dim))
```

```python
def forward(self, img):
    x = self.to_patch_embedding(img)
    b, n, _ = x.shape

    cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + self.pos_embedding[:, : (n + 1)]
    x = self.dropout(x)

    x = self.transformer(x)
    x = x[:, 0] if self.pool == "cls" else x.mean(dim=1)
    return self.mlp_head(x)
```

Attention:

```python
qkv = self.to_qkv(x).chunk(3, dim=-1)
q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
attn = self.attend(dots)
out = einsum("b h i j, b h j d -> b h i d", attn, v)
```

### 3) 간략 설명

CNN은 locality를 convolution으로 넣어주지만, ViT는 이미지를 token sequence로 보고 self-attention으로 patch 간 관계를 학습한다.

---

## 03. DETR

대상: `ai-specialist-2026-vision/03_DETR.ipynb`

### 1) NN 설명

DETR은 object detection을 Transformer set prediction 문제로 푸는 모델이다.

용도:
- 이미지 내 객체 class와 bounding box 예측
- decoder query attention/cross-attention 시각화

특징:
- CNN backbone으로 feature map을 만든다.
- Transformer encoder/decoder가 object query를 처리한다.
- NMS 대신 fixed number object query와 bipartite matching 개념을 사용한다.

### 2) 주요 코드

```python
model = torch.hub.load(
    "facebookresearch/detr",
    "detr_resnet50",
    pretrained=True
)
outputs = model(img)
```

```python
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)
```

```python
prob = outputs["pred_logits"].softmax(-1)[0, :, :-1]
boxes = rescale_bboxes(outputs["pred_boxes"][0, keep], pil_img.size)
```

Hook/attention 시각화:

```python
hook = model.transformer.encoder.layers[-1].self_attn.register_forward_hook(...)
outputs = model(img)
enc_attn_weights = enc_attn_weights[0].reshape(shape + shape)
```

### 3) 간략 설명

DETR은 "박스 후보를 많이 만들고 후처리"하는 전통 detector와 달리, object query가 객체 슬롯처럼 동작한다.

---

## 04. U-Net

대상: `ai-specialist-2026-vision/04_Unet.ipynb`

### 1) NN 설명

U-Net은 segmentation용 encoder-decoder CNN이다.

용도:
- 픽셀 단위 binary mask 예측
- synthetic circle segmentation

특징:
- encoder는 해상도를 줄이며 semantic feature를 얻는다.
- bottleneck은 encoder와 decoder 사이의 가장 깊은 표현이다. 이 노트북에서는 별도 `bottleneck` 이름은 없고, `self.down4` 출력인 `x5`가 bottleneck 역할을 한다.
- decoder는 해상도를 복원한다.
- skip connection으로 encoder의 고해상도 정보를 decoder에 concat한다.
- output은 `(B, n_classes, H, W)` logits다.

### 2) 주요 코드

```python
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)  # bottleneck

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
```

```python
def forward(self, x):
    x1 = self.inc(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)  # bottleneck

    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    logits = self.outc(x)
    return logits
```

```python
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, masks)
```

### 3) 간략 설명

Segmentation에서는 각 pixel이 class prediction 대상이다. binary segmentation은 `BCEWithLogitsLoss`를 쓰고, sigmoid는 loss 내부에 포함된다.

---

## 05. DDPM

대상: `ai-specialist-2026-vision/05_DDPM.ipynb`

### 1) NN 설명

DDPM은 이미지에 노이즈를 점점 섞는 forward process와, 노이즈를 제거하는 reverse process를 학습하는 생성 모델이다.

용도:
- Fashion-MNIST 이미지 생성
- diffusion forward/reverse 과정 이해

특징:
- 모델은 깨끗한 이미지를 직접 예측하기보다 섞인 noise `epsilon`을 예측한다.
- backbone은 time embedding이 들어간 U-Net이다.
- loss는 실제 noise와 predicted noise 간 L1/MSE/Huber다.

### 2) 주요 코드

Forward diffusion:

```python
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
```

Training loss:

```python
def p_losses(denoise_model, x_start, t, noise=None, loss_type="huber"):
    if noise is None:
        noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)
    return F.smooth_l1_loss(noise, predicted_noise)
```

Reverse step:

```python
model_mean = sqrt_recip_alphas_t * (
    x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
)

if t_index == 0:
    return model_mean
else:
    noise = torch.randn_like(x)
    return model_mean + torch.sqrt(posterior_variance_t) * noise
```

### 3) 간략 설명

DDPM 학습의 정답은 원본 이미지가 아니라 "내가 섞은 노이즈"다. 모델이 노이즈를 잘 맞히면 역방향으로 노이즈를 제거하며 샘플을 만들 수 있다.

---

## 06. Stable Diffusion v1.4

대상: `ai-specialist-2026-vision/06_Stable_Diffusion_v1_4.ipynb`

### 1) NN 설명

Stable Diffusion은 latent diffusion 기반 text-to-image 모델이다. 직접 학습이 아니라 pretrained pipeline 추론 실습이다.

구성:
- Text encoder: prompt를 embedding으로 변환
- U-Net denoiser: latent noise 제거
- VAE decoder: latent를 이미지로 복원
- Scheduler: denoising step 제어

### 2) 주요 코드

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16
).to(device)
```

```python
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=steps,
    guidance_scale=guidance_scale,
    generator=torch.Generator(device=device).manual_seed(seed),
).images[0]
```

### 3) 간략 설명

Prompt는 text encoder를 거쳐 U-Net denoising 조건이 된다. `guidance_scale`이 커질수록 prompt를 강하게 따르지만 과하면 품질이 깨질 수 있다.

---

## 11. SegFormer

대상:
- `ai-specialist-2026-vision/11_SegFormer.ipynb`
- `ai-specialist-2026-vision/11_SegFormer_torch.ipynb` 로컬/torch 버전

### 1) NN 설명

SegFormer는 Transformer encoder와 lightweight decoder를 쓰는 semantic segmentation 모델이다.

용도:
- ADE20K semantic segmentation
- 입력 이미지를 픽셀 단위 class map으로 변환

특징:
- hierarchical Transformer feature를 사용한다.
- 출력 logits를 원본 이미지 크기로 upsample한다.
- pixel별 `argmax`가 segmentation map이다.

### 2) 주요 코드

```python
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

processor = SegformerImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
```

```python
upsampled_logits = F.interpolate(
    outputs.logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)
pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
```

### 3) 간략 설명

SegFormer도 segmentation이라 최종 목표는 픽셀별 class다. U-Net과 달리 encoder가 Transformer 기반이다.

---

# 3. `ai-specialist-2026-on-device`

## 1. Pruning for CNN

대상:
- `ai-specialist-2026-on-device/1. Pruning for CNN_answer.ipynb`
- `ai-specialist-2026-on-device/실습/1. Pruning for CNN.ipynb`
- `ai-specialist-2026-on-device/실습/1. Pruning for CNN_answer.ipynb`

### 1) NN 설명

CNN 모델에서 중요도가 낮은 weight를 0으로 만들어 모델을 압축한다.

용도:
- 모델 크기 감소
- 연산량 감소 가능성 확보
- sparse model의 정확도 변화 관찰

특징:
- fine-grained pruning: 개별 weight 단위
- vector/kernel/channel pruning: 구조적 단위 pruning
- global pruning: 전체 layer weight를 한 번에 thresholding
- pruning 후 mask를 계속 적용해야 0 weight가 되살아나지 않는다.

### 2) 주요 코드

Fine-grained pruning:

```python
def prune_weight_fine_grained(weight, sparsity):
    num_pruned_elements = round(weight.numel() * sparsity)
    importance = torch.abs(weight)
    threshold = torch.kthvalue(importance.flatten(), num_pruned_elements)[0]
    mask = importance > threshold
    weight.mul_(mask)
    return mask
```

Vector/kernel/channel pruning:

```python
importance = weight.abs().sum(dim=(3,), keepdim=True)      # vector
importance = weight.abs().sum(dim=(2, 3), keepdim=True)   # kernel
importance = weight.abs().sum(dim=(0, 2, 3), keepdim=True) # channel
```

Global pruning:

```python
parameters_to_prune = [
    param.view(-1)
    for name, param in model.named_parameters()
    if param.dim() > 1
]
all_weights = torch.cat(parameters_to_prune)
num_zeros = round(all_weights.numel() * sparsity)
threshold = torch.kthvalue(torch.abs(all_weights), num_zeros)[0]

mask = torch.abs(param.data) > threshold
```

Sparsity schedule:

```python
if epoch < epoch_start:
    sparsity = 0
elif epoch >= epoch_end:
    sparsity = sparsity_end
else:
    sparsity = sparsity_end + (sparsity_start - sparsity_end) * (
        1 - (epoch - epoch_start) / (epoch_end - epoch_start)
    ) ** exponent
```

### 3) 간략 설명

Pruning의 핵심은 importance 기준이다. 가장 흔한 기준은 `abs(weight)`이며, threshold 아래 weight를 0으로 만든다.

---

## 2. Quantization for CNN

대상:
- `ai-specialist-2026-on-device/실습/2. Quantization for CNN.ipynb`
- `ai-specialist-2026-on-device/실습/2. Quantization for CNN_answer.ipynb`

### 1) NN 설명

VGG 계열 CNN을 낮은 bit-width 정수 표현으로 근사한다.

용도:
- 모델 크기 감소
- integer 연산으로 추론 가속 가능
- activation/weight/bias quantization 이해

특징:
- activation은 asymmetric quantization: scale + zero point
- weight는 대칭 quantization: zero point 0
- per-channel weight scale이 per-tensor보다 정확도에 유리하다.
- int8 곱셈 결과는 int32 accumulator에 누적한다.

### 2) 주요 코드

VGG:

```python
class VGG(nn.Module):
    ARCH = [64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
```

Linear quantization:

```python
def get_quantization_scale_and_zero_point(fp_tensor, bitwidth):
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    fp_max = fp_tensor.max().item()
    fp_min = fp_tensor.min().item()
    scale = (fp_max - fp_min) / (quantized_max - quantized_min)
    zero_point = round(quantized_min - fp_min / scale)
    return scale, int(zero_point)
```

```python
def linear_quantize(fp_tensor, bitwidth, scale, zero_point, dtype=torch.int8):
    scaled_tensor = fp_tensor / scale
    rounded_tensor = torch.round(scaled_tensor).to(dtype)
    shifted_tensor = rounded_tensor + zero_point
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    return shifted_tensor.clamp_(quantized_min, quantized_max)
```

Weight per-channel:

```python
for oc in range(num_output_channels):
    _subtensor = tensor.select(dim_output_channels, oc)
    _scale = get_quantization_scale_for_weight(_subtensor, bitwidth)
    scale[oc] = _scale
scale = scale.view(scale_shape)
quantized_tensor = linear_quantize(tensor, bitwidth, scale, zero_point=0)
```

Quantized linear:

```python
output = F.linear(input.to(torch.int32), weight.to(torch.int32), bias)
weight_scale = weight_scale.flatten().view(1, -1)
output = output.float()
output = output * (input_scale * weight_scale / output_scale)
output = output + output_zero_point
output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
```

### 3) 간략 설명

양자화 공식은 `q = round(x / scale) + zero_point`다. 복원은 `x ≈ (q - zero_point) * scale`이다.

---

## 3. Knowledge Distillation

대상:
- `ai-specialist-2026-on-device/실습/3. Knowledge Distillation.ipynb`
- `ai-specialist-2026-on-device/실습/3. Knowledge Distillation_answer.ipynb`

### 1) NN 설명

큰 teacher CNN의 지식을 작은 student CNN으로 옮긴다.

용도:
- 작은 모델 정확도 향상
- on-device 배포용 경량 모델 학습

특징:
- Teacher: `VGGCifar9`
- Student: `VGGCifar5`
- Logit distillation: soft target KL류 손실
- Hidden distillation: cosine loss
- Feature map distillation: regressor + MSE loss

### 2) 주요 코드

Teacher/student forward:

```python
class VGGCifar9(nn.Module):
    def forward(self, x):
        x = self.backbone(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

class VGGCifar5(nn.Module):
    def forward(self, x):
        x = self.backbone(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x
```

Soft target KD:

```python
with torch.no_grad():
    teacher_logits = teacher(inputs)
student_logits = student(inputs)

soft_targets = F.softmax(teacher_logits / T, dim=-1)
student_prob = F.softmax(student_logits / T, dim=-1)

soft_targets_loss = torch.sum(
    soft_targets * (soft_targets.log() - student_prob.log())
) / student_prob.size(0) * (T ** 2)

label_loss = ce_loss(student_logits, labels)
loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss
```

Cosine hidden loss:

```python
hidden_rep_loss = cosine_loss(
    student_hidden_representation,
    teacher_hidden_representation,
    target=torch.ones(inputs.size(0)).cuda()
)
loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss
```

Feature map MSE:

```python
_, teacher_feature_map = teacher(inputs)
student_logits, regressor_feature_map = student(inputs)
hidden_rep_loss = mse_loss(regressor_feature_map, teacher_feature_map)
loss = feature_map_weight * hidden_rep_loss + ce_loss_weight * label_loss
```

### 3) 간략 설명

KD는 hard label만 보지 않고 teacher의 class distribution이나 feature를 함께 보게 한다. `T`가 커지면 softmax 분포가 부드러워져 class 간 관계가 더 잘 드러난다.

---

## 4. Pruning for LLM

대상:
- `ai-specialist-2026-on-device/실습/4. Pruning for LLM.ipynb`
- `ai-specialist-2026-on-device/실습/4. Pruning for LLM_answer.ipynb`

### 1) NN 설명

LLM의 Linear layer weight를 pruning한다. 평가 지표는 WikiText perplexity다.

용도:
- LLM weight sparsity 증가
- magnitude pruning과 activation-aware pruning 비교

특징:
- `lm_head`는 제외한다.
- Magnitude pruning은 `abs(W)`만 본다.
- Wanda는 `abs(W) * input_activation_norm`으로 importance를 계산한다.
- Calibration dataset으로 layer 입력 activation 통계를 수집한다.

### 2) 주요 코드

Magnitude pruning:

```python
@torch.no_grad()
def prune_magnitude(model, sparsity):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and "lm_head" not in n:
            W = m.weight.data
            num_elements = W.numel()
            num_zeros = round(num_elements * sparsity)
            importance = torch.abs(W)
            threshold = torch.kthvalue(importance.flatten(), num_zeros)[0]
            mask = importance > threshold
            W.mul_(mask)
```

Calibration hook:

```python
def add_batch(m, x, y, name):
    if isinstance(x, tuple):
        x = x[0]
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    if len(x.shape) == 3:
        x = x.reshape((-1, x.shape[-1]))
    x = x.t()
    input_dict[name] += torch.norm(x, p=2, dim=1) ** 2
```

Wanda:

```python
@torch.no_grad()
def prune_wanda(model, sparsity, input_feat):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and "lm_head" not in n:
            W = m.weight.data
            row, col = W.shape
            num_zeros_per_row = round(col * sparsity)
            importance = torch.abs(W) * input_feat[n]
            threshold = torch.kthvalue(importance, num_zeros_per_row, dim=1)[0]
            mask = importance > threshold.reshape(row, 1)
            W.mul_(mask)
```

### 3) 간략 설명

LLM pruning에서는 activation-aware 기준이 중요하다. 같은 weight 크기라도 자주/강하게 활성화되는 입력 채널이면 더 중요할 수 있다.

---

## 5. Quantization for LLM

대상:
- `ai-specialist-2026-on-device/실습/5. Quantization for LLM.ipynb`
- `ai-specialist-2026-on-device/실습/5. Quantization for LLM_answer.ipynb`

### 1) NN 설명

LLM weight/activation quantization을 실습한다.

용도:
- LLM 메모리 절감
- weight-only quantization, AWQ 이해
- W8A8, SmoothQuant류 스케일링 이해

특징:
- Weight-only quantization: W3A16/W4A16, weight만 저비트
- AWQ: activation outlier 채널을 고려해 중요한 weight channel을 보호
- SmoothQuant: activation 난이도를 weight 쪽으로 이동해 W8A8을 쉽게 만듦
- group-wise quantization은 group마다 scale/zero를 갖는다.

### 2) 주요 코드

Pseudo quantization:

```python
def pseudo_quantize_tensor(w, n_bit=4, q_group_size=-1):
    org_w_shape = w.shape
    if q_group_size > 0:
        w = w.reshape(-1, q_group_size)

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)

    w = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)
    w = (w - zeros) * scales
    return w.reshape(org_w_shape)
```

AWQ salient channel scale-up:

```python
importance = sum(input_feat[n]).float()
outlier_mask = torch.topk(importance, int(len(importance) * 0.01))[1]

m.weight.data[:, outlier_mask] *= scale_factor
m.weight.data = pseudo_quantize_tensor(
    m.weight.data,
    n_bit=w_bit,
    q_group_size=q_group_size
)
m.weight.data[:, outlier_mask] /= scale_factor
```

SmoothQuant-style LN/FC scaling:

```python
def scale_ln_fcs(ln, fcs, scales):
    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
```

W8A8 Linear:

```python
class W8A8Linear(nn.Module):
    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y
```

### 3) 간략 설명

LLM quantization에서 어려운 점은 activation outlier다. AWQ는 중요한 weight channel을 보호하고, SmoothQuant는 activation scale을 줄이는 대신 weight scale을 키워 균형을 맞춘다.

---

# 4. `ai-specialist-2026-rag`

## 1일차. LlamaIndex Practice

대상: `ai-specialist-2026-rag/1일차/실습 자료/Code/1. Llama_index.ipynb`

### 1) NN 설명

직접 NN을 구현하지 않는다. LlamaIndex가 embedding model과 LLM을 사용해 문서를 색인하고 질의응답한다.

용도:
- document loading
- chunking
- vector index 생성
- query engine 생성

특징:
- `VectorStoreIndex`는 문서 chunk embedding을 저장한다.
- query도 embedding으로 바꾼 뒤 유사 chunk를 찾는다.
- response synthesizer가 검색 결과를 답변으로 요약한다.

### 2) 주요 코드

```python
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is the first and the last programs the author tried writing?")
```

```python
parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
nodes = parser.get_nodes_from_documents(documents)
```

### 3) 간략 설명

RAG의 핵심은 "LLM에게 모든 문서를 넣는 것"이 아니라, 질문과 관련된 chunk를 먼저 찾은 뒤 그 context만 LLM에 넣는 것이다.

---

## 1일차. RAG Practice

대상: `ai-specialist-2026-rag/1일차/실습 자료/Code/2. RAG.ipynb`

### 1) NN 설명

LLM 단독 답변과 RAG 답변을 비교한다. NN은 API 기반 GPT와 embedding model이다.

용도:
- 오래된/최근 정보 질문 비교
- 단순/복잡 질문 비교
- WikipediaReader + VectorStoreIndex RAG 구성

특징:
- LLM 단독은 knowledge cutoff와 hallucination 문제가 있다.
- RAG는 외부 문서를 검색해 context를 제공한다.

### 2) 주요 코드

```python
def generate_answer(question):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content
```

```python
documents = reader.load_data(city_names, auto_suggest=False)
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What's the arts and culture scene in Berlin?")
```

### 3) 간략 설명

LLM 자체 지식이 부족하거나 최신 정보가 필요한 경우 RAG가 효과적이다. 단, 검색 품질이 낮으면 답변도 흔들린다.

---

## 2일차. Data Preprocessing for KDD/CRAG

대상: `ai-specialist-2026-rag/2일차/실습 자료/Code/1. Data_preprocessing.ipynb`

### 1) NN 설명

NN 모델 구현보다는 RAG 평가용 데이터 구조를 이해하고, 검색 결과를 chunk로 파싱하는 전처리 실습이다.

용도:
- CRAG dataset 분석
- domain/question type/dynamism 확인
- HTML/search result parsing
- sentence offset 기반 chunking

특징:
- 질문이 어떤 domain인지, 정답이 static/dynamic인지 파악한다.
- 검색 결과 schema를 이해해야 retriever가 쓸 수 있는 text chunk를 만들 수 있다.

### 2) 주요 코드

```python
from blingfire import text_to_sentences_and_offsets

_, offsets = text_to_sentences_and_offsets(text)
chunk = text[start:end]
```

```python
def generate_answer(question):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content
```

### 3) 간략 설명

RAG 성능은 모델만이 아니라 "문서를 얼마나 잘 쪼개고 정리했는지"에 크게 좌우된다.

---

## 2일차. Task 1 Web-Based Retrieval Summarization

대상:
- `ai-specialist-2026-rag/2일차/실습 자료/Code/2. Task_1.ipynb`
- `ai-specialist-2026-rag/2일차/실습 자료/Code/2. Task_1-Copy1.ipynb` 중복 복사본

### 1) NN 설명

Retriever + Reader 형태의 RAG를 직접 구현한다. Retriever는 embedding model을 쓰고, Reader는 LLM을 사용한다.

용도:
- web search result에서 chunk 추출
- query와 chunk embedding 유사도 계산
- top-k chunk를 context로 답변 생성

특징:
- embedding model은 text를 vector로 변환한다.
- cosine similarity/내적 기반으로 관련 chunk를 고른다.
- Reader prompt가 최종 답변 품질에 중요하다.

### 2) 주요 코드

```python
class BaseRetriever:
    def get_embeddings(self, texts):
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        embeddings = [np.array(item.embedding) for item in response.data]
        return np.array(embeddings)
```

```python
def extract_chunks(all_documents):
    for document in all_documents:
        _, offsets = text_to_sentences_and_offsets(document)
        chunk = document[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
```

```python
def retrieve(self, query, search_results, topk):
    all_documents = parse_htmls(search_results)
    all_chunks = extract_chunks(all_documents)
    query_embedding = self.get_embeddings([query])
    chunk_embeddings = self.get_embeddings(all_chunks)
    scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:topk]
```

### 3) 간략 설명

Task 1의 핵심은 retriever가 정답 근거를 가져오느냐이다. Reader는 가져온 context 안에서만 답해야 hallucination이 줄어든다.

---

## 2일차. Task 2 Knowledge Graph and Web Augmentation

대상: `ai-specialist-2026-rag/2일차/실습 자료/Code/3. Task_2.ipynb`

### 1) NN 설명

LLM이 KG API query를 생성하고, KG 결과와 web context를 함께 써서 답하는 구조다. 직접 NN 학습은 없다.

용도:
- finance/sports/movie/music 등 structured KG 질의
- LLM query generator
- KG query executor
- Web + KG augmented answer

특징:
- KG는 정확한 structured fact에 강하다.
- Web search는 비정형 context에 강하다.
- LLM은 자연어 질문을 API call/query로 바꾸고, 결과를 답변으로 종합한다.

### 2) 주요 코드

```python
class CRAG(object):
    def finance_get_company_name(self, query: str):
        data = {"query": query}
        return requests.post(f"{self.server}/finance/get_company_name", json=data)
```

```python
metric = "eps"
if metric == "price":
    result = api.finance_get_price(ticker)
elif metric == "eps":
    result = api.finance_get_eps(ticker)
```

### 3) 간략 설명

KG-RAG는 "검색된 글"뿐 아니라 "구조화된 API 결과"를 context로 쓴다. 숫자/최신값/정확한 entity lookup에 유리하다.

---

## 2일차. RAG Framework Evaluation and MCP

대상: `ai-specialist-2026-rag/2일차/실습 자료/Code/4_RAG_framework_evaluation_with_MCP.ipynb`

### 1) NN 설명

RAG framework를 평가하고 MCP 기반 KG query stage로 확장한다. NN은 LLM judge와 RAG 내부 LLM/embedding을 사용한다.

용도:
- CRAG evaluation metric 적용
- prediction과 ground truth 비교
- Perfect/Acceptable/Missing/Incorrect 분류
- MCP tool로 KG query stage 개선

특징:
- 평가에도 LLM을 쓴다.
- JSON parsing으로 judge 결과를 구조화한다.
- RAG 구성 요소를 retriever/reader/KG query engine으로 나눈다.

### 2) 주요 코드

```python
def generate_answer(user_prompt, system_prompt="You are a helpful assistant."):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content
```

```python
def CRAG_evaluation(question, ground_truth, prediction):
    context_template = f"""
    Question: {question}
    Ground Truth: {ground_truth}
    Model Prediction: {prediction}
    """
    evaluation_result = generate_answer(
        user_prompt=context_template,
        system_prompt=INSTRUCTIONS
    )
    eval_res = parse_response(evaluation_result)
    return eval_res
```

```python
class RAG:
    def __init__(self):
        self.retriever = LlamaIndexRetriever()
        self.reader = Reader()
        self.kg_query_engine = KGQueryEngine()
```

### 3) 간략 설명

RAG 시험 포인트는 pipeline 분리다. Retriever가 근거를 찾고, Reader가 답변을 만들고, Evaluation이 답변 품질을 판정한다.

---

# 4.5 `ai-specialist-2026-time`

## Time Series Practice

대상:
- `ai-specialist-2026-time/3_4-ts-practice/ts_practice.ipynb`
- `ai-specialist-2026-time/3_4-ts-practice/ts_solution.ipynb`

### 1) NN 설명

시계열의 과거 `sequence_length=50`개 값을 보고 다음 시점 값을 예측하는 forecasting 실습이다. 기본 모델은 LSTM이고, 비교 모델로 Conv1D, vanilla RNN, optional RNN encoder-decoder가 나온다.

용도:
- 단변량 시계열 next-step forecasting
- sliding window 데이터셋 생성
- recurrent model 출력 shape 이해
- RMSE/MAPE 평가

특징:
- 입력 shape은 `[B, sequence_length, num_features]`, 예시는 `[32, 50, 1]`.
- LSTM/RNN 출력은 모든 time step에 대한 hidden output을 반환한다.
- 다음 값 예측에는 마지막 time step 출력 `[:, -1, 0]`을 사용한다.
- Encoder-Decoder는 하나의 다음 값이 아니라 `target_len=10`개 미래 값을 한 번에 예측한다.

### 2) 주요 코드

Sliding window:

```python
sequence_length = 50

def convert_data_into_tensors(data_seq):
    features, labels = [], []
    for i in range(len(data_seq) - sequence_length):
        # input window: [50, 1]
        features.append(data_seq[i:i + sequence_length])

        # next-step label: scalar
        labels.append(data_seq[i + sequence_length, 0])

    features = torch.tensor(np.array(features), dtype=torch.float32)  # [N, 50, 1]
    labels = torch.tensor(np.array(labels), dtype=torch.float32)      # [N]
    return features, labels
```

LSTM:

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,      # 1
            hidden_size,     # 64
            num_layers,      # 2
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [B, 50, 1]
        out, _ = self.lstm(x)    # out: [B, 50, 64]
        return self.linear(out)  # [B, 50, 1]
```

Training:

```python
pred = model(batch_x)[:, -1, 0]  # [B], last time step prediction
loss = loss_fn(pred, batch_y)    # batch_y: [B]

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Conv1D:

```python
class Conv1DModel(nn.Module):
    def forward(self, x):
        # x: [B, T, F]
        x = x.transpose(1, 2)  # [B, F, T], Conv1d expects channel first
        x = self.conv1d(x)     # [B, hidden_size, T-1] when kernel_size=2
        x = x.transpose(1, 2)  # [B, T-1, hidden_size]
        return self.fc(x)      # [B, T-1, 1]
```

Encoder-Decoder:

```python
class RNNRNN(nn.Module):
    def forward(self, source, target_len):
        # source: [B, 50, 1]
        h = self.encoder(source)          # [num_layers, B, hidden_size]
        predictions = []
        input = source[:, -1, :].unsqueeze(1)  # [B, 1, 1]

        for t in range(target_len):
            out, h = self.decoder(input, h)    # out: [B, 1, 1]
            predictions.append(out.squeeze(1)) # [B, 1]
            input = out                        # autoregressive decoding

        outputs = torch.stack(predictions, dim=1)  # [B, target_len, 1]
        return outputs
```

Evaluation:

```python
y_hat = model(X_test)              # [N, 50, 1]
test_predictions = y_hat[:, -1, 0] # [N]

rmse = root_mean_squared_error(y_test, test_predictions)
mape = mean_absolute_percentage_error(y_test, test_predictions)
```

### 3) 간략 설명

시계열 실습의 핵심은 sliding window다. 과거 50개 값을 입력으로 만들고, 바로 다음 값을 label로 둔다. RNN/LSTM은 전체 time step output을 반환하지만, next-step 예측은 마지막 step output만 사용한다.

---

## RecSys NCF

대상:
- `ai-specialist-2026-time/7_8-recsys-practice/RecSys_NCF.ipynb`

### 1) NN 설명

NCF(Neural Collaborative Filtering)는 user id와 movie id를 각각 embedding vector로 바꾼 뒤, 두 embedding을 concat하여 MLP로 rating을 예측한다.

용도:
- explicit rating prediction
- user/movie latent factor 학습
- RMSE 및 Top-K recommendation 평가

특징:
- user와 item은 one-hot으로 직접 넣지 않고 embedding lookup을 한다.
- user embedding `[B, 32]`와 movie embedding `[B, 32]`를 concat해 `[B, 64]`를 만든다.
- MLP 출력은 rating prediction `[B, 1]`이다.
- 학습 loss는 MSELoss다.

### 2) 주요 코드

Dataset:

```python
class MovieLens:
    def __getitem__(self, item):
        users = self.users[item]
        movies = self.movies[item]
        ratings = self.ratings[item]
        return {
            "users": torch.tensor(users, dtype=torch.long).to(device),
            "movies": torch.tensor(movies, dtype=torch.long).to(device),
            "ratings": torch.tensor(ratings, dtype=torch.long).to(device),
        }
```

Model:

```python
class Neural_Collaborative_Filtering(nn.Module):
    def __init__(self, n_users, n_movies):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, 32)
        self.movie_embedding = nn.Embedding(n_movies, 32)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, users, movies, ratings=None):
        # users, movies: [B]
        user_embedding = self.user_embedding(users)    # [B, 32]
        movie_embedding = self.movie_embedding(movies) # [B, 32]

        input_embedding = torch.cat(
            [user_embedding, movie_embedding],
            dim=1
        )  # [B, 64]

        hidden_feature = self.relu(self.fc1(input_embedding)) # [B, 32]
        output = self.fc2(hidden_feature)                     # [B, 1]
        return output
```

Training:

```python
prediction = model(train_data["users"], train_data["movies"]) # [B, 1]
ground_truth = train_data["ratings"].view(batch_size, -1).to(torch.float32) # [B, 1]
loss = nn.MSELoss(reduction="mean")(prediction, ground_truth)
```

### 3) 간략 설명

NCF는 user/item id를 dense latent vector로 바꾼 다음 MLP로 상호작용을 학습한다. Matrix factorization이 dot product만 쓰는 것과 달리, NCF는 nonlinear interaction을 학습할 수 있다.

---

## RecSys NGCF/GCF

대상:
- `ai-specialist-2026-time/7_8-recsys-practice/RecSys_GCF_practice.ipynb`
- `ai-specialist-2026-time/7_8-recsys-practice/RecSys_GCF_sol.ipynb`

### 1) NN 설명

NGCF(Neural Graph Collaborative Filtering)는 user와 item을 bipartite graph의 node로 보고, interaction edge를 따라 embedding message passing을 수행하는 추천 모델이다.

용도:
- implicit feedback 기반 Top-K recommendation
- user-item graph 구조 활용
- Recall@K, Precision@K, NDCG@K 평가

특징:
- user node index는 `[0, num_users)`.
- item node index는 `movieId + num_users`로 뒤에 붙인다.
- edge_index shape은 `[2, E]`.
- node embedding은 user와 item을 합쳐 `[num_users + num_items, D]`.
- 각 NGCF layer는 이웃 feature와 element-wise interaction `src_feat * dst_feat`를 함께 사용한다.
- 최종 representation은 layer별 output을 concat한다.
- BPR loss는 positive item score가 negative item score보다 커지도록 학습한다.

### 2) 주요 코드

Graph:

```python
def create_edge_index(df, rating_threshold=1.0):
    src, dst = [], []
    for _, row in df.iterrows():
        if row["rating"] >= rating_threshold:
            src.append(row["userId"])              # user node
            dst.append(row["movieId"] + num_users) # item node offset
    return torch.tensor([src, dst], dtype=torch.long) # [2, E]
```

NGCF layer:

```python
class NGCFLayer(nn.Module):
    def forward(self, edge_index, node_features, user_num, item_num):
        # node_features: [num_users + num_items, D]
        src, dst = edge_index  # src: [E], dst: [E]

        deg = torch.zeros(node_features.size(0), device=node_features.device)
        deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float))
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        norm = 1.0 / torch.sqrt(deg[src] * deg[dst]) # [E]

        src_feat = node_features[src] # [E, D]
        dst_feat = node_features[dst] # [E, D]

        # item -> user message
        edge_messages_for_src = self.W1(dst_feat) + self.W2(dst_feat * src_feat)
        edge_messages_for_src *= norm.unsqueeze(1)

        # user -> item message
        edge_messages_for_dst = self.W1(src_feat) + self.W2(src_feat * dst_feat)
        edge_messages_for_dst *= norm.unsqueeze(1)

        aggregated_messages = torch.zeros_like(node_features)
        aggregated_messages.index_add_(0, src, edge_messages_for_src)
        aggregated_messages.index_add_(0, dst, edge_messages_for_dst)

        # self connection
        aggregated_messages[:user_num] += self.W1(node_features[:user_num])
        aggregated_messages[user_num:] += self.W1(node_features[user_num:])

        return self.leaky_relu(aggregated_messages)
```

NGCF model:

```python
class NGCF(nn.Module):
    def forward(self, edge_index):
        node_features = self.node_embeddings.weight  # [U+I, D0]
        layer_outputs = [node_features]

        for layer in self.layers:
            node_features = layer(
                edge_index,
                node_features,
                self.num_users,
                self.num_items
            )
            layer_outputs.append(node_features)

        final_features = torch.concat(layer_outputs, dim=-1) # [U+I, D0+D1+D2]
        user_features = final_features[:self.num_users]      # [U, D_final]
        item_features = final_features[self.num_users:]      # [I, D_final]
        return user_features, item_features
```

BPR loss:

```python
def bpr_loss(self, user_emb, pos_item_emb, neg_item_emb, reg_weight=1e-4):
    # user_emb, pos_item_emb, neg_item_emb: [B, D_final]
    pos_scores = torch.sum(user_emb * pos_item_emb, dim=1) # [B]
    neg_scores = torch.sum(user_emb * neg_item_emb, dim=1) # [B]

    loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
    reg_loss = reg_weight * (
        user_emb.norm(2).pow(2)
        + pos_item_emb.norm(2).pow(2)
        + neg_item_emb.norm(2).pow(2)
    ) / user_emb.size(0)
    return loss + reg_loss
```

Evaluation:

```python
scores = torch.matmul(item_features, user_emb) # [num_items]
topk_scores, topk_indices = torch.topk(scores, k=k)

recall_u = hits / n_pos
precision_u = hits / k
ndcg_u = dcg / idcg if idcg > 0 else 0.0
```

### 3) 간략 설명

NGCF는 user-item interaction graph에서 이웃 정보를 embedding에 전파한다. NCF가 관측된 `(user, item)` 쌍만 MLP에 넣는 방식이라면, NGCF는 graph 구조를 통해 여러 hop의 collaborative signal을 반영한다.

---

# 5. 시험 직전 암기 포인트

## Attention/GPT

- `Q = xWq`, `K = xWk`, `V = xWv`
- score: `Q @ K.T / sqrt(d_k)`
- causal mask: 미래 위치를 `-inf`
- softmax 후 미래 토큰 attention은 0
- GPT output: `(B, T, vocab_size)`
- generation은 `logits[:, -1, :]`만 사용
- LM loss는 `logits.flatten(0, 1)` vs `target.flatten()`

## Fine-tuning/Alignment

- 분류 fine-tuning은 마지막 token output 사용: `model(x)[:, -1, :]`
- instruction tuning은 padding target을 `-100`으로 masking
- DPO는 `(chosen - rejected)` log ratio를 policy와 reference에서 비교
- LoRA는 `Linear(x) + alpha/rank * xAB`

## Vision

- ResNet: residual connection, CIFAR용 stem 수정
- ViT: image patch -> token -> CLS + positional embedding -> Transformer
- DETR: object query + Transformer detection
- U-Net: encoder-decoder + skip concat
- DDPM: 모델이 이미지가 아니라 noise를 예측
- Stable Diffusion: text encoder + latent U-Net + VAE + scheduler
- SegFormer: Transformer encoder 기반 semantic segmentation

## On-device

- Pruning importance 기본값: `abs(weight)`
- Fine-grained: weight 원소 단위
- Structured pruning: vector/kernel/channel 단위
- Quantization: `q = round(x / scale) + zero_point`
- Weight quantization은 보통 zero point 0인 symmetric 방식
- KD: teacher logits/feature를 student가 따라가게 학습
- Wanda: `abs(W) * activation_norm`
- AWQ: 중요한 activation channel의 weight를 보호
- SmoothQuant: activation scale 부담을 weight로 이동

## RAG

- RAG = retrieve relevant context + generate answer
- Chunk size/overlap은 검색 품질에 직접 영향
- Embedding model은 text를 vector로 바꾼다.
- Retriever가 틀린 context를 가져오면 Reader도 흔들린다.
- KG/API augmentation은 숫자, entity, 최신 structured fact에 유리하다.

## Time Series

- Sliding window: 과거 `sequence_length`개 입력으로 다음 값을 예측
- LSTM/RNN input: `[B, T, F]` when `batch_first=True`
- LSTM/RNN output: `[B, T, hidden]`, next-step은 보통 마지막 step 사용
- Conv1D input은 `[B, C, T]`라 `transpose(1, 2)` 필요
- Scaler는 train에만 fit, test에는 transform
- RMSE는 큰 오차에 민감, MAPE는 실제값 0 근처에서 불안정

## Recommender

- NCF: user embedding + item embedding concat -> MLP -> rating
- NGCF: user-item graph에서 message passing
- BPR loss: `-logsigmoid(pos_score - neg_score)`
- `movieId + num_users`는 item node offset
- Recall@K, Precision@K, NDCG@K는 Top-K ranking 평가 지표

---

# 6. NN 구조 ASCII 도면 + 차원 주석

이 섹션은 시험 직전 구조 암기용이다.

표기:
- `B`: batch size
- `T`: token sequence length
- `V`: vocabulary size
- `C`: channel count
- `H, W`: image height/width
- `D`: embedding dimension
- `N`: token/patch/query count
- `K`: number of classes

ASCII 도면은 깨짐을 줄이기 위해 영문 라벨만 사용했다.

---

## 6.1 Word Embedding / GloVe

관련 노트북:
- `llm_hands_on/Chapter_1_Exercise_Vector Space.ipynb`

### Model Structure

```text
word
  |
  v
+------------------+
| Vocabulary Lookup |
+------------------+
  |
  v
embedding vector
shape: [D]

example:
"cat" -> [50]
"dog" -> [50]
cosine_similarity([50], [50]) -> scalar
```

### Annotated Code

```python
model = api.load("glove-wiki-gigaword-50")

# word1_vec, word2_vec: [D], example D=50
sim_cat_dog = model.similarity("cat", "dog")  # scalar cosine similarity

# Returns top-k nearest word vectors in embedding space.
similar_words = model.most_similar("king", topn=5)
```

차원 포인트:
- GloVe 50d 모델이면 단어 하나는 `[50]` 벡터다.
- 두 단어 유사도는 두 `[50]` 벡터의 cosine similarity다.
- 직접 forward가 있는 PyTorch NN은 아니지만, 임베딩 계층의 가장 단순한 형태로 보면 된다.

---

## 6.2 Causal Self-Attention

관련 노트북:
- `llm_hands_on/Chapter_3_Excercise_Attention.ipynb`
- `llm_hands_on/Chapter_4_Excercise_GPT.ipynb`

### Model Structure

```text
x: [B, T, D_in]
  |
  +-------------------+
  | Linear Wq         | -> Q: [B, T, D_out]
  +-------------------+
  | Linear Wk         | -> K: [B, T, D_out]
  +-------------------+
  | Linear Wv         | -> V: [B, T, D_out]
  +-------------------+
          |
          v
Q @ K^T: [B, T, T]
          |
          v
Causal Mask upper triangle -> -inf
          |
          v
Softmax over last dim: [B, T, T]
          |
          v
Attention @ V: [B, T, D_out]
```

Example:

```text
B=2, T=6, D_in=3, D_out=2

x       : [2, 6, 3]
Q/K/V   : [2, 6, 2]
scores  : [2, 6, 6]
weights : [2, 6, 6]
context : [2, 6, 2]
```

### Annotated Code

```python
def forward(self, x):
    # x: [B, T, D_in]
    b, num_tokens, d_in = x.shape

    # Q/K/V: [B, T, D_out]
    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)

    # Attention scores: [B, T, D_out] @ [B, D_out, T] -> [B, T, T]
    attn_scores = queries @ keys.transpose(1, 2)

    # mask_bool: [T, T], True means future token positions.
    # masked score becomes -inf, so softmax probability becomes 0.
    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
    attn_scores.masked_fill_(mask_bool, -torch.inf)

    # attn_weights: [B, T, T]
    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)

    # context_vec: [B, T, T] @ [B, T, D_out] -> [B, T, D_out]
    context_vec = attn_weights @ values
    return context_vec
```

차원 포인트:
- `scores[i, j]`는 i번째 token이 j번째 token을 얼마나 보는지다.
- causal mask는 `j > i`인 미래 위치를 막는다.
- softmax는 마지막 축 `T` 기준이라 각 query token마다 볼 token들의 확률 합이 1이다.
- causal mask와 padding mask는 다르다. causal mask는 미래 토큰을 막고, padding mask는 `<pad>` 위치를 loss/attention에서 제외한다.
- `register_buffer("mask", ...)`는 학습 파라미터는 아니지만 `model.to(device)`와 `state_dict`에 함께 따라가는 텐서다.

---

## 6.3 Multi-Head Attention

관련 노트북:
- `llm_hands_on/Chapter_3_Excercise_Attention.ipynb`
- `llm_hands_on/Chapter_4_Excercise_GPT.ipynb`
- `ai-specialist-2026-vision/02_ViT_CIFAR10.ipynb`

### Model Structure

```text
x: [B, T, D]
  |
  v
Linear Q/K/V: [B, T, D]
  |
  v
Split heads: [B, T, Hh, Dh]
  |
  v
Transpose: [B, Hh, T, Dh]
  |
  v
Scaled Dot Attention per head
scores: [B, Hh, T, T]
  |
  v
context per head: [B, Hh, T, Dh]
  |
  v
Concat heads: [B, T, D]
  |
  v
Output Projection: [B, T, D]
```

Example:

```text
B=2, T=6, D=4, Hh=2, Dh=2

Q before split : [2, 6, 4]
Q after split  : [2, 6, 2, 2]
Q transposed   : [2, 2, 6, 2]
scores         : [2, 2, 6, 6]
context        : [2, 2, 6, 2]
concat         : [2, 6, 4]
```

### Annotated Code

```python
def forward(self, x):
    # x: [B, T, D]
    b, num_tokens, d_in = x.shape

    # Q/K/V: [B, T, D]
    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)

    # Split D into num_heads * head_dim.
    # [B, T, D] -> [B, T, Hh, Dh]
    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
    values = values.view(b, num_tokens, self.num_heads, self.head_dim)

    # Move head dimension before token dimension.
    # [B, T, Hh, Dh] -> [B, Hh, T, Dh]
    keys = keys.transpose(1, 2)
    queries = queries.transpose(1, 2)
    values = values.transpose(1, 2)

    # [B, Hh, T, Dh] @ [B, Hh, Dh, T] -> [B, Hh, T, T]
    attn_scores = queries @ keys.transpose(2, 3)

    # Causal mask: [T, T], broadcast to [B, Hh, T, T]
    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
    attn_scores.masked_fill_(mask_bool, -torch.inf)

    # [B, Hh, T, T]
    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

    # [B, Hh, T, T] @ [B, Hh, T, Dh] -> [B, Hh, T, Dh]
    context_vec = attn_weights @ values

    # [B, Hh, T, Dh] -> [B, T, Hh, Dh] -> [B, T, D]
    context_vec = context_vec.transpose(1, 2).contiguous()
    context_vec = context_vec.view(b, num_tokens, self.d_out)

    # [B, T, D] -> [B, T, D]
    context_vec = self.out_proj(context_vec)
    return context_vec
```

차원 포인트:
- `D = Hh * Dh`여야 한다.
- head를 나누는 이유는 서로 다른 attention 관점을 병렬로 학습하기 위해서다.
- `contiguous()`는 transpose 후 메모리 배치가 view 가능한 상태가 되도록 보장한다.

---

## 6.4 GPT / Decoder-Only Transformer

관련 노트북:
- `llm_hands_on/Chapter_4_Excercise_GPT.ipynb`
- `llm_hands_on/Chapter_5_Excercise_Pretraining.ipynb`

### Model Structure

```text
token ids: [B, T]
  |
  +------------------+
  | Token Embedding  | -> [B, T, D]
  +------------------+
  | Position Emb     | -> [T, D] broadcast to [B, T, D]
  +------------------+
          |
          v
token + position: [B, T, D]
          |
          v
+----------------------------+
| Transformer Block x L      |
|  LN -> MHA -> Add          |
|  LN -> FFN -> Add          |
+----------------------------+
          |
          v
Final LayerNorm: [B, T, D]
          |
          v
LM Head Linear: [B, T, V]
```

Example:

```text
B=4, T=256, D=768, V=50257

input ids : [4, 256]
embedding : [4, 256, 768]
logits    : [4, 256, 50257]
```

### Annotated Code

```python
class GPTModel(nn.Module):
    def forward(self, in_idx):
        # in_idx: [B, T], token ids
        batch_size, seq_len = in_idx.shape

        # tok_embeds: [B, T, D]
        tok_embeds = self.tok_emb(in_idx)

        # pos_embeds: [T, D], broadcast when added to [B, T, D]
        pos = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(pos)

        # x: [B, T, D]
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        # x: [B, T, D]
        x = self.trf_blocks(x)

        # logits: [B, T, V]
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

TransformerBlock:

```python
def forward(self, x):
    # x: [B, T, D]

    shortcut = x
    x = self.norm1(x)       # [B, T, D]
    x = self.att(x)         # [B, T, D]
    x = self.drop_shortcut(x)
    x = x + shortcut        # residual add, [B, T, D]

    shortcut = x
    x = self.norm2(x)       # [B, T, D]
    x = self.ff(x)          # [B, T, D]
    x = self.drop_shortcut(x)
    x = x + shortcut        # residual add, [B, T, D]
    return x
```

LM loss:

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    # input_batch: [B, T]
    # target_batch: [B, T], next-token labels
    logits = model(input_batch)  # [B, T, V]

    # CrossEntropy expects [N, C] and [N].
    # IMPORTANT: pass raw logits, not softmax probabilities.
    # [B, T, V] -> [B*T, V]
    # [B, T]    -> [B*T]
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss
```

Generation:

```python
# idx: [B, current_T]
idx_cond = idx[:, -context_size:]  # crop to max context: [B, <=context_size]

with torch.no_grad():
    logits = model(idx_cond)       # [B, T, V]

logits = logits[:, -1, :]          # last token only: [B, V]
idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # [B, 1]
idx = torch.cat((idx, idx_next), dim=1)                # [B, T+1]
```

차원 포인트:
- 학습 때는 모든 위치의 logits를 loss에 쓴다.
- 생성 때는 마지막 위치 logits만 다음 토큰 선택에 쓴다.
- `target_batch`는 input보다 한 칸 뒤로 shift된 next-token label이다.
- `CrossEntropyLoss = log_softmax + NLLLoss`가 합쳐진 함수라서 logits에 softmax를 미리 적용하면 안 된다.

---

## 6.5 GPT Classifier Fine-Tuning

관련 노트북:
- `llm_hands_on/Chapter_6_Excercise_Finetuning_Classification.ipynb`

### Model Structure

```text
text
  |
  v
Tokenizer + Padding
  |
  v
token ids: [B, T]
  |
  v
GPT backbone: [B, T, D]
  |
  v
last token representation/logits: [B, K]
  |
  v
CrossEntropy with labels: [B]
```

Example:

```text
B=8, T=120, K=2
input_batch : [8, 120]
logits all  : [8, 120, 2]
last logits : [8, 2]
labels      : [8]
```

### Annotated Code

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    # input_batch: [B, T]
    # target_batch: [B], class ids such as 0/1
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    # model(input_batch): [B, T, K]
    # Use last token position for sequence classification.
    logits = model(input_batch)[:, -1, :]  # [B, K]

    loss = F.cross_entropy(logits, target_batch)
    return loss
```

차원 포인트:
- GPT는 sequence model이라 각 위치에 output이 있다.
- 분류에서는 마지막 위치가 전체 문장을 요약한 위치처럼 쓰인다.

---

## 6.6 LoRA Adapter

관련 노트북:
- `llm_hands_on/Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb`

### Model Structure

```text
x: [B, T, In]
  |
  +-----------------------------+
  | Frozen Linear W             | -> base: [B, T, Out]
  +-----------------------------+
  |
  +-----------------------------+
  | LoRA A: [In, R]             |
  | LoRA B: [R, Out]            | -> delta: [B, T, Out]
  +-----------------------------+
          |
          v
base + alpha/R * delta: [B, T, Out]
```

Example:

```text
B=4, T=128, In=768, Out=768, R=16

x      : [4, 128, 768]
x @ A  : [4, 128, 16]
@ B    : [4, 128, 768]
output : [4, 128, 768]
```

### Annotated Code

```python
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        # A: [In, R], B: [R, Out]
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.rank = rank

    def forward(self, x):
        # x: [B, T, In] or [B, In]
        # x @ A: [B, T, R]
        # x @ A @ B: [B, T, Out]
        return (self.alpha / self.rank) * (x @ self.A @ self.B)

class LinearWithLoRA(nn.Module):
    def forward(self, x):
        # base path and LoRA path have the same output shape.
        return self.linear(x) + self.lora(x)
```

차원 포인트:
- LoRA parameter 수는 `In*R + R*Out`이다.
- full Linear update `In*Out`보다 훨씬 작다.
- `B`는 0으로 초기화되어 처음에는 원래 Linear와 거의 같은 출력을 낸다.

---

## 6.7 Instruction Tuning and DPO Masks

관련 노트북:
- `llm_hands_on/Chapter_7_Exercise_Follow_Instructions.ipynb`
- `llm_hands_on/Chapter_7_Exercise_Follow_Instructions_dpo.ipynb`

### Model Structure

Instruction tuning:

```text
instruction + input + response
  |
  v
token ids with padding: [B, T]
  |
  v
GPT logits: [B, T, V]
  |
  v
next-token loss
padding targets -> ignore_index
```

DPO:

```text
prompt
  |
  +--> chosen sequence   -> policy logprob, reference logprob
  |
  +--> rejected sequence -> policy logprob, reference logprob

policy_ratio    = chosen_policy - rejected_policy
reference_ratio = chosen_ref    - rejected_ref
DPO logits      = policy_ratio - reference_ratio
loss            = -logsigmoid(beta * DPO logits)
```

### Annotated Code

Instruction collate:

```python
# padded: [T+1], with EOS/pad appended
inputs = torch.tensor(padded[:-1])   # [T]
targets = torch.tensor(padded[1:])   # [T], next-token labels

# Mask padding targets so CrossEntropyLoss ignores them.
mask = targets == pad_token_id       # [T]
indices = torch.nonzero(mask).squeeze()
if indices.numel() > 1:
    targets[indices[1:]] = ignore_index  # usually -100
```

DPO logprob:

```python
def compute_logprobs(logits, labels, selection_mask=None):
    # logits: [B, T, V]
    # labels: [B, T]

    # Shift for autoregressive next-token prediction.
    labels = labels[:, 1:].clone()   # [B, T-1]
    logits = logits[:, :-1, :]       # [B, T-1, V]

    log_probs = F.log_softmax(logits, dim=-1)  # [B, T-1, V]

    # Pick logprob of the correct label at each position.
    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)   # [B, T-1, 1]
    ).squeeze(-1)                    # [B, T-1]

    if selection_mask is not None:
        # mask: [B, T-1], False for prompt/padding positions
        mask = selection_mask[:, 1:].clone()
        selected_log_probs = selected_log_probs * mask
        return selected_log_probs.sum(-1) / mask.sum(-1)  # [B]

    return selected_log_probs.mean(-1)  # [B]
```

차원 포인트:
- Instruction tuning의 mask는 padding loss를 제거한다.
- DPO mask는 prompt와 padding을 제거하고 response token만 비교하게 만든다.

---

## 6.8 ResNet18 for CIFAR-10

관련 노트북:
- `ai-specialist-2026-vision/01_ResNet18_CIFAR10.ipynb`

### Model Structure

```text
image: [B, 3, 32, 32]
  |
  v
Conv3x3 s1, BN, ReLU: [B, 64, 32, 32]
  |
  v
MaxPool removed: [B, 64, 32, 32]
  |
  v
Layer1: [B, 64, 32, 32]
  |
  v
Layer2: [B, 128, 16, 16]
  |
  v
Layer3: [B, 256, 8, 8]
  |
  v
Layer4: [B, 512, 4, 4]
  |
  v
AdaptiveAvgPool: [B, 512, 1, 1]
  |
  v
Flatten: [B, 512]
  |
  v
FC: [B, 10]
```

Residual block:

```text
x
|\
| +--> Conv -> BN -> ReLU -> Conv -> BN
|                                  |
+----------------------------------+
          Add
           |
          ReLU
```

### Annotated Code

```python
def build_resnet18_for_cifar10(num_classes=10):
    model = models.resnet18(weights=None)

    # Original ImageNet stem is too aggressive for 32x32 CIFAR images.
    # input : [B, 3, 32, 32]
    # output: [B, 64, 32, 32]
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )

    # Keep spatial size; no early downsampling.
    model.maxpool = nn.Identity()

    # final classifier: [B, 512] -> [B, 10]
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
```

Grad-CAM:

```python
# activ: [B, C, Hf, Wf], grad: [B, C, Hf, Wf]
weights = grad.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
cam = (weights * activ).sum(dim=1, keepdim=True)  # [B, 1, Hf, Wf]
cam = F.relu(cam)
```

차원 포인트:
- CIFAR 입력은 작아서 초반 stride를 줄이는 것이 중요하다.
- residual add는 두 branch shape이 같아야 한다. 다르면 downsample projection이 맞춰준다.

---

## 6.9 ViT for CIFAR-10

관련 노트북:
- `ai-specialist-2026-vision/02_ViT_CIFAR10.ipynb`

### Model Structure

```text
image: [B, 3, 32, 32]
  |
  v
Split into patches
  |
  v
patch vectors: [B, N, patch_dim]
  |
  v
Linear projection: [B, N, D]
  |
  v
Add CLS token: [B, N+1, D]
  |
  v
Add position embedding: [B, N+1, D]
  |
  v
Transformer Encoder x L: [B, N+1, D]
  |
  v
CLS or mean pooling: [B, D]
  |
  v
MLP Head: [B, 10]
```

Example:

```text
image_size=32, patch_size=4
N = (32/4) * (32/4) = 64
patch_dim = 3 * 4 * 4 = 48

image        : [B, 3, 32, 32]
patch vector : [B, 64, 48]
embedding    : [B, 64, D]
with CLS     : [B, 65, D]
logits       : [B, 10]
```

### Annotated Code

```python
def forward(self, img):
    # img: [B, C, H, W], example [B, 3, 32, 32]

    # [B, C, H, W] -> [B, N, patch_dim] -> [B, N, D]
    x = self.to_patch_embedding(img)
    b, n, _ = x.shape

    # cls_tokens: [B, 1, D]
    cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)

    # [B, N, D] -> [B, N+1, D]
    x = torch.cat((cls_tokens, x), dim=1)

    # pos_embedding: [1, N+1, D], broadcast over batch
    x = x + self.pos_embedding[:, : (n + 1)]
    x = self.dropout(x)

    # [B, N+1, D]
    x = self.transformer(x)

    # CLS pooling: [B, D]
    x = x[:, 0] if self.pool == "cls" else x.mean(dim=1)

    # [B, D] -> [B, 10]
    return self.mlp_head(x)
```

차원 포인트:
- ViT의 token 수는 patch 수 + CLS 1개다.
- attention matrix는 `[B, heads, N+1, N+1]`이다.
- patch size가 작아지면 token 수가 늘어 attention 비용이 커진다.

시험 주의:
- 이 노트북의 `ViT.forward()`는 `self.mlp_head(x)`를 그대로 반환하므로 기본적으로 raw logits다.
- raw logits에는 `nn.CrossEntropyLoss()` 또는 `F.cross_entropy()`를 써야 한다.
- `F.nll_loss()`를 쓰려면 모델 출력이 `F.log_softmax(logits, dim=-1)`인 log-probability여야 한다.

---

## 6.10 DETR

관련 노트북:
- `ai-specialist-2026-vision/03_DETR.ipynb`

### Model Structure

```text
image: [B, 3, H, W]
  |
  v
CNN Backbone
  |
  v
feature map: [B, C, Hf, Wf]
  |
  v
Flatten + Positional Encoding
  |
  v
Transformer Encoder: [B, Hf*Wf, D]
  |
  v
Object Queries: [B, Q, D]
  |
  v
Transformer Decoder: [B, Q, D]
  |
  +--> Class Head: [B, Q, num_classes+1]
  |
  +--> Box Head:   [B, Q, 4]
```

Example:

```text
Q=100 object queries

pred_logits: [B, 100, 92]   # COCO classes + no-object
pred_boxes : [B, 100, 4]    # normalized cx, cy, w, h
```

### Annotated Code

```python
outputs = model(img)

# outputs["pred_logits"]: [B, Q, num_classes + 1]
# outputs["pred_boxes"] : [B, Q, 4], normalized cxcywh
prob = outputs["pred_logits"].softmax(-1)[0, :, :-1]  # [Q, num_classes]

# Keep high-confidence object queries.
keep = prob.max(-1).values > 0.7  # [Q]

# Convert [cx, cy, w, h] -> [xmin, ymin, xmax, ymax]
boxes = rescale_bboxes(outputs["pred_boxes"][0, keep], pil_img.size)
```

차원 포인트:
- DETR의 Q는 object query 수다.
- 각 query가 하나의 object slot처럼 class와 box를 예측한다.
- `num_classes + 1`에서 `+1`은 no-object class다.

---

## 6.11 U-Net

관련 노트북:
- `ai-specialist-2026-vision/04_Unet.ipynb`

### Model Structure

```text
input image: [B, 1, 128, 128]
  |
  v
inc DoubleConv: [B, 64, 128, 128] ---- skip x1 ----+
  |                                                |
  v                                                |
down1: [B, 128, 64, 64] ------- skip x2 -------+   |
  |                                            |   |
  v                                            |   |
down2: [B, 256, 32, 32] ----- skip x3 -----+   |   |
  |                                        |   |   |
  v                                        |   |   |
down3: [B, 512, 16, 16] --- skip x4 ---+   |   |   |
  |                                    |   |   |   |
  v                                    |   |   |   |
down4 bottleneck: [B, 1024, 8, 8]      |   |   |   |
  |                                    |   |   |   |
  v                                    |   |   |   |
up1 concat x4: [B, 512, 16, 16] <------+   |   |   |
  |                                        |   |   |
  v                                        |   |   |
up2 concat x3: [B, 256, 32, 32] <----------+   |   |
  |                                            |   |
  v                                            |   |
up3 concat x2: [B, 128, 64, 64] <--------------+   |
  |                                                |
  v                                                |
up4 concat x1: [B, 64, 128, 128] <--------------+
  |
  v
1x1 Conv: [B, 1, 128, 128]
```

### Annotated Code

```python
def forward(self, x):
    # x: [B, 1, H, W], example [B, 1, 128, 128]

    # Encoder path; save skip tensors.
    x1 = self.inc(x)       # [B, 64, H, W]
    x2 = self.down1(x1)    # [B, 128, H/2, W/2]
    x3 = self.down2(x2)    # [B, 256, H/4, W/4]
    x4 = self.down3(x3)    # [B, 512, H/8, W/8]

    # Bottleneck: deepest and lowest-resolution feature map.
    x5 = self.down4(x4)    # [B, 1024, H/16, W/16]

    # Decoder path; upsample and concatenate skip tensors.
    x = self.up1(x5, x4)   # [B, 512, H/8, W/8]
    x = self.up2(x, x3)    # [B, 256, H/4, W/4]
    x = self.up3(x, x2)    # [B, 128, H/2, W/2]
    x = self.up4(x, x1)    # [B, 64, H, W]

    logits = self.outc(x)  # [B, n_classes, H, W]
    return logits
```

Up block:

```python
def forward(self, x1, x2):
    # x1: decoder tensor, lower spatial resolution
    # x2: encoder skip tensor, higher spatial resolution
    x1 = self.up(x1)

    # Pad x1 if H/W do not exactly match x2.
    diff_y = x2.size(2) - x1.size(2)
    diff_x = x2.size(3) - x1.size(3)
    x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                    diff_y // 2, diff_y - diff_y // 2])

    # Concatenate on channel axis.
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)
```

차원 포인트:
- bottleneck은 `x5`다.
- skip concat은 channel 축 `dim=1`에서 일어난다.
- segmentation output은 class별 pixel logits라서 `[B, K, H, W]`다.

---

## 6.12 DDPM U-Net

관련 노트북:
- `ai-specialist-2026-vision/05_DDPM.ipynb`

### Model Structure

```text
x_t noisy image: [B, C, H, W]
t timestep: [B]
  |
  +--> Sinusoidal Time Emb -> MLP -> [B, time_dim]
  |
  v
Initial Conv: [B, dim, H, W]
  |
  v
Down Blocks + Linear Attention
  |
  v
Middle ResBlock + Full Attention
  |
  v
Up Blocks + Skip Connections
  |
  v
Final Conv: predicted noise [B, C, H, W]
```

Diffusion process:

```text
Forward:
x_0 + noise -> x_t
q(x_t | x_0) = sqrt(alpha_bar_t) * x_0
             + sqrt(1-alpha_bar_t) * epsilon

Training:
model(x_t, t) -> epsilon_hat
loss(epsilon_hat, epsilon)

Sampling:
x_T -> x_{T-1} -> ... -> x_0
```

### Annotated Code

```python
def q_sample(x_start, t, noise=None):
    # x_start: [B, C, H, W]
    # t      : [B]
    if noise is None:
        noise = torch.randn_like(x_start)  # [B, C, H, W]

    # extract returns [B, 1, 1, 1], broadcast over C/H/W.
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    # output x_t: [B, C, H, W]
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
```

```python
def p_losses(denoise_model, x_start, t, noise=None, loss_type="huber"):
    # x_start: [B, C, H, W]
    # t      : [B]
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)  # [B, C, H, W]

    # model predicts the noise, not the clean image.
    predicted_noise = denoise_model(x_noisy, t)  # [B, C, H, W]

    loss = F.smooth_l1_loss(noise, predicted_noise)
    return loss
```

차원 포인트:
- timestep `t`는 image처럼 4D가 아니라 `[B]`다.
- `extract(...).reshape(B, 1, 1, 1)`로 image tensor에 broadcast한다.
- DDPM U-Net의 출력 shape은 입력 image와 같다.
- 학습 target은 `x_0`가 아니라 실제로 섞은 noise `epsilon`이다.
- `model(x_t, t)`가 예측하는 값은 보통 `epsilon_hat`이며, loss는 `epsilon_hat`과 `epsilon` 사이에서 계산한다.

---

## 6.13 Stable Diffusion Pipeline

관련 노트북:
- `ai-specialist-2026-vision/06_Stable_Diffusion_v1_4.ipynb`

### Model Structure

```text
prompt text
  |
  v
Text Encoder
  |
  v
text embeddings: [B, T_text, D]

random latent z_T: [B, 4, H/8, W/8]
  |
  v
Latent U-Net Denoiser
conditioned by text embeddings
  |
  v
z_0 latent: [B, 4, H/8, W/8]
  |
  v
VAE Decoder
  |
  v
image: [B, 3, H, W]
```

Classifier-free guidance:

```text
noise_pred = uncond + guidance_scale * (cond - uncond)
```

### Annotated Code

```python
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16
).to(device)

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=steps,
    guidance_scale=guidance_scale,
    generator=torch.Generator(device=device).manual_seed(seed),
).images[0]
```

차원 포인트:
- Stable Diffusion은 pixel space가 아니라 latent space에서 denoising한다.
- 512x512 이미지는 보통 latent `[B, 4, 64, 64]`에서 처리된다.
- `guidance_scale`은 conditional/unconditional noise prediction 차이를 얼마나 강하게 반영할지 정한다.

---

## 6.14 SegFormer

관련 노트북:
- `ai-specialist-2026-vision/11_SegFormer.ipynb`
- `ai-specialist-2026-vision/11_SegFormer_torch.ipynb`

### Model Structure

```text
image: [B, 3, H, W]
  |
  v
Image Processor
  |
  v
pixel_values: [B, 3, H_in, W_in]
  |
  v
Hierarchical Transformer Encoder
  |
  v
Multi-scale features
  |
  v
Lightweight Segmentation Decoder
  |
  v
logits: [B, K, H_low, W_low]
  |
  v
Upsample to original size: [B, K, H, W]
  |
  v
Argmax over class dim: [B, H, W]
```

### Annotated Code

```python
inputs = processor(images=image, return_tensors="pt")

# outputs.logits: [B, K, H_low, W_low]
outputs = model(**inputs)

# Resize logits to original image size.
upsampled_logits = F.interpolate(
    outputs.logits,
    size=image.size[::-1],  # (H, W)
    mode="bilinear",
    align_corners=False,
)

# Pixel-wise class id map: [H, W]
pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
```

차원 포인트:
- `argmax(dim=1)`은 class 축에서 가장 높은 class를 고른다.
- segmentation logits는 class별 heatmap이라고 보면 된다.

---

## 6.15 VGG CNN for Quantization, Pruning, KD

관련 노트북:
- `ai-specialist-2026-on-device/실습/1. Pruning for CNN*.ipynb`
- `ai-specialist-2026-on-device/실습/2. Quantization for CNN*.ipynb`
- `ai-specialist-2026-on-device/실습/3. Knowledge Distillation*.ipynb`

### Model Structure

VGG-style CIFAR classifier:

```text
image: [B, 3, 32, 32]
  |
  v
Conv-BN-ReLU: [B, 64, 32, 32]
  |
  v
Conv-BN-ReLU: [B, 128, 32, 32]
  |
  v
MaxPool: [B, 128, 16, 16]
  |
  v
Conv blocks...
  |
  v
MaxPool stages...
  |
  v
feature map: [B, 512, 1 or 2, 1 or 2]
  |
  v
GlobalAvgPool or Flatten: [B, 512]
  |
  v
Linear classifier: [B, 10]
```

Teacher/student KD:

```text
Teacher VGGCifar9
image -> deeper CNN -> logits_t: [B, 10]
                    -> feature_t: [B, C_t, H_t, W_t]

Student VGGCifar5
image -> smaller CNN -> logits_s: [B, 10]
                    -> feature_s: [B, C_s, H_s, W_s]

Distillation losses:
CE(logits_s, labels)
KL/soft target(logits_s/T, logits_t/T)
Cosine(feature_s, feature_t)
MSE(regressor(feature_s), feature_t)
```

### Annotated Code

VGG forward:

```python
def forward(self, x):
    # x: [B, 3, 32, 32]
    x = self.backbone(x)       # example [B, 512, 1, 1] or [B, 512, 2, 2]

    # Some notebooks use mean pooling, some flatten after avgpool.
    x = x.mean([2, 3])         # [B, 512]
    # or x = x.view(x.shape[0], -1)

    x = self.classifier(x)     # [B, 10]
    return x
```

Pruning:

```python
def prune_weight_fine_grained(weight, sparsity):
    # weight: Conv [C_out, C_in, Kh, Kw] or Linear [Out, In]
    importance = torch.abs(weight)
    num_pruned = round(weight.numel() * sparsity)

    # kthvalue uses 1-based k. Guard sparsity=0 in real code.
    # if num_pruned == 0: return torch.ones_like(weight)
    threshold = torch.kthvalue(importance.flatten(), num_pruned)[0]

    # mask has the same shape as weight.
    mask = importance > threshold
    weight.mul_(mask)
    return mask
```

Quantization:

```python
def linear_quantize(fp_tensor, bitwidth, scale, zero_point):
    # fp_tensor: any shape, for example activation [B, C, H, W]
    # q = round(x / scale) + zero_point
    q = torch.round(fp_tensor / scale) + zero_point
    qmin, qmax = get_quantized_range(bitwidth)
    return q.clamp(qmin, qmax).to(torch.int8)
```

Bias quantization:

```python
# For Conv/Linear, bias uses int32 scale.
# input_scale: scalar activation scale
# weight_scale: per-output-channel scale, shape [Out] or [Out, 1, 1, 1]
bias_scale = input_scale * weight_scale
quantized_bias = round(fp_bias / bias_scale).to(torch.int32)
```

KD soft target:

```python
with torch.no_grad():
    teacher_logits = teacher(inputs)  # [B, 10]

student_logits = student(inputs)      # [B, 10]

soft_targets = F.softmax(teacher_logits / T, dim=-1)  # [B, 10]
student_prob = F.softmax(student_logits / T, dim=-1)  # [B, 10]

soft_targets_loss = torch.sum(
    soft_targets * (soft_targets.log() - student_prob.log())
) / student_prob.size(0) * (T ** 2)
```

차원 포인트:
- Conv weight shape은 `[out_channels, in_channels, kernel_h, kernel_w]`다.
- Channel pruning에서 `dim=(0, 2, 3)`으로 합치면 input channel별 중요도가 된다.
- KD에서 teacher/student logits shape은 같아야 한다. feature map은 다르면 regressor로 맞춘다.
- signed int8 범위는 보통 `[-128, 127]`이지만 symmetric weight quantization은 scale을 `absmax / 127`로 잡아 zero point 0을 유지하는 경우가 많다.
- activation quantization은 입력 분포가 비대칭일 수 있어 zero point를 같이 쓰는 asymmetric 방식이 자주 나온다.
- `torch.kthvalue`의 `k`는 1부터 시작한다. `sparsity=0`이면 별도 처리 없이 `k=0`이 되어 에러가 난다.
- threshold와 같은 값이 많은 경우 `importance > threshold` 때문에 실제 sparsity가 목표와 조금 달라질 수 있다.

---

## 6.16 LLM Pruning and Quantization

관련 노트북:
- `ai-specialist-2026-on-device/실습/4. Pruning for LLM*.ipynb`
- `ai-specialist-2026-on-device/실습/5. Quantization for LLM*.ipynb`

### Model Structure

OPT/GPT-like causal LM:

```text
token ids: [B, T]
  |
  v
Token Embedding: [B, T, D]
  |
  v
Decoder Layer x L
  |
  +--> Self-Attention Linear layers
  |    q_proj/k_proj/v_proj/out_proj
  |
  +--> MLP Linear layers
       fc1/fc2 or up/down/gate proj
  |
  v
Final hidden states: [B, T, D]
  |
  v
LM Head: [B, T, V]
```

Pruning target:

```text
Only nn.Linear weights
Skip lm_head

W: [Out, In]
importance: [Out, In]
mask: [Out, In]
```

Quantization target:

```text
Weight-only:
W fp16 [Out, In]
  -> group reshape [-1, group_size]
  -> int range [0, 2^bits-1]
  -> dequantized W approx [Out, In]

W8A8:
activation x -> int8-like
weight W     -> int8-like
linear output
```

### Annotated Code

Magnitude pruning:

```python
def prune_magnitude(model, sparsity):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            W = module.weight.data       # [Out, In]
            importance = torch.abs(W)    # [Out, In]

            num_zeros = round(W.numel() * sparsity)
            threshold = torch.kthvalue(importance.flatten(), num_zeros)[0]

            mask = importance > threshold  # [Out, In]
            W.mul_(mask)
```

Wanda pruning:

```python
def prune_wanda(model, sparsity, input_feat):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            W = module.weight.data  # [Out, In]
            row, col = W.shape

            # input_feat[name]: [In], activation norm per input channel
            importance = torch.abs(W) * input_feat[name]  # broadcast to [Out, In]

            num_zeros_per_row = round(col * sparsity)
            threshold = torch.kthvalue(
                importance, num_zeros_per_row, dim=1
            )[0]  # [Out]

            mask = importance > threshold.reshape(row, 1)  # [Out, In]
            W.mul_(mask)
```

Pseudo quantization:

```python
def pseudo_quantize_tensor(w, n_bit=4, q_group_size=-1):
    # w: [Out, In] for Linear weight
    org_w_shape = w.shape

    if q_group_size > 0:
        # [Out, In] -> [-1, group_size]
        w = w.reshape(-1, q_group_size)

    # Per row/group min-max.
    max_val = w.amax(dim=1, keepdim=True)  # [-1, 1]
    min_val = w.amin(dim=1, keepdim=True)  # [-1, 1]

    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)

    # Quantize then dequantize.
    w = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)
    w = (w - zeros) * scales

    return w.reshape(org_w_shape)
```

AWQ salient channel:

```python
# input_feat[name] is a list of activation statistics.
importance = sum(input_feat[name]).float()  # [In]

# Select top 1 percent important input channels.
outlier_mask = torch.topk(importance, int(len(importance) * 0.01))[1]  # [num_outliers]

# Protect important columns before quantization.
module.weight.data[:, outlier_mask] *= scale_factor
module.weight.data = pseudo_quantize_tensor(
    module.weight.data, n_bit=w_bit, q_group_size=q_group_size
)
module.weight.data[:, outlier_mask] /= scale_factor
```

W8A8 Linear:

```python
def forward(self, x):
    # x: [B, T, In]
    q_x = self.act_quant(x)                 # int8-like [B, T, In]
    y = F.linear(q_x, self.weight, self.bias) # [B, T, Out]
    q_y = self.output_quant(y)              # optional int8-like [B, T, Out]
    return q_y
```

차원 포인트:
- LLM Linear weight는 보통 `[out_features, in_features]`다.
- Wanda의 activation 통계는 input channel 축 `[In]`에 곱해져 각 column 중요도를 바꾼다.
- group quantization에서는 마지막 차원 `In`을 group 단위로 나눈다.

---

## 6.17 RAG Neural Components

관련 노트북:
- `ai-specialist-2026-rag/**/*.ipynb`

### Model Structure

RAG는 하나의 end-to-end NN이라기보다 embedding model, retriever, LLM reader를 연결한 pipeline이다.

```text
documents
  |
  v
Chunking
  |
  v
Embedding Model
  |
  v
document vectors: [N_docs, D]
  |
  v
Vector Store

query
  |
  v
Embedding Model
  |
  v
query vector: [D]
  |
  v
Similarity Search
  |
  v
top-k chunks
  |
  v
Reader LLM
  |
  v
answer text
```

Example:

```text
N_docs=1000, D=1536, top_k=5

doc_embeddings : [1000, 1536]
query_embedding: [1536]
similarities   : [1000]
top_chunks     : 5 text chunks
```

### Annotated Code

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

# embeddings: [N, D]
embeddings = np.array([item.embedding for item in response.data])
```

```python
# query_embedding: [1, D]
# chunk_embeddings: [N, D]
scores = cosine_similarity(query_embedding, chunk_embeddings)[0]  # [N]
top_indices = np.argsort(scores)[::-1][:topk]
```

```python
# Reader LLM receives retrieved text as context.
messages = [
    {"role": "system", "content": "Answer using the provided context."},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
]
```

차원 포인트:
- embedding dimension `D`는 모델마다 다르다.
- 검색은 vector similarity이고, 생성은 LLM이 담당한다.
- RAG 품질은 `chunking -> embedding -> top-k retrieval -> prompt` 전체에 의해 결정된다.

---

## 6.18 Time Series LSTM / Conv1D / RNN

관련 노트북:
- `ai-specialist-2026-time/3_4-ts-practice/ts_solution.ipynb`

### Model Structure

LSTM next-step forecasting:

```text
raw series: [N_total, 1]
  |
  v
MinMaxScaler
  |
  v
Sliding Window
  |
  +--> X: [N, 50, 1]
  |
  +--> y: [N]
  |
  v
LSTM(input_size=1, hidden_size=64, layers=2)
  |
  v
all step outputs: [B, 50, 64]
  |
  v
Linear(64 -> 1): [B, 50, 1]
  |
  v
last step: [B]
  |
  v
MSELoss with y: [B]
```

Conv1D alternative:

```text
X: [B, T, F]
  |
  v
transpose for Conv1d
  |
  v
[B, F, T]
  |
  v
Conv1d(F -> hidden, kernel=2)
  |
  v
[B, hidden, T-1]
  |
  v
transpose back
  |
  v
[B, T-1, hidden]
  |
  v
Linear(hidden -> 1): [B, T-1, 1]
```

### Annotated Code

```python
def convert_data_into_tensors(data_seq):
    # data_seq: [N_total, 1]
    features, labels = [], []
    for i in range(len(data_seq) - sequence_length):
        features.append(data_seq[i:i + sequence_length])   # [50, 1]
        labels.append(data_seq[i + sequence_length, 0])    # scalar

    features = torch.tensor(np.array(features), dtype=torch.float32) # [N, 50, 1]
    labels = torch.tensor(np.array(labels), dtype=torch.float32)     # [N]
    return features, labels
```

```python
class LSTMModel(nn.Module):
    def forward(self, x):
        # x: [B, 50, 1]
        out, _ = self.lstm(x)       # [B, 50, 64]
        out = self.linear(out)      # [B, 50, 1]
        return out

# Use only the final time step for next-step prediction.
pred = model(batch_x)[:, -1, 0]     # [B]
loss = loss_fn(pred, batch_y)       # batch_y: [B]
```

차원 포인트:
- `batch_first=True`라서 RNN/LSTM 입력은 `[B, T, F]`다.
- PyTorch `nn.Conv1d`는 `[B, C, T]`를 기대하므로 transpose가 필요하다.
- `MinMaxScaler`는 train data에 `fit`하고 test data에는 `transform`만 해야 leakage를 막는다.

---

## 6.19 Time Series Encoder-Decoder

관련 노트북:
- `ai-specialist-2026-time/3_4-ts-practice/ts_solution.ipynb`

### Model Structure

```text
source window: [B, 50, 1]
  |
  v
Encoder RNN
  |
  v
hidden h: [num_layers, B, hidden_size]
  |
  v
Decoder initial input = last source value: [B, 1, 1]
  |
  v
Decoder step 1 -> y_hat_1: [B, 1]
  |
  v
Decoder step 2 -> y_hat_2: [B, 1]
  |
  v
...
  |
  v
stack target_len outputs
  |
  v
predictions: [B, target_len, 1]
```

Example:

```text
sequence_length=50, target_len=10, hidden_size=64, num_layers=2

source : [B, 50, 1]
h      : [2, B, 64]
output : [B, 10, 1]
label  : [B, 10, 1]
```

### Annotated Code

```python
class EncoderRNN(nn.Module):
    def forward(self, x):
        # x: [B, 50, 1]
        _, h = self.rnn(x)
        return h  # [num_layers, B, hidden_size]

class DecoderRNN(nn.Module):
    def forward(self, x, h):
        # x: [B, 1, 1], h: [num_layers, B, hidden_size]
        out, h = self.rnn(x, h)  # out: [B, 1, hidden_size]
        out = self.fc(out)       # [B, 1, 1]
        return out, h

class RNNRNN(nn.Module):
    def forward(self, source, target_len):
        h = self.encoder(source)
        input = source[:, -1, :].unsqueeze(1)  # [B, 1, 1]
        predictions = []

        for t in range(target_len):
            out, h = self.decoder(input, h)
            predictions.append(out.squeeze(1)) # [B, 1]
            input = out                        # autoregressive feedback

        return torch.stack(predictions, dim=1) # [B, target_len, 1]
```

차원 포인트:
- Encoder는 입력 sequence를 hidden state로 요약한다.
- Decoder는 이전 예측값을 다음 입력으로 사용해 multi-step forecast를 만든다.
- teacher forcing을 쓰는 구조가 아니므로 학습 중에도 자기 예측을 다음 step에 넣는다.

---

## 6.20 NCF and NGCF Recommenders

관련 노트북:
- `ai-specialist-2026-time/7_8-recsys-practice/RecSys_NCF.ipynb`
- `ai-specialist-2026-time/7_8-recsys-practice/RecSys_GCF_sol.ipynb`

### Model Structure

NCF:

```text
user id: [B]              movie id: [B]
   |                         |
   v                         v
User Embedding: [B, 32]   Movie Embedding: [B, 32]
   |                         |
   +-----------+-------------+
               |
               v
Concat: [B, 64]
               |
               v
Linear 64->32 + ReLU: [B, 32]
               |
               v
Linear 32->1: [B, 1]
               |
               v
rating prediction
```

NGCF:

```text
ratings dataframe
  |
  v
edge_index: [2, E]
src=user_id, dst=movie_id + num_users
  |
  v
node embedding table: [U + I, D0]
  |
  v
NGCF Layer 1 message passing: [U + I, D1]
  |
  v
NGCF Layer 2 message passing: [U + I, D2]
  |
  v
concat all layer outputs: [U + I, D0 + D1 + D2]
  |
  +--> user_features: [U, D_final]
  |
  +--> item_features: [I, D_final]
```

### Annotated Code

NCF:

```python
def forward(self, users, movies, ratings=None):
    # users, movies: [B]
    user_embedding = self.user_embedding(users)     # [B, 32]
    movie_embedding = self.movie_embedding(movies)  # [B, 32]

    # [B, 32] + [B, 32] -> [B, 64]
    input_embedding = torch.cat([user_embedding, movie_embedding], dim=1)

    hidden_feature = self.relu(self.fc1(input_embedding)) # [B, 32]
    output = self.fc2(hidden_feature)                     # [B, 1]
    return output
```

NGCF:

```python
def create_edge_index(df, rating_threshold=1.0):
    src, dst = [], []
    for _, row in df.iterrows():
        if row["rating"] >= rating_threshold:
            src.append(row["userId"])
            dst.append(row["movieId"] + num_users)
    return torch.tensor([src, dst], dtype=torch.long) # [2, E]
```

```python
def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    # all embeddings: [B, D_final]
    pos_scores = torch.sum(user_emb * pos_item_emb, dim=1) # [B]
    neg_scores = torch.sum(user_emb * neg_item_emb, dim=1) # [B]

    # maximize pos_scores - neg_scores
    return -torch.mean(F.logsigmoid(pos_scores - neg_scores))
```

차원 포인트:
- NCF는 explicit rating prediction에 자연스럽고 MSE/RMSE로 평가할 수 있다.
- NGCF는 ranking 추천에 자연스럽고 BPR, Recall@K, Precision@K, NDCG@K가 중요하다.
- `movieId + num_users` offset은 user node와 item node의 index 공간을 겹치지 않게 만든다.

---

# 7. 리뷰 보강: 시험에 나올 만한 함정

## 7.1 Loss Function 함정

### CrossEntropyLoss

```text
input : raw logits [N, C]
target: class id   [N]
```

주의:
- `CrossEntropyLoss`는 내부에 `log_softmax`와 `NLLLoss`가 합쳐져 있다.
- 따라서 모델 출력에 `softmax`를 먼저 적용하면 안 된다.
- GPT LM에서는 `[B, T, V] -> [B*T, V]`, target은 `[B, T] -> [B*T]`로 펴서 넣는다.

### NLLLoss

```text
input : log probabilities [N, C]
target: class id          [N]
```

주의:
- `F.nll_loss()`는 raw logits가 아니라 `F.log_softmax(logits, dim=-1)` 결과를 기대한다.
- ViT 노트북처럼 모델이 raw logits를 반환한다면 `F.cross_entropy(output, target)`가 더 맞다.

### BCEWithLogitsLoss

```text
input : raw binary logits [B, 1, H, W]
target: binary mask       [B, 1, H, W]
```

주의:
- `BCEWithLogitsLoss`는 내부에 sigmoid가 포함되어 있다.
- 학습 loss에 넣기 전에 `sigmoid(logits)`를 하면 중복 적용이다.
- 시각화나 thresholding할 때만 `torch.sigmoid(logits) > 0.5`를 사용한다.

---

## 7.2 Mask 함정

### Causal Mask vs Padding Mask

```text
Causal mask:
  purpose: block future tokens
  shape  : [T, T] or broadcast to [B, heads, T, T]
  value  : future scores -> -inf before softmax

Padding mask:
  purpose: ignore pad tokens
  shape  : [B, T]
  value  : pad target -> -100 for loss, or attention score -> -inf
```

시험 포인트:
- causal mask는 GPT의 auto-regressive 성질을 보장한다.
- padding mask는 batch 길이를 맞추기 위해 넣은 pad가 학습에 영향을 주지 않게 한다.
- `ignore_index=-100`은 CrossEntropyLoss에서 해당 위치를 loss 계산에서 제외한다.

Instruction tuning에서 첫 번째 pad/EOS를 남기는 이유:

```python
mask = targets == pad_token_id
indices = torch.nonzero(mask).squeeze()
if indices.numel() > 1:
    targets[indices[1:]] = -100
```

첫 번째 `pad_token_id`는 문장 종료 토큰처럼 학습시키고, 그 뒤의 padding만 무시한다는 의도다.

---

## 7.3 DPO 함정

DPO 공식 핵심:

```text
policy_logratio    = log p_policy(chosen) - log p_policy(rejected)
reference_logratio = log p_ref(chosen)    - log p_ref(rejected)
logits             = policy_logratio - reference_logratio
loss               = -logsigmoid(beta * logits)
```

시험 포인트:
- `chosen - rejected` 순서가 바뀌면 preference 방향이 반대가 된다.
- reference model은 gradient를 계산하지 않는다.
- prompt token은 chosen/rejected 모두 공통이므로 보통 loss 비교에서 제외한다.

마스크 주의:

```python
mask[:prompt.shape[0] + 2] = False
```

이 코드는 노트북의 단순화된 구현이다. 실제로는 `"\n\n### Response:\n"` 같은 response header가 몇 token인지 tokenizer로 정확히 계산해 prompt 영역을 마스킹하는 편이 안전하다.

---

## 7.4 Attention 함정

Scaled dot-product attention:

```text
Q: [B, heads, T, Dh]
K: [B, heads, T, Dh]
V: [B, heads, T, Dh]

scores = Q @ K^T / sqrt(Dh)
scores: [B, heads, T, T]
weights = softmax(scores, dim=-1)
out = weights @ V
out: [B, heads, T, Dh]
```

시험 포인트:
- scaling은 `sqrt(Dh)`로 나눈다. head를 나눈 뒤의 차원이다.
- softmax는 key/token 축인 마지막 차원에 적용한다.
- mask는 softmax 전에 `-inf`로 적용한다.
- `register_buffer`로 저장한 mask는 parameter가 아니지만 device 이동과 저장에 포함된다.

---

## 7.5 ViT 함정

Patch 수:

```text
N = (image_height / patch_height) * (image_width / patch_width)
patch_dim = channels * patch_height * patch_width
```

예:

```text
image: [B, 3, 32, 32]
patch_size: 4
N = 8 * 8 = 64
patch_dim = 3 * 4 * 4 = 48
with CLS: [B, 65, D]
attention: [B, heads, 65, 65]
```

시험 포인트:
- patch size가 작아질수록 token 수가 늘고 attention 비용이 커진다.
- CNN은 locality bias가 강하고, ViT는 attention으로 전역 관계를 직접 학습한다.
- raw logits 반환이면 `CrossEntropyLoss`, log-softmax 반환이면 `NLLLoss`다.

---

## 7.6 U-Net 함정

```text
encoder output x4 ----+
                      v
bottleneck x5 -> up -> concat -> conv
```

시험 포인트:
- `x5 = self.down4(x4)`가 bottleneck이다.
- skip connection은 add가 아니라 channel concat이다.
- concat 축은 `dim=1`, 즉 channel 축이다.
- output shape은 input image와 같은 공간 크기를 갖는 `[B, K, H, W]`다.
- binary segmentation이면 `K=1`, multi-class segmentation이면 `K=num_classes`다.

---

## 7.7 DDPM 함정

Forward diffusion:

```text
x_t = sqrt(alpha_bar_t) * x_0
    + sqrt(1 - alpha_bar_t) * epsilon
```

시험 포인트:
- 모델이 예측하는 것은 보통 clean image `x_0`가 아니라 noise `epsilon`이다.
- `t`는 `[B]`이고, 계수는 `[B, 1, 1, 1]`로 reshape되어 image에 broadcast된다.
- sampling은 `x_T`에서 시작해 `x_0` 방향으로 한 단계씩 denoise한다.

---

## 7.8 DETR 함정

```text
pred_logits: [B, Q, C+1]
pred_boxes : [B, Q, 4]
```

시험 포인트:
- `Q`는 object query 수다.
- `C+1`에서 `+1`은 no-object class다.
- `pred_boxes`는 normalized `[cx, cy, w, h]`다.
- 시각화할 때는 `[xmin, ymin, xmax, ymax]`로 바꾸고 이미지 크기로 rescale한다.
- DETR은 NMS가 핵심인 전통 detector와 달리 set prediction 관점이다.

---

## 7.9 Pruning 함정

```text
importance = abs(weight)
threshold = kth smallest value
mask = importance > threshold
```

시험 포인트:
- fine-grained pruning은 weight 원소 단위다.
- structured pruning은 vector/kernel/channel 단위라 실제 하드웨어 가속과 연결되기 쉽다.
- `torch.kthvalue`의 `k`는 1부터 시작한다. `sparsity=0`은 별도 처리해야 한다.
- threshold와 같은 값이 많으면 목표 sparsity와 실제 sparsity가 다를 수 있다.
- pruning 후 fine-tuning 중에는 mask를 계속 적용해야 0인 weight가 다시 살아나지 않는다.

---

## 7.10 Quantization 함정

기본 공식:

```text
q = round(x / scale) + zero_point
x_hat = (q - zero_point) * scale
```

시험 포인트:
- activation은 비대칭 분포가 많아 `scale + zero_point`를 쓴다.
- weight는 대칭 분포로 보고 `zero_point=0`을 쓰는 경우가 많다.
- per-channel weight quantization은 output channel별 scale을 갖는다.
- bias는 보통 int32로 양자화하고 `bias_scale = input_scale * weight_scale`을 쓴다.
- quantized matmul/conv는 int8 곱셈을 int32 accumulator에 누적한 뒤 다시 output scale로 requantize한다.

---

## 7.11 KD 함정

Soft target KD:

```text
teacher_logits / T -> soft teacher distribution
student_logits / T -> soft student distribution
loss = KL(teacher || student) * T^2 + CE(student, label)
```

시험 포인트:
- teacher는 `eval()`이고 `torch.no_grad()`로 감싼다.
- `T`가 클수록 확률 분포가 부드러워져 class 간 관계가 드러난다.
- 구현 안정성은 `softmax(...).log()`보다 `log_softmax()`가 더 좋다.
- feature distillation에서는 teacher/student feature shape이 다르면 regressor로 맞춘다.

안정적인 형태 예:

```python
teacher_prob = F.softmax(teacher_logits / T, dim=-1)
student_log_prob = F.log_softmax(student_logits / T, dim=-1)
kd_loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (T ** 2)
```

---

## 7.12 LoRA 함정

```text
delta W = A @ B
A: [In, R]
B: [R, Out]
```

시험 포인트:
- LoRA는 base weight를 크게 업데이트하지 않고 low-rank adapter만 학습한다.
- 노트북 구현 기준 forward는 `x @ A @ B`다.
- trainable parameter 수는 `In*R + R*Out`이다.
- `B`를 0으로 초기화하면 학습 시작 시 LoRA branch 출력이 0이라 base model 동작을 보존한다.

---

## 7.13 LLM Pruning/Quantization 함정

LLM Linear:

```text
W: [Out, In]
x: [B, T, In]
y = x @ W^T
y: [B, T, Out]
```

Wanda:

```text
importance = abs(W) * input_activation_norm
abs(W): [Out, In]
input_activation_norm: [In]
importance: [Out, In]
```

시험 포인트:
- `lm_head`는 pruning에서 제외하는 경우가 많다.
- Wanda는 weight 크기만 보지 않고 입력 activation scale도 본다.
- AWQ는 activation outlier channel을 기준으로 중요한 weight column을 보호한다.
- SmoothQuant는 activation의 큰 값을 줄이고 weight 쪽으로 scale 부담을 옮긴다.
- group-wise quantization은 `[Out, In]`을 `[-1, group_size]`로 reshape해 group마다 scale/zero를 둔다.

---

## 7.14 RAG 함정

RAG는 단일 NN 구조가 아니라 pipeline이다.

```text
chunking -> embedding -> vector search -> context -> LLM reader -> answer
```

시험 포인트:
- embedding model은 text를 vector로 바꾸는 neural model이다.
- retriever가 잘못된 chunk를 가져오면 reader가 아무리 좋아도 답이 틀릴 수 있다.
- chunk size가 너무 작으면 맥락이 끊기고, 너무 크면 검색이 둔해진다.
- KG/API augmentation은 숫자, entity lookup, 최신 structured fact에 강하다.

---

## 7.15 Time Series Forecasting 함정

Sliding window:

```text
input  = series[t : t + sequence_length]
target = series[t + sequence_length]
```

시험 포인트:
- train/test split은 시간 순서를 보존해야 한다. 시계열에서 무작위 split은 미래 정보 leakage를 만들 수 있다.
- scaler는 train data에만 `fit`하고, test data에는 `transform`만 적용한다.
- LSTM/RNN 출력 `out`은 `[B, T, hidden]`이고, next-step 예측은 보통 마지막 time step `out[:, -1, :]`를 사용한다.
- `batch_first=True`이면 입력 shape은 `[B, T, F]`다. 기본값은 `[T, B, F]`라 헷갈리기 쉽다.
- Conv1D는 `[B, C, T]`를 기대하므로 시계열 입력 `[B, T, F]`를 `transpose(1, 2)`해야 한다.
- MAPE는 실제값이 0 또는 0에 가까우면 매우 불안정하다.

---

## 7.16 Recommender System 함정

NCF:

```text
user id -> user embedding
item id -> item embedding
concat -> MLP -> rating
```

NGCF:

```text
user/item nodes + interaction edges -> message passing -> ranking score
```

시험 포인트:
- userId/movieId 원본 값은 연속 index가 아닐 수 있으므로 LabelEncoder/dictionary mapping이 필요하다.
- NCF의 `nn.Embedding` 입력은 long dtype id tensor여야 한다.
- explicit rating prediction은 MSE/RMSE가 자연스럽다.
- implicit feedback ranking은 BPR loss, Recall@K, Precision@K, NDCG@K가 더 자연스럽다.
- BPR은 positive item score가 negative item score보다 커지도록 `-logsigmoid(pos - neg)`를 최소화한다.
- NGCF에서 item node index에 `num_users`를 더하는 이유는 user node와 item node index 공간을 분리하기 위해서다.
- Recall@K는 relevant item을 얼마나 놓치지 않는지, Precision@K는 추천 리스트 Top-K가 얼마나 정확한지, NDCG@K는 맞춘 item의 rank까지 반영한다.
