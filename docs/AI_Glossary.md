# AI 용어집

이 문서는 `llm_hands_on` 실습을 따라가며 자주 만나는 AI, 머신러닝, 딥러닝, Transformer, LLM 용어를 한곳에 모은 참고 문서입니다.
용어를 외우기보다 "무엇을 계산하는가", "코드에서 어디에 나타나는가", "헷갈리기 쉬운 점은 무엇인가"를 기준으로 읽으면 좋습니다.

## 빠른 사용법

- 처음 읽을 때는 `AI 기초 -> 학습 과정 -> Transformer/LLM -> Fine-tuning/Alignment` 순서로 봅니다.
- 노트북을 읽다가 모르는 단어가 나오면 해당 용어의 `실습 연결`을 먼저 확인합니다.
- `logits`, `embedding`, `attention`, `loss`, `shape`처럼 코드에 직접 등장하는 용어는 특히 중요합니다.

## 목차

1. [AI 기본 개념](#1-ai-기본-개념)
2. [데이터와 표현](#2-데이터와-표현)
3. [모델과 신경망](#3-모델과-신경망)
4. [학습 과정](#4-학습-과정)
5. [NLP와 LLM 기본](#5-nlp와-llm-기본)
6. [Transformer와 Attention](#6-transformer와-attention)
7. [텍스트 생성과 프롬프트](#7-텍스트-생성과-프롬프트)
8. [Fine-tuning과 Alignment](#8-fine-tuning과-alignment)
9. [평가와 운영](#9-평가와-운영)
10. [안전성, 윤리, 보안](#10-안전성-윤리-보안)
11. [헷갈리기 쉬운 용어 비교](#11-헷갈리기-쉬운-용어-비교)

---

## 1. AI 기본 개념

### AI, Artificial Intelligence, 인공지능

AI는 사람이 지능적으로 수행한다고 여겨지는 작업을 컴퓨터가 수행하도록 만드는 기술 전체를 말합니다. 이미지 인식, 음성 인식, 번역, 추천, 문서 요약, 코드 생성, 대화형 assistant 등이 모두 AI 범위에 들어갑니다.

- 핵심 질문: 사람이 직접 규칙을 모두 짜지 않아도, 데이터와 모델을 이용해 판단이나 생성을 할 수 있는가?
- 실습 연결: 이 repo에서는 AI 중에서도 텍스트를 이해하고 생성하는 LLM을 중심으로 다룹니다.
- 주의점: AI는 하나의 알고리즘 이름이 아니라 큰 분야명입니다. 머신러닝, 딥러닝, LLM은 AI 안에 포함됩니다.

### Machine Learning, 머신러닝

머신러닝은 사람이 규칙을 직접 코딩하는 대신, 데이터에서 패턴을 학습해 예측 규칙을 만드는 방식입니다. 예를 들어 스팸 메일을 판별할 때 "특정 단어가 있으면 스팸" 같은 규칙을 직접 나열하지 않고, 많은 메일과 라벨을 보고 모델이 경향을 학습하게 합니다.

- 핵심 질문: 데이터로부터 예측 함수를 배울 수 있는가?
- 실습 연결: Chapter 5의 사전학습은 텍스트 데이터에서 다음 토큰 예측 패턴을 학습하는 머신러닝입니다.
- 주의점: 머신러닝 모델은 학습 데이터의 패턴을 배웁니다. 데이터가 편향되어 있으면 모델도 편향될 수 있습니다.

### Deep Learning, 딥러닝

딥러닝은 여러 층의 신경망을 사용해 복잡한 패턴을 학습하는 머신러닝 방법입니다. 이미지, 음성, 텍스트처럼 사람이 직접 좋은 feature를 설계하기 어려운 데이터에서 강력합니다.

- 핵심 질문: 여러 층의 계산을 쌓아 단순한 입력을 높은 수준의 표현으로 바꿀 수 있는가?
- 실습 연결: GPT는 TransformerBlock을 여러 층 쌓은 딥러닝 모델입니다.
- 주의점: "깊다"는 말은 보통 layer가 많다는 뜻입니다. 하지만 layer 수만 많다고 좋은 모델이 되는 것은 아닙니다.

### Generative AI, 생성형 AI

생성형 AI는 기존 데이터를 분류하거나 점수화하는 것에서 더 나아가, 새로운 텍스트, 이미지, 오디오, 코드 등을 생성하는 AI입니다. LLM은 대표적인 텍스트 생성형 AI입니다.

- 핵심 질문: 입력 조건에 맞는 새 결과물을 만들 수 있는가?
- 실습 연결: Chapter 4의 생성 루프는 다음 토큰을 반복해서 붙여 새 문장을 만듭니다.
- 주의점: 생성 결과가 그럴듯해 보여도 항상 사실이라는 뜻은 아닙니다. 검증이 필요합니다.

### Model, 모델

모델은 입력을 받아 출력으로 바꾸는 계산 구조입니다. 머신러닝에서는 이 계산 구조 안에 학습 가능한 숫자, 즉 parameter가 들어 있습니다.

- 예: `GPTModel(input_ids) -> logits`
- 실습 연결: Chapter 4의 `GPTModel`은 token id를 받아 다음 토큰 후보 점수를 출력합니다.
- 주의점: 모델 구조와 학습된 parameter는 구분해야 합니다. 같은 구조라도 parameter가 다르면 다른 모델입니다.

### Parameter, 파라미터

파라미터는 모델이 학습을 통해 조정하는 숫자입니다. 선형층의 weight와 bias, embedding table의 값 등이 대표적입니다.

- 핵심 질문: 학습 중 optimizer가 바꾸는 값인가?
- 실습 연결: `loss.backward()`가 gradient를 계산하고 `optimizer.step()`이 parameter를 수정합니다.
- 주의점: 사용자가 직접 정하는 값인 hyperparameter와 다릅니다.

### Hyperparameter, 하이퍼파라미터

하이퍼파라미터는 모델이 학습으로 자동 결정하지 않고 사람이 설정하는 값입니다. learning rate, batch size, epoch 수, layer 수, embedding 차원 등이 해당합니다.

- 핵심 질문: 학습 전에 사람이 정하는 설정값인가?
- 실습 연결: HTML 시각화의 layer 수, embedding 차원, batch size 슬라이더가 하이퍼파라미터에 가깝습니다.
- 주의점: 하이퍼파라미터는 성능과 비용에 큰 영향을 줍니다.

---

## 2. 데이터와 표현

### Dataset, 데이터셋

데이터셋은 모델 학습, 검증, 평가에 사용하는 데이터 묶음입니다. 텍스트 모델에서는 문장, 문서, 질문/답변, 선호 응답 쌍 등이 데이터셋이 됩니다.

- 실습 연결: Chapter 2의 dataset은 긴 텍스트를 잘라 input/target 토큰 쌍을 만듭니다.
- 코드 관점: PyTorch에서는 보통 `Dataset.__len__()`과 `Dataset.__getitem__()`을 구현합니다.
- 주의점: 데이터셋 품질이 낮으면 모델이 아무리 커도 결과가 나빠집니다.

### Sample, 샘플

샘플은 데이터셋 안의 개별 항목 하나입니다. 텍스트 분류에서는 문장 하나와 라벨 하나가 한 샘플이고, 언어 모델 학습에서는 토큰 시퀀스 하나가 한 샘플입니다.

- 예: `("이 메일은 스팸입니다", spam)`
- 실습 연결: DataLoader가 여러 샘플을 모아 batch를 만듭니다.
- 주의점: 샘플 하나의 길이가 모두 같지 않으면 padding이나 truncation이 필요합니다.

### Feature, 특징

Feature는 모델이 입력으로 사용하는 정보입니다. 전통적인 머신러닝에서는 사람이 feature를 설계하는 경우가 많았고, 딥러닝에서는 raw data를 넣으면 모델이 내부 표현을 학습하는 경우가 많습니다.

- 예: 문서 길이, 단어 빈도, 이미지 픽셀, token id
- 실습 연결: LLM에서는 token id가 embedding을 거쳐 모델이 사용할 feature vector가 됩니다.
- 주의점: LLM에서는 사람이 feature를 직접 만들기보다 tokenizer와 embedding을 통해 표현을 만듭니다.

### Label, 라벨

라벨은 supervised learning에서 정답 역할을 하는 값입니다. 분류에서는 class id, 회귀에서는 숫자 값, 언어 모델에서는 다음 토큰 id가 라벨이 될 수 있습니다.

- 실습 연결: Chapter 6 분류 fine-tuning에서는 spam/ham 같은 class label을 사용합니다.
- 코드 관점: `CrossEntropyLoss(logits, labels)`에서 `labels`가 정답입니다.
- 주의점: 언어 모델의 next-token target도 넓은 의미의 label입니다.

### Tensor, 텐서

텐서는 숫자를 담는 다차원 배열입니다. PyTorch의 핵심 자료구조이며, scalar, vector, matrix를 모두 포함하는 일반화된 개념입니다.

- 예: `[batch, seq_len]`, `[batch, seq_len, emb_dim]`
- 실습 연결: 노트북을 읽을 때 가장 먼저 확인할 것은 각 tensor의 shape입니다.
- 주의점: 텐서의 숫자 값보다 shape가 먼저 이해되어야 코드 흐름을 따라갈 수 있습니다.

### Shape, 텐서 모양

Shape는 텐서의 각 차원 크기입니다. LLM 코드에서는 shape를 잘못 이해하면 matrix multiplication, attention score, loss 계산이 모두 헷갈립니다.

- 예: `input_ids: [batch, seq_len]`
- 예: `logits: [batch, seq_len, vocab_size]`
- 실습 연결: Chapter 3-5는 shape를 추적하면 대부분의 계산 흐름이 정리됩니다.
- 주의점: PyTorch는 차원 순서가 중요합니다. `[batch, seq, emb]`와 `[seq, batch, emb]`는 다릅니다.

### Vector, 벡터

벡터는 숫자의 1차원 배열입니다. 단어, 토큰, 문장, 이미지 조각 등 다양한 대상을 숫자 벡터로 표현할 수 있습니다.

- 실습 연결: Chapter 1의 단어 벡터와 Chapter 2 이후의 token embedding이 모두 벡터 표현입니다.
- 직관: 의미가 비슷한 단어나 문장은 벡터 공간에서 가까운 위치에 놓이도록 학습될 수 있습니다.
- 주의점: 벡터는 사람이 읽기 쉬운 의미 단위가 아니라 모델이 계산하기 좋은 숫자 표현입니다.

### Embedding, 임베딩

임베딩은 단어나 토큰 같은 discrete ID를 dense vector로 바꾼 표현입니다. 토큰 id 자체는 단순한 번호라 의미 계산을 하기 어렵기 때문에, 모델은 embedding vector를 사용합니다.

- 예: token id `15496` -> `[0.12, -0.03, ...]`
- 실습 연결: Chapter 4의 `tok_emb`와 `pos_emb`가 embedding입니다.
- 주의점: token embedding은 단어 사전처럼 사람이 직접 뜻을 넣은 것이 아니라 학습으로 만들어진 숫자 표입니다.

### Vocabulary, 어휘집

Vocabulary는 tokenizer와 모델이 알고 있는 token 목록입니다. 각 token은 고유한 integer ID를 가집니다.

- 예: `"hello"` -> 31373
- 실습 연결: `vocab_size`는 모델이 다음 토큰 후보로 점수를 내야 하는 전체 token 개수입니다.
- 주의점: vocabulary가 크면 표현할 수 있는 token은 많아지지만 output layer 비용도 커집니다.

---

## 3. 모델과 신경망

### Neural Network, 신경망

신경망은 여러 층의 선형 변환과 비선형 함수를 조합해 입력을 출력으로 바꾸는 모델입니다. 딥러닝 모델은 대부분 신경망입니다.

- 실습 연결: GPT는 embedding, attention, FFN, layer norm 등이 결합된 큰 신경망입니다.
- 직관: 층을 지나며 원본 입력이 점점 더 유용한 내부 표현으로 바뀝니다.
- 주의점: 신경망의 각 층은 단독으로 해석하기 어렵고, 전체 흐름에서 의미가 생깁니다.

### Layer, 층

Layer는 모델을 구성하는 계산 단위입니다. `Linear`, `LayerNorm`, `Dropout`, `TransformerBlock` 등이 layer 역할을 합니다.

- 실습 연결: Chapter 4에서 `TransformerBlock`을 여러 개 쌓아 GPT 본체를 만듭니다.
- 주의점: "층 수"가 늘면 모델 표현력은 커질 수 있지만, 학습 비용과 메모리 사용량도 증가합니다.

### Linear Layer, 선형층

선형층은 입력 벡터에 weight matrix를 곱하고 bias를 더해 출력 벡터로 바꾸는 층입니다.

- 수식: `y = xW + b`
- 실습 연결: attention의 Q/K/V projection, FFN, classification head, output head 모두 Linear와 관련됩니다.
- 주의점: Linear layer만 여러 개 쌓아도 중간에 비선형 함수가 없으면 전체는 결국 하나의 선형 변환과 같습니다.

### Activation Function, 활성화 함수

활성화 함수는 신경망에 비선형성을 넣는 함수입니다. ReLU, GELU, sigmoid, tanh 등이 있습니다.

- 실습 연결: GPT의 FFN에서는 보통 GELU 계열 활성화가 사용됩니다.
- 직관: 비선형성이 있어야 모델이 단순한 직선 관계보다 복잡한 패턴을 학습할 수 있습니다.
- 주의점: activation의 종류는 학습 안정성과 성능에 영향을 줍니다.

### MLP, Multi-Layer Perceptron

MLP는 여러 Linear layer와 activation을 쌓은 feed-forward 신경망입니다. Transformer에서는 FFN이라고도 부르며, 각 토큰 위치에 독립적으로 적용됩니다.

- 실습 연결: Chapter 4의 `FeedForward`는 `emb_dim -> 4 * emb_dim -> GELU -> emb_dim` 흐름입니다.
- 주의점: Attention은 토큰 간 정보를 섞고, FFN/MLP는 각 토큰의 표현을 가공합니다.

### Feed Forward Network, FFN

TransformerBlock 안에서 attention 뒤에 있는 MLP 계열 모듈입니다. 토큰 간 정보를 섞는 것이 아니라, 각 토큰의 hidden vector를 더 풍부하게 변환합니다.

- 실습 연결: Chapter 4의 TransformerBlock 내부 구성 요소입니다.
- 직관: attention이 "어디를 볼지" 정한다면, FFN은 "본 정보를 어떻게 가공할지" 담당합니다.
- 주의점: FFN은 보통 모델 파라미터의 상당 부분을 차지합니다.

### Residual Connection, 잔차 연결

Residual connection은 layer 출력에 원래 입력을 더하는 구조입니다. 깊은 모델에서 gradient가 잘 흐르게 하고, 기존 정보를 잃지 않도록 도와줍니다.

- 수식: `output = layer(x) + x`
- 실습 연결: Chapter 4의 TransformerBlock에는 attention 뒤와 FFN 뒤에 residual add가 있습니다.
- 주의점: residual이 없으면 깊은 모델은 학습이 불안정해질 수 있습니다.

### Layer Normalization, LayerNorm

LayerNorm은 각 샘플의 feature 차원 기준으로 값을 정규화해 학습을 안정화하는 층입니다. Transformer에서는 거의 필수적으로 사용됩니다.

- 실습 연결: Chapter 4의 `LayerNorm`과 `final_norm`
- 직관: 층을 지날 때 값의 스케일이 너무 커지거나 작아지는 문제를 줄입니다.
- 주의점: BatchNorm과 다르게 batch 전체 통계가 아니라 각 샘플 내부 feature 통계를 사용합니다.

### Dropout

Dropout은 학습 중 일부 값을 랜덤하게 0으로 만들어 모델이 특정 경로에 과하게 의존하지 않게 하는 regularization 기법입니다.

- 실습 연결: GPT 구현에서 embedding dropout, attention dropout, residual dropout 형태로 나타날 수 있습니다.
- 주의점: 학습 모드에서는 적용되고 평가 모드에서는 꺼집니다. PyTorch에서는 `model.train()`과 `model.eval()` 차이가 중요합니다.

### GPU, CUDA

GPU는 많은 숫자 연산을 병렬로 빠르게 처리하는 장치입니다. CUDA는 NVIDIA GPU에서 딥러닝 연산을 실행하기 위한 플랫폼입니다.

- 실습 연결: PyTorch에서 `.to(device)`로 tensor와 model을 GPU로 옮깁니다.
- 주의점: model과 tensor가 서로 다른 device에 있으면 에러가 납니다.

---

## 4. 학습 과정

### Training, 학습

학습은 모델 출력과 정답을 비교해 loss를 계산하고, loss가 줄어들도록 parameter를 반복적으로 수정하는 과정입니다.

- 기본 흐름: forward -> loss -> backward -> optimizer step
- 실습 연결: Chapter 5의 pretraining loop가 이 구조입니다.
- 주의점: 학습은 단순 실행이 아니라 parameter가 바뀌는 과정입니다.

### Inference, 추론

추론은 학습된 모델을 사용해 새로운 입력에 대한 출력을 얻는 과정입니다. 이때는 보통 parameter를 업데이트하지 않습니다.

- 실습 연결: Chapter 4의 text generation은 inference입니다.
- 코드 관점: PyTorch에서는 보통 `model.eval()`과 `torch.no_grad()`를 사용합니다.
- 주의점: 추론에서도 sampling 설정에 따라 출력이 달라질 수 있습니다.

### Supervised Learning, 지도학습

지도학습은 입력과 정답 라벨이 함께 있는 데이터로 학습하는 방식입니다. 분류, 회귀, instruction tuning 등이 여기에 속할 수 있습니다.

- 실습 연결: Chapter 6의 분류 fine-tuning은 텍스트와 class label을 사용합니다.
- 주의점: 라벨 품질이 성능을 크게 좌우합니다.

### Unsupervised Learning, 비지도학습

비지도학습은 명시적인 정답 라벨 없이 데이터의 구조나 패턴을 찾는 학습입니다. clustering, dimensionality reduction 등이 대표적입니다.

- LLM 연결: 엄밀히 말해 LLM 사전학습은 다음 토큰을 정답으로 만드는 self-supervised learning에 가깝습니다.
- 주의점: "라벨 파일이 없다"와 "학습 목표가 없다"는 같은 말이 아닙니다.

### Self-Supervised Learning, 자기지도학습

자기지도학습은 데이터 자체에서 정답을 만들어 학습하는 방식입니다. LLM은 문장에서 다음 토큰을 정답으로 삼아 학습합니다.

- 실습 연결: Chapter 2의 input/target shift가 자기지도학습의 핵심입니다.
- 예: `input = [오늘, 날씨가]`, `target = [날씨가, 좋다]`
- 주의점: 사람이 라벨을 달지 않아도 학습 signal을 만들 수 있다는 점이 중요합니다.

### Reinforcement Learning, 강화학습

강화학습은 agent가 행동을 선택하고 보상을 받아 더 좋은 행동 정책을 학습하는 방식입니다.

- LLM 연결: RLHF에서는 사람이 선호한 답변을 바탕으로 reward model을 만들고, policy model을 조정합니다.
- 주의점: 일반적인 next-token 학습과는 학습 방식이 다릅니다.

### Loss, 손실

Loss는 모델이 얼마나 틀렸는지를 나타내는 숫자입니다. 학습은 이 숫자를 줄이는 방향으로 parameter를 수정합니다.

- 실습 연결: `CrossEntropyLoss`는 분류와 next-token prediction에서 자주 사용됩니다.
- 직관: loss가 낮아진다는 것은 모델이 학습 데이터의 정답을 더 잘 맞힌다는 뜻입니다.
- 주의점: train loss만 낮고 validation loss가 높으면 overfitting일 수 있습니다.

### Cross Entropy Loss, 교차 엔트로피 손실

Cross entropy는 정답 class의 확률을 높이도록 학습하는 loss입니다. LLM에서는 각 위치의 정답 다음 토큰 확률을 높이는 데 사용됩니다.

- 실습 연결: Chapter 5의 `torch.nn.functional.cross_entropy`
- 입력: 예측 점수 `logits`
- 정답: class id 또는 target token id
- 주의점: PyTorch의 `cross_entropy`는 내부에서 `log_softmax`를 처리하므로 보통 logits를 그대로 넣습니다.

### Gradient, 그래디언트

Gradient는 parameter를 어느 방향으로 얼마나 바꾸면 loss가 증가하거나 감소하는지 알려주는 값입니다.

- 실습 연결: `loss.backward()`가 각 parameter의 `.grad`를 채웁니다.
- 직관: gradient는 학습 방향을 알려주는 기울기입니다.
- 주의점: gradient 자체가 parameter를 바꾸지는 않습니다. 실제 수정은 optimizer가 합니다.

### Backpropagation, 역전파

역전파는 loss에서 시작해 계산 그래프를 거꾸로 따라가며 각 parameter의 gradient를 계산하는 알고리즘입니다.

- 실습 연결: PyTorch에서는 `loss.backward()`로 실행됩니다.
- 주의점: backward 전에 forward 계산이 있어야 하며, gradient를 누적하지 않으려면 `optimizer.zero_grad()`가 필요합니다.

### Optimizer, 옵티마이저

Optimizer는 gradient를 사용해 parameter를 실제로 업데이트하는 알고리즘입니다. SGD, Adam, AdamW 등이 있습니다.

- 실습 연결: `optimizer.step()`
- 직관: gradient가 "어느 방향으로 바꿀지"라면 optimizer는 "얼마나, 어떤 방식으로 바꿀지" 결정합니다.
- 주의점: learning rate가 너무 크면 학습이 불안정하고, 너무 작으면 학습이 느립니다.

### Learning Rate, 학습률

Learning rate는 parameter를 한 번에 얼마나 크게 업데이트할지 정하는 하이퍼파라미터입니다.

- 실습 연결: optimizer 생성 시 `lr=...`로 지정합니다.
- 직관: 발걸음 크기입니다. 너무 크면 목표를 지나치고, 너무 작으면 오래 걸립니다.
- 주의점: LLM 학습에서는 warmup, decay 같은 schedule을 함께 쓰는 경우가 많습니다.

### Batch, Step, Epoch

Batch는 한 번에 모델에 넣는 샘플 묶음입니다. Step은 batch 하나로 parameter update를 한 번 수행하는 단위입니다. Epoch는 전체 학습 데이터를 한 번 모두 본 단위입니다.

- 예: 데이터 1,000개, batch size 100이면 1 epoch는 10 step입니다.
- 실습 연결: Chapter 5의 DataLoader와 training loop
- 주의점: 대규모 LLM에서는 전체 데이터가 너무 커서 epoch보다 step 수를 기준으로 말하는 경우가 많습니다.

### Overfitting, 과적합

과적합은 모델이 학습 데이터에는 잘 맞지만 새로운 데이터에는 잘 일반화하지 못하는 상태입니다.

- 신호: train loss는 낮은데 validation loss가 높거나 증가합니다.
- 대응: 더 많은 데이터, regularization, early stopping, dropout, 모델 크기 조정
- 주의점: 학습 성능만 보고 모델이 좋다고 판단하면 위험합니다.

### Underfitting, 과소적합

과소적합은 모델이 학습 데이터의 패턴조차 충분히 배우지 못한 상태입니다.

- 신호: train loss와 validation loss가 모두 높습니다.
- 원인: 모델이 너무 작음, 학습 부족, learning rate 문제, feature 부족
- 주의점: overfitting과 반대 문제이므로 대응도 다릅니다.

### Validation, 검증

검증은 학습 중 모델의 일반화 성능을 확인하기 위해 별도 validation set으로 평가하는 과정입니다.

- 실습 연결: Chapter 5에서 train loss와 validation loss를 비교합니다.
- 주의점: validation set으로 하이퍼파라미터를 계속 조정하면 validation에도 과적합될 수 있습니다.

---

## 5. NLP와 LLM 기본

### NLP, Natural Language Processing, 자연어 처리

NLP는 사람이 쓰는 언어를 컴퓨터가 처리하도록 하는 분야입니다. 형태소 분석, 번역, 요약, 질의응답, 감성 분석, 문서 검색 등이 포함됩니다.

- LLM 연결: LLM은 NLP 문제를 범용 텍스트 생성 문제로 많이 통합합니다.
- 주의점: 한국어는 띄어쓰기, 조사, 어미 변화가 있어 tokenizer 동작을 특히 확인해야 합니다.

### Language Model, 언어 모델

언어 모델은 텍스트의 확률 구조를 학습하는 모델입니다. GPT 스타일 언어 모델은 앞의 토큰들을 보고 다음 토큰을 예측합니다.

- 실습 연결: Chapter 5의 next-token prediction
- 수식 직관: `P(next token | previous tokens)`
- 주의점: 언어 모델이 다음 토큰을 잘 맞힌다는 것과 사실을 항상 정확히 안다는 것은 다릅니다.

### LLM, Large Language Model, 대규모 언어 모델

LLM은 많은 데이터와 큰 parameter 수로 학습된 대규모 언어 모델입니다. 텍스트 생성, 요약, 번역, 코드 작성, 질의응답 등 다양한 작업을 하나의 모델로 수행할 수 있습니다.

- 실습 연결: 이 repo는 LLM의 핵심 부품을 작은 규모로 직접 구현하며 이해합니다.
- 주의점: LLM의 능력은 모델 구조, 데이터, 학습 규모, alignment, tool 사용 방식이 함께 만든 결과입니다.

### Token, 토큰

토큰은 tokenizer가 문장을 나눈 기본 단위입니다. 단어일 수도 있고, 단어 조각일 수도 있고, 공백을 포함한 문자열 조각일 수도 있습니다.

- 예: `"unbelievable"`이 `"un"`, `"believ"`, `"able"` 같은 subword로 나뉠 수 있습니다.
- 실습 연결: Chapter 2의 tokenizer 실습
- 주의점: 토큰은 사람이 생각하는 단어와 항상 같지 않습니다.

### Tokenizer, 토크나이저

Tokenizer는 문자열을 token 목록으로 나누고 각 token을 integer ID로 바꾸는 도구입니다. 모델은 문자열이 아니라 token id를 입력으로 받습니다.

- 실습 연결: `tokenizer.encode(text)`, `tokenizer.decode(ids)`
- 주의점: 같은 문장도 tokenizer가 다르면 token id와 길이가 달라집니다.

### BPE, Byte Pair Encoding

BPE는 자주 함께 등장하는 문자 또는 subword 조각을 병합해 vocabulary를 만드는 tokenization 방법입니다.

- 장점: 모르는 단어도 subword 조각으로 표현할 수 있습니다.
- 실습 연결: GPT 계열 tokenizer를 이해할 때 중요합니다.
- 주의점: BPE token은 단어 경계와 다르게 나뉠 수 있습니다.

### Special Token, 특수 토큰

특수 토큰은 일반 텍스트가 아니라 모델 입력 구조를 표시하는 token입니다. 예: `<pad>`, `<unk>`, `<eos>`, `<bos>`, `<|endoftext|>`

- 실습 연결: padding, instruction format, text boundary 처리
- 주의점: 특수 토큰은 loss 계산에서 제외해야 하는 경우가 많습니다.

### Context Window, 문맥 창

Context window는 모델이 한 번에 입력으로 받을 수 있는 최대 token 길이입니다.

- 예: context length 1024면 한 번에 1024 tokens까지 볼 수 있습니다.
- 실습 연결: Chapter 4의 generation loop에서 `idx[:, -context_size:]`
- 주의점: 문맥 창을 넘는 정보는 잘리거나 별도 검색/RAG 구조가 필요합니다.

### Logits, 로짓

Logits는 softmax를 거쳐 확률이 되기 전의 원시 점수입니다. LLM 출력의 마지막 차원은 보통 vocabulary 전체에 대한 logits입니다.

- 예: `logits: [batch, seq_len, vocab_size]`
- 실습 연결: Chapter 4의 `out_head`, Chapter 5의 cross entropy
- 주의점: logits는 확률이 아닙니다. 하지만 점수가 높을수록 해당 token이 선택될 가능성이 커집니다.

### Softmax

Softmax는 여러 점수를 합이 1인 확률처럼 바꾸는 함수입니다. attention weight 계산과 token 후보 확률 계산에 사용됩니다.

- 실습 연결: Chapter 3의 attention weight, Chapter 4의 generation
- 주의점: softmax는 큰 점수를 더 크게 강조하는 경향이 있습니다.

### Perplexity, 퍼플렉시티

Perplexity는 언어 모델이 다음 토큰을 얼마나 헷갈려 하는지 나타내는 지표입니다. 낮을수록 모델이 정답 토큰을 더 잘 예측한다는 뜻입니다.

- 직관: 가능한 후보 중 평균적으로 몇 개 사이에서 헷갈리는가?
- 주의점: perplexity가 낮아도 사용자가 원하는 답변 품질이 항상 좋다는 뜻은 아닙니다.

---

## 6. Transformer와 Attention

### Transformer

Transformer는 attention을 중심으로 구성된 딥러닝 구조입니다. RNN처럼 순서대로 하나씩 처리하는 대신, sequence 안의 token 관계를 병렬적으로 계산할 수 있습니다.

- 실습 연결: Chapter 3 attention, Chapter 4 GPTModel
- 핵심 부품: attention, FFN, residual connection, LayerNorm, positional encoding
- 주의점: Transformer는 하나의 모델 이름이 아니라 구조 패턴입니다.

### Attention

Attention은 현재 token이 다른 token들을 얼마나 참고할지 계산하는 메커니즘입니다. 중요한 token의 정보를 더 많이 섞어 현재 token 표현을 갱신합니다.

- 실습 연결: Chapter 3에서 Q/K/V, score, mask, softmax, context vector 순서로 구현합니다.
- 직관: "지금 이 단어를 이해하려면 앞의 어떤 단어를 봐야 하는가?"
- 주의점: attention weight가 항상 사람이 해석하는 중요도와 완전히 일치하지는 않습니다.

### Self-Attention

Self-attention은 같은 sequence 안의 token들이 서로를 참고하는 attention입니다. 문장 내부 단어 관계를 모델링하는 데 사용됩니다.

- 예: "그것"이 앞의 어떤 명사를 가리키는지 파악
- 실습 연결: Chapter 3의 attention 구현은 self-attention입니다.
- 주의점: encoder self-attention은 양방향을 볼 수 있지만 GPT의 causal self-attention은 미래를 볼 수 없습니다.

### Causal Attention, Causal Mask

Causal attention은 현재 위치가 미래 token을 보지 못하게 막는 attention입니다. GPT처럼 왼쪽에서 오른쪽으로 다음 token을 예측하는 모델에 필요합니다.

- 실습 연결: `masked_fill_(mask, -torch.inf)`
- 직관: 시험 문제를 풀 때 정답지를 미리 보면 안 되는 것과 같습니다.
- 주의점: causal mask는 padding mask와 다릅니다. causal mask는 미래 차단, padding mask는 가짜 token 무시입니다.

### Query, Key, Value, Q/K/V

Attention에서 입력 token vector는 Query, Key, Value 세 역할의 벡터로 변환됩니다.

| 구성 | 의미 |
|---|---|
| Query | 현재 token이 찾고 싶은 정보 |
| Key | 각 token이 가진 검색용 이름표 |
| Value | 실제로 가져와 섞을 내용 |

- 실습 연결: `W_query`, `W_key`, `W_value`
- 주의점: Q/K/V는 사람이 직접 정한 의미가 아니라 학습되는 projection 결과입니다.

### Attention Score

Attention score는 Query와 Key를 비교해 token 간 관련도를 나타내는 원시 점수입니다.

- 수식: `scores = queries @ keys.transpose(...)`
- 실습 연결: Chapter 3의 attention matrix
- 주의점: score는 softmax 전 값이고, 최종 참고 비율은 attention weight입니다.

### Attention Weight

Attention weight는 attention score에 mask와 scaling을 적용한 뒤 softmax로 정규화한 값입니다. 각 row의 합은 보통 1입니다.

- 실습 연결: `attn_weights = torch.softmax(...)`
- 직관: 어떤 token의 value를 얼마나 섞을지 정하는 비율입니다.
- 주의점: mask된 위치의 weight는 0이 됩니다.

### Context Vector

Context vector는 attention weight로 value vector들을 가중합한 결과입니다. 즉, 문맥 정보가 반영된 현재 token의 새 표현입니다.

- 수식: `context = attn_weights @ values`
- 실습 연결: Chapter 3의 최종 attention 출력
- 주의점: context vector의 shape는 보통 입력 embedding 차원과 맞춰집니다.

### Multi-Head Attention

Multi-head attention은 attention을 여러 head로 나누어 병렬로 수행한 뒤 다시 합치는 방식입니다. 각 head가 서로 다른 관계를 볼 수 있게 합니다.

- 실습 연결: Chapter 3의 `MultiHeadAttention`
- 직관: 한 문장을 여러 관점에서 동시에 읽는 것과 비슷합니다.
- 주의점: head 수가 늘면 각 head의 차원은 보통 `emb_dim / num_heads`가 됩니다.

### Positional Encoding, Position Embedding

Transformer는 token을 병렬로 처리하기 때문에 순서 정보를 별도로 넣어야 합니다. position embedding은 각 위치의 정보를 vector로 만들어 token embedding에 더합니다.

- 실습 연결: Chapter 4의 `pos_emb`
- 주의점: token embedding만 있으면 같은 token이 문장 어디에 있는지 알기 어렵습니다.

### TransformerBlock

TransformerBlock은 attention, FFN, residual connection, LayerNorm 등을 묶은 반복 단위입니다.

- 실습 연결: Chapter 4의 `TransformerBlock`
- GPT 흐름: token embedding -> TransformerBlock x N -> final norm -> output head
- 주의점: block 하나 안에서도 tensor shape가 대부분 `[batch, seq_len, emb_dim]`로 유지됩니다.

### Encoder, Decoder, Decoder-only

Transformer 원 구조는 encoder와 decoder로 구성됩니다. GPT는 decoder-only 구조로, causal mask를 사용해 다음 token을 생성합니다.

- Encoder: 입력 전체를 양방향으로 이해하는 쪽
- Decoder: 이전 출력들을 조건으로 다음 token을 생성하는 쪽
- Decoder-only: GPT처럼 생성 중심으로 decoder stack만 사용하는 구조

---

## 7. 텍스트 생성과 프롬프트

### Prompt, 프롬프트

Prompt는 모델에게 주는 입력 지시문입니다. 질문, 배경 정보, 형식 요구, 예시 등이 포함될 수 있습니다.

- 예: "다음 문장을 세 줄로 요약해줘"
- 실습 연결: Chapter 7의 instruction format
- 주의점: prompt는 모델의 출력을 강하게 좌우하지만, 모델 지식을 완전히 바꾸지는 않습니다.

### Completion, 완성 응답

Completion은 prompt 뒤에 모델이 생성한 출력입니다. GPT 스타일 모델은 prompt 다음에 올 token을 반복적으로 생성합니다.

- 실습 연결: Chapter 4의 generation loop
- 주의점: completion은 한 번에 만들어지는 문장이 아니라 token 단위 반복 결과입니다.

### Decoding, 디코딩

Decoding은 모델이 낸 token 후보 점수에서 실제 다음 token을 선택하는 과정입니다.

- 방법: greedy, sampling, top-k, top-p, beam search
- 실습 연결: Chapter 4의 `argmax` 기반 간단 생성
- 주의점: 같은 모델이라도 decoding 설정에 따라 결과가 크게 달라집니다.

### Greedy Decoding

Greedy decoding은 매 순간 가장 점수가 높은 token 하나를 선택하는 방식입니다.

- 장점: 단순하고 재현성이 높습니다.
- 단점: 다양성이 낮고, 초반 선택 실수를 되돌릴 수 없습니다.
- 실습 연결: `torch.argmax(logits, dim=-1)`

### Sampling

Sampling은 token 확률 분포에서 무작위로 다음 token을 뽑는 방식입니다. 더 다양한 답변을 만들 수 있습니다.

- 실습 연결: temperature, top-k, top-p와 함께 사용됩니다.
- 주의점: 무작위성이 있으므로 같은 prompt에도 결과가 달라질 수 있습니다.

### Temperature

Temperature는 token 후보 분포를 얼마나 날카롭거나 부드럽게 만들지 조절하는 값입니다.

- 낮은 temperature: 높은 점수 token에 집중, 안정적이고 보수적
- 높은 temperature: 낮은 점수 token도 선택 가능, 다양하지만 엉뚱할 수 있음
- 주의점: temperature가 높다고 항상 창의적인 좋은 답변이 되는 것은 아닙니다.

### Top-k Sampling

Top-k는 점수가 높은 k개 token만 후보로 남기고 그 안에서 sampling하는 방식입니다.

- k가 작음: 안정적이지만 반복적일 수 있음
- k가 큼: 다양하지만 품질이 흔들릴 수 있음
- 실습 연결: HTML generation loop의 top-k 슬라이더

### Top-p, Nucleus Sampling

Top-p는 누적 확률이 p가 될 때까지의 token 후보만 남기는 방식입니다. 후보 개수가 상황에 따라 달라집니다.

- 예: top-p 0.9면 누적 확률 90%를 구성하는 후보만 사용
- 주의점: top-k는 후보 개수를 고정하고, top-p는 누적 확률 기준으로 후보를 정합니다.

### Beam Search

Beam search는 한 번에 하나의 후보만 고르지 않고, 여러 가능한 sequence를 유지하며 가장 좋은 문장을 찾는 decoding 방식입니다.

- 장점: 번역처럼 정답성이 중요한 작업에 유리할 수 있습니다.
- 단점: 창의적 생성에서는 뻔한 결과가 나올 수 있습니다.

### Hallucination, 환각

Hallucination은 모델이 사실이 아닌 내용을 그럴듯하게 생성하는 현상입니다.

- 원인: 모델은 "사실 DB 조회"가 아니라 다음 token 확률을 기반으로 생성합니다.
- 대응: RAG, 출처 확인, tool 사용, 후처리 검증
- 주의점: 자신감 있는 문체와 정확성은 별개입니다.

### Grounding, 근거 연결

Grounding은 모델 답변을 특정 문서, 데이터베이스, 도구 결과 같은 외부 근거에 연결하는 것입니다.

- 실무 연결: RAG에서 검색된 문서를 근거로 답변하게 합니다.
- 주의점: 근거 문서 자체가 틀렸거나 검색이 잘못되면 답변도 틀릴 수 있습니다.

### RAG, Retrieval-Augmented Generation

RAG는 질문과 관련된 문서를 검색한 뒤, 그 문서를 prompt에 넣어 답변을 생성하는 방식입니다.

- 기본 흐름: query -> retrieve documents -> prompt 구성 -> generate answer
- 장점: 최신/사내/긴 문서 지식을 모델 parameter에 다시 학습하지 않고 활용할 수 있습니다.
- 주의점: 검색 품질, chunking, reranking, prompt 구성에 따라 결과가 크게 달라집니다.

---

## 8. Fine-tuning과 Alignment

### Pretraining, 사전학습

사전학습은 대량의 일반 데이터로 모델의 기본 언어 능력을 만드는 단계입니다. LLM에서는 보통 다음 token 예측을 대규모로 반복합니다.

- 실습 연결: Chapter 5
- 주의점: 사전학습 모델은 언어를 잘 이어 쓸 수 있지만, 사용자 지시를 항상 잘 따르지는 않습니다.

### Fine-tuning, 미세튜닝

미세튜닝은 이미 학습된 모델을 특정 task나 데이터에 맞게 추가 학습하는 과정입니다.

- 예: 스팸 분류, 고객 상담 답변, 법률 문서 요약
- 실습 연결: Chapter 6의 classification fine-tuning
- 주의점: 작은 데이터로 과하게 fine-tuning하면 모델의 일반 능력이 떨어질 수 있습니다.

### Instruction Tuning

Instruction tuning은 모델이 사용자의 지시를 잘 따르도록 instruction/input/output 형식의 데이터로 학습하는 과정입니다.

- 실습 연결: Chapter 7 SFT
- 주의점: 단순히 지식을 추가하는 것보다 "답변 형식과 지시 따르기 행동"을 맞추는 성격이 강합니다.

### SFT, Supervised Fine-Tuning

SFT는 좋은 예시 답변을 정답으로 두고 지도학습 방식으로 fine-tuning하는 방법입니다.

- 실습 연결: Chapter 7에서 prompt 부분은 loss에서 제외하고 response token만 학습합니다.
- 주의점: SFT 데이터 품질이 낮으면 모델이 나쁜 답변 형식을 그대로 배울 수 있습니다.

### Label Masking

Label masking은 loss를 계산하지 않을 위치의 label을 특수 값으로 바꾸는 방식입니다. PyTorch cross entropy에서는 보통 `-100`이 ignore index로 사용됩니다.

- 실습 연결: Chapter 7의 instruction tuning
- 직관: 질문 부분은 조건으로만 쓰고, 답변 부분만 맞히게 합니다.
- 주의점: prompt token까지 loss에 넣으면 모델이 질문을 그대로 복사하는 방향으로 학습될 수 있습니다.

### LoRA, Low-Rank Adaptation

LoRA는 큰 weight matrix를 직접 모두 바꾸지 않고, 작은 low-rank 행렬 두 개를 추가해 효율적으로 fine-tuning하는 방법입니다.

- 실습 연결: Chapter 6 LoRA
- 직관: 기존 큰 모델은 고정하고, 작은 보정 경로만 학습합니다.
- 장점: 학습 parameter 수와 GPU 메모리를 크게 줄일 수 있습니다.

### Adapter

Adapter는 기존 모델 사이에 작은 학습 모듈을 끼워 넣어 task에 맞게 조정하는 방식입니다.

- LoRA 연결: LoRA도 넓은 의미에서 parameter-efficient adaptation 방식입니다.
- 주의점: adapter를 어디에 넣는지에 따라 성능과 비용이 달라집니다.

### Freezing, 동결

Freezing은 특정 parameter를 학습 중 업데이트하지 않도록 고정하는 것입니다.

- 실습 연결: LoRA에서는 base model weight를 freeze하고 LoRA parameter만 학습합니다.
- 코드 관점: `param.requires_grad = False`
- 주의점: freeze된 parameter는 gradient가 계산되지 않거나 업데이트되지 않습니다.

### PEFT, Parameter-Efficient Fine-Tuning

PEFT는 전체 parameter를 모두 학습하지 않고 일부 작은 parameter만 학습해 fine-tuning 비용을 줄이는 방법들의 총칭입니다.

- 예: LoRA, adapter, prefix tuning, prompt tuning
- 주의점: 비용은 줄지만 task에 따라 full fine-tuning보다 성능이 낮을 수도 있습니다.

### RLHF, Reinforcement Learning from Human Feedback

RLHF는 사람의 선호 피드백을 사용해 모델을 조정하는 방법입니다. 일반적으로 reward model을 학습하고, 강화학습으로 policy model을 업데이트합니다.

- 목적: 더 유용하고 안전하며 사람이 선호하는 답변을 만들기
- 주의점: 구현과 운영이 복잡하며 reward hacking 같은 문제가 생길 수 있습니다.

### DPO, Direct Preference Optimization

DPO는 chosen/rejected 응답 쌍을 사용해 선호 답변의 상대 확률을 높이는 alignment 방법입니다. RLHF보다 단순한 학습 흐름으로 선호 학습을 수행할 수 있습니다.

- 실습 연결: Chapter 7 DPO
- 주요 구성: policy model, reference model, chosen response, rejected response
- 주의점: 좋은 preference dataset이 중요합니다.

### Policy Model

Policy model은 현재 학습 또는 업데이트 대상인 모델입니다. DPO에서는 선호 답변을 더 내도록 policy model을 조정합니다.

- 실습 연결: Chapter 7 DPO
- 주의점: reference model과 혼동하지 않아야 합니다.

### Reference Model

Reference model은 DPO에서 비교 기준으로 사용하는 고정 모델입니다. 보통 policy model의 초기 상태를 복사해 freeze합니다.

- 역할: policy가 기존 모델에서 너무 멀리 벗어나지 않도록 기준을 제공합니다.
- 주의점: reference model은 일반적으로 업데이트하지 않습니다.

### Chosen / Rejected

Chosen은 선호되는 답변, rejected는 선호되지 않는 답변입니다. DPO는 같은 prompt에 대해 chosen이 rejected보다 더 높은 점수를 받도록 학습합니다.

- 실습 연결: `instruction-data-with-preference.json`
- 주의점: chosen/rejected 품질이 낮으면 모델이 잘못된 선호를 배울 수 있습니다.

---

## 9. 평가와 운영

### Metric, 평가 지표

Metric은 모델 성능을 숫자로 측정하는 기준입니다. task에 따라 accuracy, F1, loss, perplexity, BLEU, ROUGE, win rate 등이 사용됩니다.

- 주의점: 하나의 metric만으로 모델 품질을 판단하기 어렵습니다.
- LLM 연결: 자동 지표와 사람 평가를 함께 보는 경우가 많습니다.

### Accuracy, 정확도

Accuracy는 전체 샘플 중 맞힌 비율입니다.

- 예: 100개 중 90개 정답이면 accuracy 90%
- 실습 연결: Chapter 6 분류 fine-tuning
- 주의점: class imbalance가 심하면 accuracy만 보면 misleading할 수 있습니다.

### Precision, Recall, F1

Precision은 모델이 양성이라고 예측한 것 중 실제 양성 비율입니다. Recall은 실제 양성 중 모델이 찾아낸 비율입니다. F1은 precision과 recall의 조화 평균입니다.

- 예: 스팸 탐지에서 precision은 "스팸이라고 한 메일이 진짜 스팸인가", recall은 "진짜 스팸을 얼마나 잡았는가"입니다.
- 주의점: 어떤 지표가 중요한지는 task 비용 구조에 따라 다릅니다.

### Benchmark, 벤치마크

Benchmark는 모델을 비교하기 위한 표준 데이터셋이나 평가 절차입니다.

- 예: MMLU, GSM8K, HumanEval 등
- 주의점: benchmark 점수가 실제 서비스 품질을 완전히 보장하지 않습니다.

### Latency, 지연시간

Latency는 요청을 보낸 뒤 응답이 나오기까지 걸리는 시간입니다.

- LLM 연결: 첫 token까지 시간과 전체 응답 완료 시간이 모두 중요합니다.
- 개선 방법: 작은 모델, quantization, batching, caching, 더 빠른 serving engine

### Throughput, 처리량

Throughput은 단위 시간당 처리할 수 있는 요청 수나 token 수입니다.

- 예: tokens/sec, requests/sec
- 주의점: latency와 throughput은 trade-off가 있을 수 있습니다.

### Quantization, 양자화

Quantization은 모델 weight나 activation을 더 낮은 bit 정밀도로 표현해 메모리와 연산 비용을 줄이는 방법입니다.

- 예: FP16, INT8, 4-bit
- 장점: 더 작은 GPU 메모리로 모델 실행 가능
- 주의점: 너무 강한 quantization은 품질 저하를 만들 수 있습니다.

### Distillation, 지식 증류

Distillation은 큰 teacher model의 출력을 작은 student model이 따라 배우게 하는 방법입니다.

- 목적: 작은 모델로 큰 모델에 가까운 성능을 얻기
- 주의점: teacher의 오류나 편향도 student가 배울 수 있습니다.

### Pruning, 가지치기

Pruning은 중요도가 낮은 weight, neuron, head 등을 제거해 모델을 작게 만드는 방법입니다.

- 목적: 메모리와 계산량 감소
- 주의점: 잘못 제거하면 성능이 크게 떨어질 수 있습니다.

### MoE, Mixture of Experts

MoE는 여러 expert network 중 입력마다 일부 expert만 선택해 사용하는 구조입니다. 전체 parameter 수는 크지만, 한 token 처리에 쓰는 active parameter를 줄일 수 있습니다.

- 직관: 모든 문제를 한 모델 부분이 처리하지 않고, 전문 모듈 중 일부를 골라 사용합니다.
- 주의점: MoE는 보통 attention이 아니라 FFN/MLP 영역에 적용되는 경우가 많습니다.

### Vector Database, 벡터 데이터베이스

Vector DB는 문서나 이미지 embedding을 저장하고, query embedding과 가까운 항목을 빠르게 검색하는 데이터베이스입니다.

- RAG 연결: 질문과 의미적으로 가까운 문서 chunk를 찾는 데 사용합니다.
- 주의점: embedding 품질과 chunking 전략이 검색 품질을 좌우합니다.

### Semantic Search, 의미 기반 검색

Semantic search는 정확한 키워드 일치가 아니라 의미적으로 가까운 문서를 찾는 검색 방식입니다.

- 예: "퇴사 절차"를 검색했을 때 "사직서 제출 방법" 문서를 찾기
- RAG 연결: embedding similarity를 사용합니다.

### Chunking, 청킹

Chunking은 긴 문서를 검색이나 모델 입력에 넣기 좋게 작은 조각으로 나누는 과정입니다.

- RAG 연결: chunk 단위로 embedding을 만들고 검색합니다.
- 주의점: 너무 작으면 문맥이 부족하고, 너무 크면 검색 정확도와 context 효율이 떨어질 수 있습니다.

### Reranking, 재순위화

Reranking은 1차 검색 결과를 더 정교한 모델로 다시 정렬하는 과정입니다.

- RAG 연결: embedding 검색으로 후보를 넓게 가져온 뒤 reranker가 더 관련성 높은 문서를 위로 올립니다.
- 주의점: 품질은 좋아질 수 있지만 latency가 늘어납니다.

### Agent

AI agent는 목표를 달성하기 위해 모델이 계획을 세우고, 도구를 호출하고, 중간 결과를 바탕으로 다음 행동을 선택하는 시스템입니다.

- 예: 코드 수정 agent, 브라우저 자동화 agent, 데이터 분석 agent
- 주의점: agent는 LLM 하나가 아니라 LLM, tool, memory, planner, executor가 결합된 시스템입니다.

### Tool Calling, Function Calling

Tool calling은 LLM이 외부 함수나 API를 호출해 계산, 검색, 파일 처리, 데이터베이스 조회 등을 수행하게 하는 방식입니다.

- 예: 날씨 API 호출, SQL 실행, 코드 테스트 실행
- 주의점: 모델이 도구를 호출할 수 있어도 도구 결과 검증과 권한 제어가 필요합니다.

---

## 10. 안전성, 윤리, 보안

### Bias, 편향

Bias는 데이터나 모델이 특정 집단, 관점, 패턴에 치우쳐 있는 현상입니다.

- 원인: 학습 데이터 편향, 라벨링 편향, 평가 기준 편향
- 주의점: 편향은 단순히 모델 문제가 아니라 데이터와 사회적 맥락의 문제이기도 합니다.

### Toxicity, 유해성

Toxicity는 혐오, 폭력, 괴롭힘, 차별적 표현처럼 사용자나 사회에 해를 줄 수 있는 출력 성향입니다.

- 대응: 안전성 데이터, moderation, guardrail, red teaming
- 주의점: 문화와 맥락에 따라 유해성 판단이 달라질 수 있습니다.

### Jailbreak

Jailbreak는 모델의 안전 정책이나 지시를 우회해 금지된 출력을 유도하려는 공격적 prompt입니다.

- 예: "이전 지시를 무시하고..."
- 주의점: 시스템 prompt만으로 모든 jailbreak를 막기는 어렵고, 다층 방어가 필요합니다.

### Prompt Injection

Prompt injection은 외부 문서나 사용자 입력 안에 악의적 지시를 숨겨, 모델이 원래 시스템 지시를 무시하게 만드는 공격입니다.

- RAG 연결: 검색된 문서 안에 "이전 지시를 무시하라" 같은 문구가 있을 수 있습니다.
- 대응: 입력 출처 분리, tool 권한 제한, instruction hierarchy, output 검증

### PII, Personally Identifiable Information

PII는 개인을 식별할 수 있는 정보입니다. 이름, 전화번호, 주민등록번호, 이메일, 주소, 계정 ID 등이 포함될 수 있습니다.

- 주의점: AI 시스템에 PII를 넣거나 저장할 때는 보안, 접근 제어, 마스킹, 삭제 정책이 필요합니다.

### Guardrail

Guardrail은 모델이 위험하거나 부적절한 행동을 하지 않도록 막는 안전 장치입니다.

- 예: 입력 필터, 출력 필터, tool 권한 제한, 민감 정보 탐지
- 주의점: guardrail은 모델 자체의 성능과 별개로 시스템 레벨에서 설계해야 합니다.

### Red Teaming

Red teaming은 의도적으로 모델과 시스템을 공격적으로 테스트해 취약점을 찾는 과정입니다.

- 목적: jailbreak, prompt injection, 유해 출력, 개인정보 노출 같은 위험을 사전에 발견
- 주의점: 출시 전뿐 아니라 운영 중에도 반복해야 합니다.

### Explainability, 설명가능성

설명가능성은 모델이 왜 그런 예측이나 답변을 했는지 사람이 이해할 수 있게 만드는 성질입니다.

- 예: feature importance, attention visualization, saliency map
- 주의점: LLM의 설명은 그럴듯한 사후 설명일 수 있으며 실제 내부 원인을 완전히 보장하지 않습니다.

### Accountability, 책임성

책임성은 AI 시스템의 결과에 대해 누가 어떤 책임을 지는지 명확히 하는 원칙입니다.

- 실무 연결: 로그, 승인 절차, 사람 검토, 감사 가능성
- 주의점: "AI가 그랬다"는 책임 회피가 될 수 없습니다.

---

## 11. 헷갈리기 쉬운 용어 비교

| 용어 A | 용어 B | 차이 |
|---|---|---|
| AI | 머신러닝 | AI는 큰 분야, 머신러닝은 데이터로 학습하는 AI 방법 |
| 머신러닝 | 딥러닝 | 딥러닝은 신경망을 깊게 쌓는 머신러닝 방법 |
| Parameter | Hyperparameter | parameter는 학습으로 바뀌는 값, hyperparameter는 사람이 설정하는 값 |
| Token | Word | token은 tokenizer 단위, word는 사람이 보는 단어 단위 |
| Token ID | Embedding | token ID는 정수 번호, embedding은 그 번호를 바꾼 dense vector |
| Logits | Probability | logits는 softmax 전 점수, probability는 softmax 후 확률 |
| Loss | Metric | loss는 학습 최적화 대상, metric은 성능 관찰 지표 |
| Training | Inference | training은 parameter update가 있고, inference는 보통 update가 없음 |
| Validation | Test | validation은 개발 중 조정에 사용, test는 최종 평가에 사용 |
| Attention Score | Attention Weight | score는 softmax 전 관련도 점수, weight는 정규화된 참고 비율 |
| Causal Mask | Padding Mask | causal mask는 미래 token 차단, padding mask는 가짜 token 무시 |
| Fine-tuning | Prompting | fine-tuning은 parameter를 바꾸고, prompting은 입력 지시만 바꿈 |
| SFT | DPO | SFT는 좋은 답변을 정답으로 학습, DPO는 선호/비선호 답변 쌍을 비교 학습 |
| Policy Model | Reference Model | policy는 업데이트 대상, reference는 비교용 고정 모델 |
| RAG | Fine-tuning | RAG는 외부 문서를 검색해 prompt에 넣고, fine-tuning은 모델 parameter를 조정 |
| Temperature | Top-k | temperature는 분포의 날카로움을 조절, top-k는 남길 후보 개수를 제한 |

## 이 repo에서 특히 먼저 익힐 용어

1. `token`, `token id`, `tokenizer`, `vocab_size`
2. `tensor`, `shape`, `batch`, `seq_len`, `emb_dim`
3. `embedding`, `hidden state`, `logits`, `softmax`
4. `Q/K/V`, `attention score`, `attention weight`, `causal mask`
5. `loss`, `cross entropy`, `backward`, `optimizer.step`
6. `pretraining`, `fine-tuning`, `SFT`, `LoRA`, `DPO`

## 관련 문서

- [Chapter 1: Vector Space](Chapter_1_Exercise_Vector_Space.md)
- [Chapter 2: Dataset and Tokenization](Chapter_2_Exercise_Dataset.md)
- [Chapter 3: Attention](Chapter_3_Excercise_Attention.md)
- [Chapter 4: GPT](Chapter_4_Excercise_GPT.md)
- [Chapter 5: Pretraining](Chapter_5_Excercise_Pretraining.md)
- [Chapter 6: Fine-tuning Classification](Chapter_6_Excercise_Finetuning_Classification.md)
- [Chapter 6: LoRA](Chapter_6_Excercise_Finetuning_Classification_LoRA.md)
- [Chapter 7: Instruction Tuning](Chapter_7_Exercise_Follow_Instructions.md)
- [Chapter 7: DPO](Chapter_7_Exercise_Follow_Instructions_dpo.md)
