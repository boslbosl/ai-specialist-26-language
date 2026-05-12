# Chapter 5 Excercise Pretraining

- 노트북: [Chapter_5_Excercise_Pretraining.ipynb](../Chapter_5_Excercise_Pretraining.ipynb)
- 학습 주제: 언어 모델 사전학습 루프
- 대상 독자: 파이썬은 익숙하지만 `torch`와 Transformer/GPT 내부 구조는 처음 보는 개발자

## Overview Diagram

```mermaid
flowchart LR
    A["원문 텍스트 코퍼스"] --> B["Tokenizer<br/>텍스트를 token id로 변환"]
    B --> C["Next-token Dataset<br/>input과 target을 한 칸 밀어 구성"]
    C --> D["GPTModel<br/>각 위치의 다음 토큰 logits 계산"]
    D --> E["Loss<br/>logits와 target 비교"]
    E --> F["Backpropagation<br/>gradient 계산"]
    F --> G["Optimizer Step<br/>모델 파라미터 업데이트"]
    G --> H["평가와 샘플 생성<br/>학습 진행 상태 확인"]
    H --> C

    subgraph Goal["사전학습 목표"]
        I["문맥을 보고 다음 토큰을 예측하는 일반 언어 모델 만들기"]
    end

    H -. "반복 후" .-> I
```

## 핵심 흐름

```mermaid
flowchart TD
    A[텍스트 코퍼스] --> B[DataLoader]
    B --> C[미니배치 input/target]
    C --> D[GPT forward]
    D --> E[CrossEntropyLoss]
    E --> F[backward]
    F --> G[optimizer step]
    G --> H[주기적 검증/샘플 생성]
    H --> B
```

## 노트북 설명

이 노트북은 언어 모델 사전학습 루프를 실습 코드로 따라가며, 문자열/토큰/텐서/모델 출력이 어떻게 연결되는지 확인하는 자료입니다. 각 코드 셀에는 셀 목적, torch 관점, 실행 후 확인할 포인트를 주석으로 추가했습니다. 처음 읽을 때는 세부 수식보다 텐서의 차원 변화와 `nn.Module.forward()`의 입출력을 우선 추적하세요.

## 학습 목표

- 사전학습 루프의 forward, loss, backward, optimizer step을 연결합니다.
- train/validation loss를 주기적으로 측정하는 이유를 이해합니다.
- 학습 중 샘플 생성을 통해 모델 변화를 관찰합니다.

## torch 관점에서 볼 부분

- loss.backward()가 파라미터별 gradient를 채우고 optimizer.step()이 값을 갱신합니다.
- model.train()과 model.eval()은 dropout 같은 계층 동작을 바꿉니다.

## 추천 읽기 순서

1. 데이터가 문자열에서 토큰 ID 또는 텐서로 바뀌는 지점을 찾습니다.
2. 모델 계층이 입력 텐서를 어떤 shape으로 받아 어떤 shape으로 반환하는지 확인합니다.
3. loss 또는 평가 지표가 어떤 출력과 정답을 비교하는지 확인합니다.
4. 생성, 분류, 평가처럼 노트북의 최종 목적에 해당하는 셀을 다시 실행하며 전체 흐름을 연결합니다.

## 자주 헷갈리는 포인트

- `logits`는 확률이 아니라 정규화되지 않은 점수입니다.
- `batch` 차원은 여러 샘플을 한 번에 처리하기 위해 앞에 붙는 차원입니다.
- `sequence_length`는 한 샘플 안의 토큰 개수이며, GPT에서는 시간축처럼 다룹니다.
- `vocab_size` 차원은 다음 토큰 후보 전체에 대한 점수입니다.
