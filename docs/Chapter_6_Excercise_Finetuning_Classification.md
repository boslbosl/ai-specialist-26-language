# Chapter 6 Excercise Finetuning Classification

- 노트북: [Chapter_6_Excercise_Finetuning_Classification.ipynb](../Chapter_6_Excercise_Finetuning_Classification.ipynb)
- 학습 주제: 텍스트 분류 미세튜닝
- 대상 독자: 파이썬은 익숙하지만 `torch`와 Transformer/GPT 내부 구조는 처음 보는 개발자

## Overview Diagram

```mermaid
flowchart LR
    A["분류용 텍스트 데이터"] --> B["Tokenizer + Padding<br/>길이가 다른 문장을 batch로 정렬"]
    B --> C["Pretrained GPT Backbone<br/>문장 표현 계산"]
    C --> D["Classification Head<br/>분류 클래스 점수 계산"]
    D --> E["Class Logits<br/>[batch, num_classes]"]
    E --> F["Loss + Accuracy<br/>정답 라벨과 비교"]
    F --> G["Fine-tuning<br/>분류 문제에 맞게 가중치 업데이트"]
    G --> C

    subgraph Goal["미세튜닝 목표"]
        H["사전학습된 언어 모델을 특정 분류 태스크에 맞게 재사용하기"]
    end

    F -. "평가 지표" .-> H
```

## 핵심 흐름

```mermaid
flowchart TD
    A[분류 데이터 CSV] --> B[토큰화/패딩]
    B --> C[GPT backbone]
    C --> D[마지막 토큰 hidden state]
    D --> E[분류 Linear head]
    E --> F[class logits]
    F --> G[CrossEntropyLoss]
    G --> H[accuracy 평가]
```

## 노트북 설명

이 노트북은 텍스트 분류 미세튜닝를 실습 코드로 따라가며, 문자열/토큰/텐서/모델 출력이 어떻게 연결되는지 확인하는 자료입니다. 각 코드 셀에는 셀 목적, torch 관점, 실행 후 확인할 포인트를 주석으로 추가했습니다. 처음 읽을 때는 세부 수식보다 텐서의 차원 변화와 `nn.Module.forward()`의 입출력을 우선 추적하세요.

## 학습 목표

- 사전학습 GPT를 분류 문제에 맞게 재사용하는 방법을 이해합니다.
- 텍스트 길이 정렬, padding, attention mask의 의미를 확인합니다.
- 분류 head와 accuracy 평가 흐름을 익힙니다.

## torch 관점에서 볼 부분

- 분류 head의 출력 shape은 (batch, num_classes)입니다.
- 마지막 토큰 hidden state를 문장 전체 표현으로 사용하는 패턴을 확인합니다.

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
