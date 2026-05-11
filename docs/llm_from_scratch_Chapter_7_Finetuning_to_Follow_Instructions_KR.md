# Chapter 7 Finetuning to Follow Instructions KR

- 노트북: [llm_from_scratch/Chapter_7_Finetuning_to_Follow_Instructions_KR.ipynb](../llm_from_scratch/Chapter_7_Finetuning_to_Follow_Instructions_KR.ipynb)
- 학습 주제: 지시 따르기 미세튜닝
- 대상 독자: 파이썬은 익숙하지만 `torch`와 Transformer/GPT 내부 구조는 처음 보는 개발자

## 핵심 흐름

```mermaid
flowchart TD
    A[instruction/input/output JSON] --> B[프롬프트 포맷팅]
    B --> C[토큰화]
    C --> D[응답 영역 label 구성]
    D --> E[GPT fine-tuning]
    E --> F[생성 응답]
    F --> G[정성/정량 평가]
```

## 노트북 설명

이 노트북은 지시 따르기 미세튜닝를 실습 코드로 따라가며, 문자열/토큰/텐서/모델 출력이 어떻게 연결되는지 확인하는 자료입니다. 각 코드 셀에는 셀 목적, torch 관점, 실행 후 확인할 포인트를 주석으로 추가했습니다. 처음 읽을 때는 세부 수식보다 텐서의 차원 변화와 `nn.Module.forward()`의 입출력을 우선 추적하세요.

## 학습 목표

- instruction 데이터가 프롬프트-응답 형식으로 변환되는 흐름을 이해합니다.
- 응답 토큰만 loss에 반영하는 방식의 의미를 확인합니다.
- 튜닝 후 생성 결과를 평가하는 기본 절차를 익힙니다.

## torch 관점에서 볼 부분

- label에서 무시할 위치는 보통 -100으로 두어 loss 계산에서 제외합니다.
- 생성 시에는 학습과 달리 gradient가 필요 없으므로 no_grad/eval 모드를 사용합니다.

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
