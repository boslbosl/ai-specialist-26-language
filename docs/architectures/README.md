# LLM Hands-On 아키텍처 다이어그램 (Architecture Documentation)

이 디렉토리에는 LLM Hands-On 코스의 각 챕터별 아키텍처 다이어그램과 상세 설명이 포함되어 있습니다.

## 📁 파일 구조

```
architectures/
├── Chapter_1_Vector_Space.md              # 단어 벡터와 의미 공간
├── Chapter_2_Dataset.md                   # 토큰화와 GPT 데이터셋
├── Chapter_3_Attention.md                 # Self-Attention과 Multi-Head Attention
├── Chapter_4_GPT_Model.md                 # GPT 모델 구조
├── Chapter_5_Pretraining.md               # GPT 사전학습 루프
├── Chapter_6_Finetuning_Classification.md # 텍스트 분류 미세조정
└── Chapter_7_Instruction_Following.md     # 지시 따르기 미세조정
```

## 📚 각 챕터 설명

### Chapter 1️⃣: 단어 벡터와 의미 공간
- **주제**: Word2Vec, 코사인 유사도, 벡터 산술
- **아키텍처**: 임베딩 → 정규화 → 벡터 연산
- **학습 목표**: 단어 벡터의 의미와 벡터 연산 이해

### Chapter 2️⃣: 토큰화와 GPT 데이터셋
- **주제**: BPE 토큰화, 슬라이딩 윈도우, Dataset/DataLoader
- **아키텍처**: 텍스트 → 토큰화 → 입력-타겟 쌍 → 배치
- **학습 목표**: 데이터 준비의 전체 파이프라인

### Chapter 3️⃣: Self-Attention과 Multi-Head Attention
- **주제**: Query/Key/Value, Scaled Dot-Product, 인과 마스크
- **아키텍처**: Q,K,V 생성 → 점수 계산 → Softmax → 컨텍스트
- **학습 목표**: Attention 메커니즘의 동작 원리

### Chapter 4️⃣: GPT 모델 구조
- **주제**: 토큰/위치 임베딩, Transformer 블록, 로짓 생성, 텍스트 생성
- **아키텍처**: 입력 → 임베딩 → N개 블록 → 선형 투영 → 생성
- **학습 목표**: 전체 GPT 모델의 구조와 생성 루프

### Chapter 5️⃣: GPT 사전학습 루프
- **주제**: Forward Pass, Loss 계산, 역전파, 최적화
- **아키텍처**: 배치 로드 → Forward → Loss → Backward → Step
- **학습 목표**: 학습 루프의 각 단계와 최적화 기법

### Chapter 6️⃣: 텍스트 분류 미세조정
- **주제**: 분류 헤드 추가, 패딩, 어텐션 마스크, Accuracy 평가
- **아키텍처**: 전처리 → 모델 → 분류 헤드 → Loss → 평가
- **학습 목표**: 사전학습 모델을 분류 작업으로 재사용

### Chapter 7️⃣: 지시 따르기 미세조정
- **주제**: 포맷팅, 라벨 마스킹, 응답 부분만 학습
- **아키텍처**: 포맷팅 → 토큰화 → 라벨 마스킹 → Loss (응답만)
- **학습 목표**: 모델이 자연어 지시를 따르도록 훈련

## 🔍 다이어그램 형식

모든 파일은 다음과 같은 구조를 가집니다:

1. **📚 개요** - 장의 주제와 목표
2. **🏗️ 아키텍처 다이어그램** - Mermaid.js로 작성된 시각적 흐름도
3. **📖 상세 설명** - 각 단계별 상세 설명과 코드 예시
4. **🎯 핵심 포인트** - 중요한 개념 요약 표
5. **💡 팁/다음 장과의 연결** - 실무 팁과 학습 경로

## 🎨 Mermaid 다이어그램 보기

이 파일들의 다이어그램은 다음 플랫폼에서 렌더링됩니다:
- GitHub (마크다운 파일)
- GitBook
- Notion
- VS Code (Markdown Preview Enhanced 확장)

## 🚀 활용 방법

### 1. 학습 자료로 사용
- 각 장을 공부하기 전에 해당 아키텍처 파일 읽기
- 다이어그램으로 전체 흐름 파악
- 코드 예시로 구체적 구현 학습

### 2. 복습 자료로 사용
- 특정 개념을 빠르게 상기하기
- 과정 중간에 흐름도 다시 보기
- 핵심 포인트 표로 개념 정리

### 3. 참고 자료로 사용
- 프로젝트 진행 중 아키텍처 다시 확인
- 다른 모델 구현 시 비교 참고

## 📝 주요 개념 맵

```
단어 벡터 (Ch.1)
    ↓
토큰화 & 데이터셋 (Ch.2)
    ↓
Attention (Ch.3)
    ↓
GPT 모델 구조 (Ch.4)
    ↓
사전학습 루프 (Ch.5)
    ↓
미세조정 분야별 선택:
├─→ 분류 (Ch.6)
└─→ 지시 따르기 (Ch.7)
```

## 💬 추가 리소스

- **공식 논문**
  - "Attention is All You Need" (Transformer)
  - "Language Models are Unsupervised Multitask Learners" (GPT-2)

- **관련 개념**
  - Self-Attention
  - Transformer Architecture
  - Language Model Pre-training
  - Transfer Learning and Fine-tuning

## ✅ 체크리스트

각 장을 학습할 때 이 체크리스트를 참고하세요:

- [ ] 아키텍처 다이어그램 전체 이해
- [ ] 각 컴포넌트의 입출력 shape 파악
- [ ] 코드 예시 실행 및 이해
- [ ] 핵심 포인트 3가지 이상 설명 가능

## 📞 피드백

이 문서에 대한 피드백이나 개선 사항이 있으면 알려주세요!

---

**마지막 업데이트**: 2026-05-15
