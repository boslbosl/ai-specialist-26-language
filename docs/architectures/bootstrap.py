#!/usr/bin/env python3
"""
Architecture Diagram Generator for LLM Hands-On Course
"""

if __name__ == '__main__':
    import os
    
    # Base directory
    base = r'C:\work\llm_hands_on.worktrees\agents-architecture-diagrams-mermaid-js'
    arch = os.path.join(base, 'architectures')
    
    # Create directory
    os.makedirs(arch, exist_ok=True)
    print(f"Created: {arch}")
    
    # Chapter 1
    ch1 = """# Chapter 1: 단어 벡터와 의미 공간 (Word Vectors and Semantic Space)

## 📚 개요
이 장에서는 단어가 어떻게 숫자 벡터로 표현되고, 이 벡터들이 어떻게 의미적 관계를 드러내는지 학습합니다.

---

## 🏗️ 아키텍처 다이어그램

```mermaid
graph TB
    subgraph "입력"
        A["텍스트 데이터"]
    end
    
    subgraph "임베딩 계층"
        B["Word Embedding<br/>(Gensim/Word2Vec)"]
    end
    
    subgraph "벡터 연산"
        C["벡터 정규화<br/>(Normalization)"]
        D["코사인 유사도<br/>(Cosine Similarity)"]
        E["벡터 산술<br/>(Vector Arithmetic)"]
    end
    
    subgraph "출력"
        F["단어 벡터<br/>shape: d"]
        G["유사 단어 탐색"]
        H["의미 관계 분석<br/>king - man + woman = queen"]
    end
    
    A -->|토큰화| B
    B --> C
    B --> D
    B --> E
    C --> F
    D --> G
    E --> H
```

---

## 📖 상세 설명

### 1️⃣ **입력 데이터**
- 대규모 텍스트 말뭉치(Corpus)
- 단어 간 문맥 관계가 풍부한 데이터

### 2️⃣ **임베딩 계층 (Embedding Layer)**
- **Word2Vec (Gensim 사용)**
  - CBOW: 주변 단어로부터 중심 단어 예측
  - Skip-gram: 중심 단어로부터 주변 단어 예측
  - 결과: 각 단어를 d차원의 밀집 벡터로 표현

### 3️⃣ **벡터 연산**
- 벡터 정규화: L2 정규화로 방향만 고려
- 코사인 유사도: 두 벡터 사이의 각도로 유사도 측정
- 벡터 산술: 의미 관계 표현

### 4️⃣ **출력**
- 단어 벡터: d차원의 밀집 수치 벡터
- 유사 단어 탐색: 코사인 유사도로 가장 유사한 단어 찾기
- 의미 관계 분석: 벡터 산술로 유추 문제 풀이

---

## 🎯 핵심 포인트

| 개념 | 설명 |
|------|------|
| **임베딩 차원** | 보통 100~300차원 |
| **정규화의 이유** | 벡터 방향(각도)만 중요 |
| **벡터 산술 작동 원리** | 의미적 방향이 자동으로 인코딩됨 |

---

## 💡 다음 장과의 연결
- **Chapter 2**: 토큰화된 텍스트를 임베딩하는 과정
- **Chapter 3~4**: Transformer 아키텍처에서 토큰 임베딩 활용
"""
    
    with open(os.path.join(arch, 'Chapter_1_Vector_Space.md'), 'w', encoding='utf-8') as f:
        f.write(ch1)
    
    print("✓ Chapter 1 created")
    
    # Chapter 2
    ch2 = """# Chapter 2: 토큰화와 GPT 데이터셋 (Tokenization and Dataset)

## 📚 개요
이 장에서는 텍스트가 어떻게 토큰 ID로 변환되고, 학습을 위해 입력-타겟 쌍으로 준비되는지 학습합니다.

---

## 🏗️ 아키텍처 다이어그램

```mermaid
graph TB
    subgraph "입력"
        A["원본 텍스트"]
    end
    
    subgraph "토큰화"
        B["Tokenizer<br/>(tiktoken/GPT-2)"]
        C["텍스트 → 토큰 ID 리스트"]
    end
    
    subgraph "데이터 준비"
        D["슬라이딩 윈도우<br/>(Sliding Window)"]
        E["입력-타겟 쌍 생성"]
    end
    
    subgraph "배치 구성"
        F["Dataset<br/>.__getitem__"]
        G["DataLoader<br/>(배치 생성)"]
    end
    
    subgraph "출력"
        H["배치 텐서<br/>shape: batch, seq_len"]
        I["입력 텐서<br/>shape: B, L"]
        J["타겟 텐서<br/>shape: B, L"]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    H --> J
```

---

## 📖 상세 설명

### 1️⃣ **입력: 원본 텍스트**
- 대규모 텍스트 데이터
- GPT-2 인코더로 토큰화

### 2️⃣ **토큰화 (Tokenization)**
- tiktoken (GPT-2): Byte Pair Encoding 기반
- encode(): 텍스트 → 토큰 ID
- decode(): 토큰 ID → 텍스트

### 3️⃣ **데이터 준비**
- 슬라이딩 윈도우: 고정 길이 청크 생성
- 입력-타겟 쌍: 다음 토큰 예측 형식

### 4️⃣ **배치 구성**
- Dataset: 개별 샘플 반환
- DataLoader: 배치로 묶음

---

## 🎯 핵심 포인트

| 개념 | 설명 |
|------|------|
| **토큰화** | 유연하고 효율적인 텍스트 분할 |
| **슬라이딩 윈도우** | 데이터 재사용 & 메모리 효율성 |
| **입력-타겟 오프셋** | Next-token prediction의 핵심 |
| **배치 구성** | 병렬 처리로 학습 효율 향상 |
"""
    
    with open(os.path.join(arch, 'Chapter_2_Dataset.md'), 'w', encoding='utf-8') as f:
        f.write(ch2)
    
    print("✓ Chapter 2 created")
