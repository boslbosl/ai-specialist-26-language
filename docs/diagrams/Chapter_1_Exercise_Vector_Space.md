# Chapter 1 Vector Space Diagrams

- 관련 문서: [../Chapter_1_Exercise_Vector_Space.md](../Chapter_1_Exercise_Vector_Space.md)
- 관련 노트북: [../../Chapter_1_Exercise_Vector Space.ipynb](../../Chapter_1_Exercise_Vector%20Space.ipynb)

## 전체 흐름

```mermaid
flowchart TD
    A["단어 또는 문장"] --> B["사전학습 단어 벡터<br/>gensim / GloVe"]
    B --> C["벡터 공간 좌표<br/>dense numeric vector"]
    C --> D["유사도 계산<br/>cosine similarity"]
    C --> E["벡터 산술<br/>king - man + woman"]
    C --> F["그룹 내 이상치 탐지<br/>doesnt_match"]
    D --> G["의미적으로 가까운 단어 확인"]
    E --> G
    F --> G
```

## 실습 코드 흐름

```mermaid
flowchart LR
    A["pip install gensim"] --> B["gensim.downloader.load"]
    B --> C["glove-wiki-gigaword-50"]
    C --> D["model.similarity"]
    C --> E["model.most_similar"]
    C --> F["model.doesnt_match"]
    D --> G["단어 간 거리 해석"]
    E --> G
    F --> G
```

## 개념 연결

```mermaid
flowchart TD
    A["one-hot id<br/>의미 거리 없음"] --> B["embedding vector<br/>의미 거리 학습"]
    B --> C["가까운 벡터<br/>비슷한 의미"]
    B --> D["먼 벡터<br/>다른 의미"]
    C --> E["Chap2 token embedding"]
    E --> F["Chap3 attention 입력"]
```

## 체크포인트

```mermaid
flowchart LR
    A["벡터 차원 확인"] --> B["cosine similarity 범위 확인"]
    B --> C["most_similar 결과 해석"]
    C --> D["벡터 산술이 항상 정확한 규칙은 아님을 확인"]
```

