# HOWTO Setup

이 문서는 `llm_hands_on` 실습 환경을 준비하고 Jupyter Lab을 실행하는 방법을 정리합니다.

## 1. 사전 준비

- Python 3.12 이상
- Windows PowerShell 또는 명령 프롬프트
- Git

이 저장소에는 Windows용 `uv.exe`, `uvw.exe`, `uvx.exe`가 포함되어 있으므로 별도 설치 없이 `uv`를 사용할 수 있습니다.

## 2. 저장소 위치로 이동

```powershell
cd C:\work\llm_hands_on
```

다른 위치에 저장소를 받았다면 해당 경로로 이동합니다.

## 3. Python 버전 확인

```powershell
python --version
```

프로젝트는 `.python-version` 기준으로 Python 3.12를 사용합니다.

## 4. 의존성 설치

가장 권장되는 방법은 `uv`로 잠금 파일(`uv.lock`)에 맞춰 환경을 동기화하는 것입니다.

```powershell
.\uv.exe sync
```

또는 포함된 배치 파일을 사용할 수 있습니다.

```powershell
.\sync_proj.bat
```

설치되는 주요 패키지는 다음과 같습니다.

- `torch`, `torchvision`
- `transformers`
- `datasets`
- `tiktoken`
- `jupyterlab`
- `matplotlib`, `pandas`
- `openai`, `python-dotenv`

`pyproject.toml`에는 PyTorch CUDA 12.6 인덱스가 설정되어 있습니다.

## 5. Jupyter Lab 실행

`uv` 환경에서 Jupyter Lab을 실행합니다.

```powershell
.\uv.exe run jupyter lab
```

또는 포함된 배치 파일을 사용할 수 있습니다.

```powershell
.\run_uv_jupyter.bat
```

기존 Python 환경에 Jupyter가 이미 설치되어 있다면 다음 배치 파일도 사용할 수 있습니다.

```powershell
.\run_jupyter.bat
```

브라우저가 열리면 루트 폴더의 챕터별 노트북을 실행합니다.

## 6. 실습 노트북 순서

권장 순서는 다음과 같습니다.

1. `Chapter_1_Exercise_Vector Space.ipynb`
2. `Chapter_2_Exercise_Dataset.ipynb`
3. `Chapter_3_Excercise_Attention.ipynb`
4. `Chapter_3_Excercise_Viz_Multi_head_attention.ipynb`
5. `Chapter_4_Excercise_GPT.ipynb`
6. `Chapter_5_Excercise_Pretraining.ipynb`
7. `Chapter_6_Excercise_Finetuning_Classification.ipynb`
8. `Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb`
9. `Chapter_7_Exercise_Follow_Instructions.ipynb`
10. `Chapter_7_Exercise_Follow_Instructions_dpo.ipynb`

참고용 원본/보조 노트북과 코드는 `llm_from_scratch` 폴더에 있습니다.

## 7. 테스트 실행

테스트가 필요한 경우 다음 명령을 실행합니다.

```powershell
.\uv.exe run pytest
```

`pytest`가 설치되어 있지 않다는 오류가 나오면 개발 의존성으로 추가하거나 임시로 실행합니다.

```powershell
.\uvx.exe pytest
```

## 8. OpenAI API 키 사용 시

OpenAI API를 사용하는 노트북을 실행할 때는 환경 변수 또는 `.env` 파일에 API 키를 설정합니다.

PowerShell 환경 변수 예시:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

`.env` 파일 예시:

```text
OPENAI_API_KEY=your_api_key_here
```

API 키는 Git에 커밋하지 않도록 주의합니다.

## 9. 자주 발생하는 문제

### `python` 명령을 찾을 수 없음

Python 3.12 이상을 설치하고 PATH에 추가합니다. Windows에서는 Python 설치 시 `Add python.exe to PATH` 옵션을 켜는 것이 좋습니다.

### CUDA 또는 PyTorch 설치 문제

`pyproject.toml`은 CUDA 12.6용 PyTorch 인덱스를 사용합니다. GPU/CUDA 환경이 맞지 않는 경우 CPU용 PyTorch로 변경해야 할 수 있습니다.

### Jupyter 커널이 패키지를 찾지 못함

Jupyter를 반드시 `uv` 환경에서 실행합니다.

```powershell
.\uv.exe run jupyter lab
```

### 노트북 실행 중 데이터 파일을 찾지 못함

저장소 루트(`C:\work\llm_hands_on`)에서 Jupyter Lab을 실행했는지 확인합니다.

