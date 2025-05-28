# CFM MOLS (Continuous Flow Matching for Mutually Orthogonal Latin Squares)

이 프로젝트는 Continuous Flow Matching을 사용하여 Mutually Orthogonal Latin Squares (MOLS) 트리플을 찾는 구현체입니다.

## 프로젝트 구조

- `train_baseline.py`: 기본 학습 스크립트
- `pl_latin_model.py`: PyTorch Lightning 기반 Latin Square 모델 구현
- `orthogonality.py`: MOLS의 직교성 검증 및 관련 유틸리티
- `generate_latin_squares.py`: Latin Square 생성 스크립트

## 설치 방법

```bash
# 가상환경 생성 및 활성화
conda create -n CFM_MOLS python=3.10
conda activate CFM_MOLS

# 필요한 패키지 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pytorch-lightning
```

## 사용 방법

1. Latin Square 데이터 생성:
```bash
python generate_latin_squares.py
```

2. 모델 학습:
```bash
python train_baseline.py
```

## 참고 사항

- `latins.npy`: 생성된 Latin Square 데이터 파일 (gitignore에 포함되어 있음)
- DIFUSCO/: 관련 참조 구현 디렉토리