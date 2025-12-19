# DIS Optical Flow CPU Baseline Benchmarking

CPU 성능 측정 및 분석을 위한 스크립트 모음이다.

## 준비 사항

### 1. 의존성 설치
```bash
# macOS
brew install opencv eigen

# Ubuntu/Linux
sudo apt-get install libopencv-dev libeigen3-dev
```

### 2. 테스트 이미지 준비
```bash
mkdir -p benchmark/test_images
cd benchmark/test_images

# Option 1: Sintel Dataset (권장)
# http://sintel.is.tue.mpg.de/downloads 에서 다운로드
# 'training/clean' 또는 'training/final' 폴더의 연속 프레임 사용

# Option 2: 자체 이미지
# 연속된 두 프레임을 frame1.png, frame2.png 형식으로 저장
```

## 사용 방법

### 1단계: 빌드 및 환경 구성
```bash
cd /Users/lizrd/Develops/class/CFDS2/OF_DIS
chmod +x benchmark/*.sh
./benchmark/setup_baseline.sh
```

이 스크립트는 두 가지 버전을 빌드한다:
- `build_cpu_nomp/`: OpenMP 없는 CPU 버전
- `build_cpu_omp/`: OpenMP 활성화 CPU 버전

### 2단계: 성능 측정
```bash
./benchmark/run_baseline_benchmark.sh
```

측정 항목:
- Operating Points 1, 2, 3 (각각 다른 품질/속도 trade-off)
- 각 이미지 쌍에 대한 반복 실행
- 단계별 시간 분해 (Pyramid, OFlow, Save)

### 3단계: 결과 분석
```bash
# 최신 결과 파일 찾기
LATEST=$(ls -t benchmark/results/baseline_*.csv | head -1)

# 분석 실행
python3 benchmark/analyze_results.py "$LATEST"

# LaTeX 표 생성 (보고서용)
python3 benchmark/analyze_results.py "$LATEST" --latex benchmark/results/table.tex
```

## 결과 해석

### 출력 예시
```
1. Overall Performance by Build Configuration
----------------------------------------
                time_total_ms
                mean   std    min    max   count
build_type
CPU-noMP       245.3  12.1   228.4  267.8  9
CPU-OMP        156.7   8.4   142.1  172.3  9

2. Performance by Operating Point
----------------------------------------
OpPoint 1: faster, lower quality
OpPoint 2: balanced (default)
OpPoint 3: slower, higher quality

3. OpenMP Speedup Analysis
----------------------------------------
1.56x average speedup with OpenMP
```

### 병목 분석
- **Pyramid 비중 높음**: 이미지 리사이즈/Sobel 필터가 주요 비용
- **OFlow 비중 높음**: 패치 최적화와 밀집화가 주요 비용
- **OpenMP 효과 제한적**: 메모리 대역폭 제약 또는 직렬 구간 존재

## GPU 가속 목표 설정

Baseline 측정 후 GPU 구현 목표:
1. **전체 파이프라인**: 3-6x speedup (Variational 제외)
2. **패치 최적화**: 5-10x speedup
3. **밀집화**: 3-5x speedup

## 추가 실험

### 다양한 해상도 테스트
```bash
# 이미지 리사이즈
for scale in 0.5 0.75 1.0 1.5; do
    convert original.png -resize ${scale}00% frame_${scale}.png
done
```

### OpenMP 스레드 수 조정
```bash
export OMP_NUM_THREADS=4
./build_cpu_omp/run_OF_INT img1.png img2.png out.flo 2
```

### 프로파일링 (Linux)
```bash
# perf로 hotspot 분석
perf record -g ./build_cpu_omp/run_OF_INT img1.png img2.png out.flo 2
perf report
```

## 파일 구조
```
benchmark/
├── setup_baseline.sh          # 빌드 스크립트
├── run_baseline_benchmark.sh  # 벤치마크 실행
├── analyze_results.py         # 결과 분석
├── README.md                  # 이 문서
├── test_images/               # 테스트 이미지 (사용자가 추가)
└── results/                   # 측정 결과 CSV
    ├── baseline_YYYYMMDD_HHMMSS.csv
    └── table.tex              # LaTeX 표
```

## 문제 해결

### "No test images found"
→ `benchmark/test_images/`에 연속 프레임 이미지를 추가한다.

### "OpenCV not found"
→ `brew install opencv` 또는 `apt-get install libopencv-dev`

### "Eigen3 not found"
→ `brew install eigen` 또는 `apt-get install libeigen3-dev`

### 성능이 예상보다 느림
→ Release 빌드 확인: `CMAKE_BUILD_TYPE=Release`
→ 최적화 플래그 확인: `-O3 -march=native`
