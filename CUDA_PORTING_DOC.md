# CUDA 포팅 및 병렬화 분석 보고서

## 1. 개요
본 문서는 Optical Flow 추정을 위한 OF_DIS 프로젝트의 CUDA 포팅 결과에서 적용된 주요 병렬화 기법과 최적화 포인트를 상세히 기술합니다.

## 2. 주요 병렬화 포인트 (Key Parallelization Points)

### 2.1. GPU 레벨 병렬화: PatchGrid 최적화 (`patchgrid_cuda.cu`)
핵심 연산인 PatchGrid 최적화 단계에서 다음과 같은 다계층 병렬화를 적용하였습니다.

*   **Patch 단위 병렬화 (Inter-Patch Parallelism)**
    *   **설명**: 각 Patch의 최적화 연산은 다른 Patch와 독립적이므로, 개별 Patch 처리를 하나의 CUDA Thread Block에 매핑하여 완전한 병렬 처리를 구현했습니다.
    *   **구현**: `OptimizePatchesKernel` 커널 실행 시 `blockIdx.x`를 Patch 인덱스로 사용하여, GPU의 모든 SM(Streaming Multiprocessor)이 수천 개의 Patch를 동시에 처리하도록 스케줄링했습니다.

*   **Pixel 단위 병렬화 (Intra-Patch Parallelism)**
    *   **설명**: 각 Patch 내부(예: 8x8, 12x12)의 픽셀 단위 연산(Gradient 추출, Error 계산, Hessian 구성 등)을 CUDA Thread에 매핑하였습니다.
    *   **구현**: `Dim3 block(p_samp_s, p_samp_s)` 형태로 스레드 블록을 구성하여, Patch 내의 모든 픽셀을 동시에 로드하고 연산합니다. 이는 SIMT(Single Instruction, Multiple Threads) 구조에 최적화된 방식입니다.

*   **Parallel Reduction (병렬 축소 연산)**
    *   **설명**: Patch 내의 평균값(Mean), SSD(Sum of Squared Differences), 내적(Dot Product) 계산 시 순차적 누적 대신 트리 구조의 병렬 Reduction 알고리즘을 적용했습니다.
    *   **효과**: `O(N)` 복잡도의 연산을 `O(log N)` 수준으로 단축시키며, 스레드 간 동기화(`__syncthreads()`)를 효율적으로 사용하여 연산 속도를 비약적으로 향상시켰습니다.

### 2.2. 메모리 아키텍처 최적화

*   **Shared Memory 활용을 통한 대역폭 최적화**
    *   **설명**: 최적화 루프 내에서 반복적으로 접근되는 Reference Patch 데이터, 미분값(Gradient), 그리고 중간 계산용 버퍼를 고속의 Shared Memory (`__shared__`)에 할당했습니다.
    *   **효과**: 느린 Global Memory 접근을 획기적으로 줄여 메모리 대역폭 병목(Memory Bandwidth Bottleneck)을 해소했습니다.

*   **Texture Memory 및 하드웨어 보간 (Hardware Interpolation)**
    *   **설명**: Target Image에서 임의의 실수 좌표(Floating-point coordinates) 값을 읽어올 때, CUDA Texture Object (`cudaTextureObject_t`)를 사용했습니다.
    *   **효과**: GPU 내장 Texture Unit 하드웨어가 Bilinear Interpolation을 수행하므로, 별도의 연산 비용 없이 고속으로 보간된 픽셀 값을 얻을 수 있습니다 (Zero-overhead Interpolation).

### 2.3. CPU/System 레벨 병렬화 (`run_dense_cuda.cpp`)

*   **OpenMP를 이용한 배치 데이터 병렬 처리**
    *   **설명**: 벤치마크나 대량의 데이터셋 처리 시, `main` 함수에서 OpenMP 라이브러리(`omp.h`)를 사용하여 이미지 쌍(Pair) 단위의 병렬 처리를 구현했습니다.
    *   **코드**: `#pragma omp parallel for schedule(dynamic)`
    *   **효과**: 멀티코어 CPU 자원을 활용하여 이미지 로딩, 전처리, 호스트 측 제어 로직을 병렬화함으로써 전체 파이프라인의 처리량(Throughput)을 높였습니다.

## 3. 결론
현재 구현된 포팅 결과는 **Fine-grained Parallelism(GPU의 Patch/Pixel 병렬화)**과 **Coarse-grained Parallelism(CPU의 배치 병렬화)**이 조화롭게 결합된 형태입니다. 
특히 `patchgrid_cuda.cu`에 구현된 커널은 GPU의 구조적 특성(Shared Memory, Texture Unit, Massive Threading)을 적극적으로 활용하도록 설계되어 있어, 기존 CPU 구현 대비 높은 성능 효율을 제공합니다.
