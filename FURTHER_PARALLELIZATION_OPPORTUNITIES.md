# 추가 병렬화 가능성 분석 보고서

## 1. 개요
현재 CUDA 포팅은 핵심 연산인 `PatchGrid::Optimize` (Patch 최적화) 단계에 집중되어 있습니다. 그러나 전체 파이프라인의 성능을 극대화하기 위해서는 아직 CPU에서 수행되거나 최적화 여지가 있는 부분들에 대한 추가 포팅이 필요합니다.

## 2. 병렬화 가능 영역 (Parallelization Opportunities)

### 2.1. Densification 단계 (`AggregateFlowDense`)
*   **현황**: `patchgrid.cpp` 내의 `AggregateFlowDense` 함수는 최적화된 Patch의 변위(Flow)를 픽셀 단위의 Dense Flow로 변환하는 과정을 **CPU에서 수행**하고 있습니다.
*   **문제점**:
    *   수천 개의 Patch 결과를 픽셀 그리드에 누적(Scatter)하는 작업으로, CPU에서는 반복문을 통해 수행되며 메모리 접근이 불규칙할 수 있습니다.
    *   OpenMP 적용 시 Race Condition 우려로 인해 주석 처리되어 있거나 제한적으로 사용됩니다.
*   **개선 방안 (GPU 커널 구현)**:
    *   **Gather 방식 (권장)**: 각 픽셀 스레드가 자신에게 영향을 주는 Patch들을 역으로 계산하여 가중 평균을 구하는 방식입니다. Race Condition 없이 완전한 병렬 처리가 가능합니다. (Patch 위치가 규칙적인 격자이므로 인덱싱 가능)
    *   **Atomic Add 방식**: Patch 단위 스레드가 자신의 영역에 값을 더할 때 `atomicAdd`를 사용하는 방식입니다.

### 2.2. Variational Refinement (`refine_variational.cpp`)
*   **현황**: Patch 단계 후 흐름을 매끄럽게 보정하는 Variational Refinement 단계가 **전적으로 CPU에서 수행**됩니다.
*   **문제점**:
    *   픽셀 단위의 반복 연산(SOR Solver, 미분, Warp 등)이 많아 해상도가 높을수록 큰 병목이 됩니다.
    *   `Optimize` 단계에서 GPU로 가속한 이득이 이 단계에서 다시 CPU로 내려오면서 희석됩니다.
*   **개선 방안**:
    *   전체 `VarRefClass` 로직을 CUDA로 포팅해야 합니다.
    *   SOR Solver는 Red-Black SOR 등의 기법을 사용하여 GPU 병렬 처리에 매우 적합합니다.
    *   Image Warping과 미분 계산 또한 텍스처 메모리를 활용하면 매우 빠르게 처리 가능합니다.

### 2.3. 전체 파이프라인 데이터 흐름 (Data Pipeline Optimization)
*   **현황**: Scale Loop(`oflow.cpp`)가 돌 때마다 **CPU <-> GPU 데이터 복사**가 빈번하게 발생합니다.
    *   이미지 피라미드 -> GPU 복사
    *   GPU 최적화 -> CPU로 결과 복사 (`h_p_out`)
    *   CPU에서 Densification -> CPU에서 다음 스케일 초기화 -> 다시 GPU로 복사
*   **개선 방안**:
    *   **GPU Resident Pipeline**: 이미지 로딩 후 피라미드 생성부터 마지막 결과 저장 직전까지 모든 데이터를 GPU 메모리에 상주시켜야 합니다.
    *   `InitializeFromCoarserOF`와 같은 초기화 로직도 GPU 커널로 구현하여, 데이터가 호스트로 돌아오지 않도록 해야 합니다.

### 2.4. 전처리 단계 (Preprocessing)
*   **현황**: `run_dense_cuda.cpp`의 `ConstructImgPyramide` 함수에서 `cv::resize`, `cv::Sobel` 등이 **CPU에서 실행**됩니다.
*   **개선 방안**:
    *   OpenCV CUDA 모듈(`cv::cuda::resize`, `cv::cuda::Sobel`)을 사용하거나 커스텀 커널을 작성하여 전처리 속도를 높일 수 있습니다.

## 3. 우선순위
1.  **High**: `AggregateFlowDense` GPU 포팅 (구현 난이도 낮음, 병목 해소 효과 큼)
2.  **High**: `Variational Refinement` GPU 포팅 (전체 수행 시간에서 큰 비중 차지)
3.  **Medium**: Data Pipeline 최적화 (Memcpy 오버헤드 제거)
4.  **Low**: Preprocessing (상대적으로 연산 비중 적음)
