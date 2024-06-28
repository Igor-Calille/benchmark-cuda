#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// Kernel para adicionar dois vetores
__global__ void add(int n, float* x, float* y, float* out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        out[i] = x[i] + y[i];
    }
}

int main(void) {
    int N = 10000000; 
    float* x, * y, * out;

    // Alocar memória unificada
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));
    cudaMallocManaged(&out, N * sizeof(float));

    // Inicializar os vetores x e y no host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Configurar os blocos e threads
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Medir o tempo de execução do kernel
    auto start = std::chrono::high_resolution_clock::now();

    // Executar o kernel para adicionar os vetores
    add << <numBlocks, blockSize >> > (N, x, y, out);

    // Esperar a GPU finalizar antes de acessar os resultados no host
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

    // Verificar por erros (todos os valores devem ser 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(out[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    // Liberar memória
    cudaFree(x);
    cudaFree(y);
    cudaFree(out);

    return 0;
}
