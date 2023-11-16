
#include <cstdio>
#include <fstream>
#include <iostream>
#include <runner.cuh>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Please select a kernel (range 0 - 12, 0 for NVIDIA cuBLAS)"
                << std::endl;
        exit(EXIT_FAILURE);
    }
    // get environment variable for device
    int deviceIdx = 0;
    if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
    }
    cudaCheck(cudaSetDevice(deviceIdx));

    printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

    std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};

    // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
    // publishing event tasks in the target stream
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    long m, n, k, max_size;
    max_size = SIZE[SIZE.size() - 1];
    std::cout << "Max size: " << max_size << std::endl;

    float *A = nullptr, *B = nullptr, *C = nullptr,
        *C_ref = nullptr; // host matrices
    float *dA = nullptr, *dB = nullptr, *dC = nullptr,
        *dC_ref = nullptr; // device matrices

    A = (float *)malloc(sizeof(float) * max_size * max_size);
    B = (float *)malloc(sizeof(float) * max_size * max_size);
    C = (float *)malloc(sizeof(float) * max_size * max_size);
    C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

    randomize_matrix(A, max_size * max_size);
    randomize_matrix(B, max_size * max_size);
    randomize_matrix(C, max_size * max_size);

    cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));

    cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size,
                        cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                        cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                        cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size,
                        cudaMemcpyHostToDevice));

    for (int size : SIZE) {
        m = n = k = size;

        std::cout << "dimensions(m=n=k) " << m << std::endl;
        // Verify the correctness of the calculation, and execute it once before the
        // kernel function timing to avoid cold start errors
        if (kernel_num != 0) {
            run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref); // cuBLAS
            run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC,
                        handle); // Executes the kernel, modifies the result matrix
            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
            cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

            if (!verify_matrix(C_ref, C, m * n)) {
                std::cout
                    << "Failed to pass the correctness verification against NVIDIA "
                    "cuBLAS."
                    << std::endl;
                if (m <= 128) {
                    std::cout << " Logging faulty output into " << errLogFile << "\n";
                    std::ofstream fs;
                    fs.open(errLogFile);
                    fs << "A:\n";
                    print_matrix(A, m, n, fs);
                    fs << "B:\n";
                    print_matrix(B, m, n, fs);
                    fs << "C:\n";
                    print_matrix(C, m, n, fs);
                    fs << "Should:\n";
                    print_matrix(C_ref, m, n, fs);
                }
                    exit(EXIT_FAILURE);
            }
        }
        float alpha = 1.0, beta = 0.0; // GEMM input parameters, C=α*AB+β*C
        cudaEventRecord(beg);
        for (int j = 0; j < repeat_times; j++) {
        // We don't reset dC between runs to save time
        run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.; // Convert to seconds

        long flops = 2 * m * n * k;
        printf(
            "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
            "(%ld).\n",
            elapsed_time / repeat_times,
            (repeat_times * flops * 1e-9) / elapsed_time, m);
        fflush(stdout);
        // make dC and dC_ref equal again (we modified dC while calling our kernel
        // for benchmarking)
        cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n,
                            cudaMemcpyDeviceToDevice));
    }
    // Free up CPU and GPU space
    free(A);
    free(B);
    free(C);
    free(C_ref);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dC_ref);

  return 0;
}