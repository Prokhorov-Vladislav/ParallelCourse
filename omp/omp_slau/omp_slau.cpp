#include <iostream>
#include <omp.h>


void multiplyMatrixVector(const float* matrix, const float* vec, float* result, const int N) {
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        float sum = 0;
        for (int j = 0; j < N; ++j) {
            sum += matrix[i * N + j] * vec[j];
        }
        result[i] = sum;
    }
}

float ScalarProduct(const float* a, const float* b, int n) {
    double scalar_result = 0.0;
#pragma omp parallel for reduction(+:scalar_result)
    for (int i = 0; i < n; ++i) {
        scalar_result += a[i] * b[i];
    }
    return scalar_result;
}

void SubVector(const float* a, const float* b, float* c, int n) {
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] - b[i];
    }
}



void methodMinimalResiduals(const float* A, const float* b, float* answ, const int N) {
    float* Ar = new float[N];
    float* Ax = new float[N];
    float* r = new float[N];
    float tau = 0;
    float norm_coefficient = sqrt(ScalarProduct(b, b, N));
    float residual = 1;
    int k = 0;

    while (residual > 1e-5) {
        k++;
        multiplyMatrixVector(A, answ, Ax, N);
        SubVector(b, Ax, r, N);
        multiplyMatrixVector(A, r, Ar, N);
        residual = sqrt(ScalarProduct(r, r, N)) / norm_coefficient;
        if (residual < 1e-3) break;
        tau = ScalarProduct(Ar, r, N) / ScalarProduct(Ar, Ar, N);

        for (int i = 0; i < N; ++i) {
            answ[i] += tau * r[i];
        }
    }
    std::cout << k << std::endl;
}


int main()
{
    omp_set_num_threads(12);
    double start = omp_get_wtime();
    int N = 10000;
    float* matrixA = new float[N * N];
    float* b = new float[N];
    float* answer = new float[N];
    for (int i = 0; i < N; ++i) {
        b[i] = 1;
        answer[i] = 0;
    }

#pragma omp parallel for
    for (int i = 0; i < N * N; ++i) {
        if (i / N == i % N) matrixA[i] = 2 * N + 1;
        else matrixA[i] = 1;
    }

    methodMinimalResiduals(matrixA, b, answer, N);
    //for (int i = 0; i < 5; ++i) {
    //    std::cout << answer[i] << std::endl;
    //}
    double end = omp_get_wtime();
    printf("Execution time: %.6f seconds\n", end - start);
}
