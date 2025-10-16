#include <iostream>
#include <cmath>
#include <mpi.h>

void multiplyMatrixVector(const double* matrix, const double* vec, double* local_result, const int N, int start_row, int end_row) {
    int local_size = end_row - start_row;
    for (int local_i = 0; local_i < local_size; ++local_i) {
        int global_i = start_row + local_i;
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
            sum += matrix[global_i * N + j] * vec[j];
        }
        local_result[local_i] = sum;
    }
}

double ScalarProduct(const double* a, const double* b, int size) {
    double scalar_result = 0.0;
    for (int i = 0; i < size; ++i) {
        scalar_result += a[i] * b[i];
    }
    return scalar_result;
}

void SubVector(const double* a, const double* b, double* c, int size) {
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] - b[i];
    }
}

void methodMinimalResiduals(const double* A, const double* b, double* answ, const int N, int start_row, int end_row, int local_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double* local_Ar = new double[local_size];
    double* local_Ax = new double[local_size];
    double* local_r = new double[local_size];

    double* local_b = new double[local_size];
    for (int i = 0; i < local_size; i++) {
        local_b[i] = b[start_row + i];
    }

    double* Ar = new double[N];
    double* Ax = new double[N];
    double* r = new double[N];

    double tau = 0.0;
    double residual = 1.0;
    int k = 0;

    double local_norm_b = ScalarProduct(local_b, local_b, local_size);
    double norm_b;
    MPI_Allreduce(&local_norm_b, &norm_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    norm_b = sqrt(norm_b);

    int* number_in_process = new int[size];
    int* dislplacement = new int[size];


    for (int i = 0; i < size; i++) {
        int current_start = i * (N / size);
        int current_end = (i == size - 1) ? N : (i + 1) * (N / size);
        number_in_process[i] = current_end - current_start;
        dislplacement[i] = current_start;
    }

    while (residual > 1e-5 && k < 100) {
        k++;

        // Вычисляем Ax
        multiplyMatrixVector(A, answ, local_Ax, N, start_row, end_row);
        MPI_Allgatherv(local_Ax, local_size, MPI_DOUBLE, Ax, number_in_process, dislplacement, MPI_DOUBLE, MPI_COMM_WORLD);

        // Вычисляем r = b - Ax
        SubVector(local_b, local_Ax, local_r, local_size);
        MPI_Allgatherv(local_r, local_size, MPI_DOUBLE,
            r, number_in_process, dislplacement, MPI_DOUBLE, MPI_COMM_WORLD);

        // Вычисляем Ar
        multiplyMatrixVector(A, r, local_Ar, N, start_row, end_row);
        MPI_Allgatherv(local_Ar, local_size, MPI_DOUBLE,
            Ar, number_in_process, dislplacement, MPI_DOUBLE, MPI_COMM_WORLD);

        // Вычисляем норму невязки
        double local_norm_r = ScalarProduct(local_r, local_r, local_size);
        double norm_r;
        MPI_Allreduce(&local_norm_r, &norm_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        residual = sqrt(norm_r) / norm_b;

        if (residual < 1e-5) break;

        // Вычисляем скалярное произведение (Ar, r)
        double local_Ar_r = ScalarProduct(local_Ar, local_r, local_size);
        double scalar_Ar_r;
        MPI_Allreduce(&local_Ar_r, &scalar_Ar_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Вычисляем скалярное проивзедение (Ar, Ar)
        double local_norm_Ar = ScalarProduct(local_Ar, local_Ar, local_size);
        double norm_Ar;
        MPI_Allreduce(&local_norm_Ar, &norm_Ar, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (fabs(norm_Ar) < 1e-5) break;

        //Вычисляем tau
        tau = scalar_Ar_r / norm_Ar;

        // Обновляем решение
        for (int local_i = 0; local_i < local_size; ++local_i) {
            int global_i = start_row + local_i;
            answ[global_i] += tau * r[global_i];
        }

        // Синхронизируем решение
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            answ, number_in_process, dislplacement, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        std::cout << "Total iterations: " << k << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int psize, prank;
    MPI_Comm_size(MPI_COMM_WORLD, &psize);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    int N = 10000;

    int rows_per_process = N / psize;
    int start_row = prank * rows_per_process;
    int end_row = (prank == psize - 1) ? N : start_row + rows_per_process;
    int local_size = end_row - start_row;

    double* matrixA = new double[N * N];
    double* b = new double[N];
    double* answer = new double[N];

    if (prank == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j)
                    matrixA[i * N + j] = 2.0 * N + 1.0;
                else
                    matrixA[i * N + j] = 1.0;
            }
            b[i] = 1.0;
            answer[i] = 0.0;
        }
    }

    //Рассылаем из 0 процесса данные СЛАУ по всем процессам
    MPI_Bcast(matrixA, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(answer, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    methodMinimalResiduals(matrixA, b, answer, N, start_row, end_row, local_size);
    double end_time = MPI_Wtime();

    if (prank == 0) {
        std::cout << "Execution time = " << end_time - start_time << " seconds" << std::endl;

        // Вывод первых 5 элементов решения
        std::cout << "First 5 elements of solution: ";
        for (int i = 0; i < 5; i++) {
            std::cout << answer[i] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}