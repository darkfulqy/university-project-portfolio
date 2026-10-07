#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kBlockSize = 16;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " in " << #call << ": " << cudaGetErrorString(err__)  \
                      << std::endl;                                             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

__global__ void matrixMultiplyKernel(const int* a, const int* b, long long* c, int n) {
    __shared__ int tileA[kBlockSize][kBlockSize];
    __shared__ int tileB[kBlockSize][kBlockSize];

    const int row = blockIdx.y * kBlockSize + threadIdx.y;
    const int col = blockIdx.x * kBlockSize + threadIdx.x;
    long long sum = 0;

    for (int tile = 0; tile < n; tile += kBlockSize) {
        const int aCol = tile + threadIdx.x;
        const int bRow = tile + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < n && aCol < n) ? a[row * n + aCol] : 0;
        tileB[threadIdx.y][threadIdx.x] = (bRow < n && col < n) ? b[bRow * n + col] : 0;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kBlockSize; ++k) {
            sum += 1LL * tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

struct Config {
    std::vector<int> sizes{512, 1024, 2048};
    int runs = 3;
    std::string csvPath = "matrix_cuda_perf.csv";
    bool printSmall = false;
};

struct RunResult {
    int n = 0;
    int run = 0;
    double totalMs = 0.0;
    float kernelMs = 0.0f;
    long long cudaChecksum = 0;
    long long expectedChecksum = 0;
    bool correct = false;
};

std::vector<int> parseSizes(const std::string& text) {
    std::vector<int> sizes;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            sizes.push_back(std::stoi(item));
        }
    }
    return sizes;
}

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [--sizes 512,1024,2048] [--n 1024]\n"
              << "       [--runs 3] [--csv matrix_cuda_perf.csv] [--print-small]\n";
}

Config parseArgs(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (arg == "--sizes" && i + 1 < argc) {
            config.sizes = parseSizes(argv[++i]);
        } else if (arg == "--n" && i + 1 < argc) {
            config.sizes = {std::stoi(argv[++i])};
        } else if (arg == "--runs" && i + 1 < argc) {
            config.runs = std::stoi(argv[++i]);
        } else if (arg == "--csv" && i + 1 < argc) {
            config.csvPath = argv[++i];
        } else if (arg == "--print-small") {
            config.printSmall = true;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            printUsage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }

    if (config.sizes.empty()) {
        std::cerr << "At least one matrix size is required.\n";
        std::exit(EXIT_FAILURE);
    }
    if (config.runs <= 0) {
        std::cerr << "Run count must be positive.\n";
        std::exit(EXIT_FAILURE);
    }
    for (int n : config.sizes) {
        if (n <= 0) {
            std::cerr << "Matrix size must be positive.\n";
            std::exit(EXIT_FAILURE);
        }
    }
    return config;
}

void fillMatrices(std::vector<int>& a, std::vector<int>& b) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dis(0, 9);

    for (int& value : a) {
        value = dis(gen);
    }
    for (int& value : b) {
        value = dis(gen);
    }
}

long long expectedProductChecksum(const std::vector<int>& a, const std::vector<int>& b, int n) {
    std::vector<long long> colSumA(n, 0);
    std::vector<long long> rowSumB(n, 0);

    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            colSumA[k] += a[i * n + k];
        }
    }
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            rowSumB[k] += b[k * n + j];
        }
    }

    long long checksum = 0;
    for (int k = 0; k < n; ++k) {
        checksum += colSumA[k] * rowSumB[k];
    }
    return checksum;
}

long long checksumMatrix(const std::vector<long long>& c) {
    return std::accumulate(c.begin(), c.end(), 0LL);
}

void printMatrixSample(const std::vector<long long>& c, int n) {
    if (n > 8) {
        return;
    }
    std::cout << "C =\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::setw(6) << c[i * n + j] << ' ';
        }
        std::cout << '\n';
    }
}

void printDeviceInfo() {
    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "CUDA device: " << prop.name << '\n'
              << "Compute capability: " << prop.major << "." << prop.minor << '\n'
              << "Global memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MiB\n"
              << "Kernel block: " << kBlockSize << "x" << kBlockSize << "\n\n";
}

RunResult runOnce(const std::vector<int>& a,
                  const std::vector<int>& b,
                  std::vector<long long>& c,
                  int n,
                  int run,
                  long long expectedChecksum) {
    const size_t elements = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytesInt = elements * sizeof(int);
    const size_t bytesLongLong = elements * sizeof(long long);

    int* dA = nullptr;
    int* dB = nullptr;
    long long* dC = nullptr;
    cudaEvent_t start{};
    cudaEvent_t stop{};

    const auto totalStart = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMalloc(&dA, bytesInt));
    CUDA_CHECK(cudaMalloc(&dB, bytesInt));
    CUDA_CHECK(cudaMalloc(&dC, bytesLongLong));
    CUDA_CHECK(cudaMemcpy(dA, a.data(), bytesInt, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, b.data(), bytesInt, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((n + kBlockSize - 1) / kBlockSize, (n + kBlockSize - 1) / kBlockSize);

    CUDA_CHECK(cudaEventRecord(start));
    matrixMultiplyKernel<<<grid, block>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernelMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, start, stop));
    CUDA_CHECK(cudaMemcpy(c.data(), dC, bytesLongLong, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    const auto totalEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> totalDuration = totalEnd - totalStart;

    const long long cudaChecksum = checksumMatrix(c);
    return RunResult{n,
                     run,
                     totalDuration.count(),
                     kernelMs,
                     cudaChecksum,
                     expectedChecksum,
                     cudaChecksum == expectedChecksum};
}

void writeCsvHeader(std::ofstream& out) {
    out << "mode,n,run,block_dim,total_ms,kernel_ms,cuda_checksum,expected_checksum,correct\n";
}

void writeCsvRow(std::ofstream& out, const RunResult& result) {
    out << "cuda," << result.n << ',' << result.run << ',' << kBlockSize << ','
        << std::fixed << std::setprecision(6) << result.totalMs << ','
        << result.kernelMs << ',' << result.cudaChecksum << ','
        << result.expectedChecksum << ',' << (result.correct ? "yes" : "no") << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    const Config config = parseArgs(argc, argv);
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));

    printDeviceInfo();

    std::ofstream csv(config.csvPath);
    if (!csv) {
        std::cerr << "Cannot open CSV output: " << config.csvPath << std::endl;
        return EXIT_FAILURE;
    }
    writeCsvHeader(csv);

    bool allCorrect = true;

    for (int n : config.sizes) {
        const size_t elements = static_cast<size_t>(n) * static_cast<size_t>(n);
        std::vector<int> a(elements);
        std::vector<int> b(elements);
        std::vector<long long> c(elements);

        fillMatrices(a, b);
        const long long expectedChecksum = expectedProductChecksum(a, b, n);

        std::cout << "Matrix size n = " << n << ", expected checksum = "
                  << expectedChecksum << '\n';

        for (int run = 1; run <= config.runs; ++run) {
            RunResult result = runOnce(a, b, c, n, run, expectedChecksum);
            writeCsvRow(csv, result);
            allCorrect = allCorrect && result.correct;

            std::cout << "  run " << run
                      << ": total = " << std::fixed << std::setprecision(3)
                      << result.totalMs << " ms, kernel = " << result.kernelMs
                      << " ms, checksum = " << result.cudaChecksum
                      << ", correct = " << (result.correct ? "yes" : "no") << '\n';

            if (config.printSmall && n <= 8 && run == 1) {
                printMatrixSample(c, n);
            }
        }
        std::cout << '\n';
    }

    std::cout << "CSV written to " << config.csvPath << '\n';
    return allCorrect ? EXIT_SUCCESS : EXIT_FAILURE;
}
