#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cfloat>
#include <algorithm>

// TODO: Implement kernel to find max value using parallel reduction
// Use shared memory for efficiency
__global__ void maxReduceKernel(const float* input, float* output, int n) {
    // TODO: Load elements into shared memory
    
    // TODO: Perform parallel reduction to find maximum
    // TODO: Write block result to output
}

// TODO: Implement kernel to compute exp(x - maxVal) for each element
// This ensures numerical stability
__global__ void computeExpKernel(const float* input, float* output, float maxVal, int n) {
    // TODO: Compute expf(input[idx] - maxVal) for each valid index
}

// TODO: Implement kernel to find sum using parallel reduction
// Use shared memory for efficiency
__global__ void sumReduceKernel(const float* input, float* output, int n) {
    // TODO: Load elements into shared memory
    // TODO: Perform parallel reduction to find sum
    // TODO: Write block result to output
}

// TODO: Implement kernel to normalize values by dividing by sum
__global__ void normalizeKernel(float* data, float sumVal, int n) {
    // TODO: Divide each element by sumVal
}

// Host function: Compute softmax of the input scores using CUDA
// Parameters:
//   scores - input vector of float scores
// Returns:
//   vector of softmax probabilities
std::vector<float> softmax(const std::vector<float>& scores) {
    // TODO:
    // 1. Handle edge case of empty input
    // 2. Allocate device memory for input, output, and partial results
    // 3. Copy input scores to device
    // 4. Launch maxReduceKernel to find max value
    // 5. Copy partial max results back and find global max on host
    // 6. Launch computeExpKernel with the max value
    // 7. Launch sumReduceKernel to find sum of exponentials
    // 8. Copy partial sum results back and find global sum on host
    // 9. Launch normalizeKernel to divide by sum
    // 10. Copy result back to host
    // 11. Free all device memory
    // 12. Return result vector
    return {};
}