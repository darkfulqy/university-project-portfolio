#include <omp.h>

#include <iomanip>
#include <iostream>

using namespace std;

double serialPi(long long n) {
    double step = 1.0 / (double)n;
    double sum = 0.0;

    for (long long i = 0; i < n; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    return sum * step;
}

double parallelPi(long long n, int threadCount) {
    double step = 1.0 / (double)n;
    double sum = 0.0;

    omp_set_num_threads(threadCount);

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long long i = 0; i < n; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    return sum * step;
}

int main() {
    long long n;
    int threadCount;

    cout << "Input number of intervals n: ";
    cin >> n;
    cout << "Input thread count: ";
    cin >> threadCount;

    double start1 = omp_get_wtime();
    double pi1 = serialPi(n);
    double end1 = omp_get_wtime();

    double start2 = omp_get_wtime();
    double pi2 = parallelPi(n, threadCount);
    double end2 = omp_get_wtime();

    double serialTime = end1 - start1;
    double parallelTime = end2 - start2;

    cout << fixed << setprecision(12);
    cout << "\nSerial pi = " << pi1 << endl;
    cout << "Serial time = " << serialTime << " s" << endl;

    cout << "\nOpenMP pi = " << pi2 << endl;
    cout << "Parallel time = " << parallelTime << " s" << endl;

    if (parallelTime > 0) {
        cout << "\nSpeedup = " << serialTime / parallelTime << endl;
        cout << "Efficiency = " << serialTime / parallelTime / threadCount << endl;
    }

    return 0;
}
