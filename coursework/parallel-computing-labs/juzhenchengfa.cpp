#include <omp.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

using namespace std;

void fillMatrix(int** m, int n) {
    static mt19937 gen(42);
    uniform_int_distribution<int> dis(0, 9);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[i][j] = dis(gen);
        }
    }
}

void clearMatrix(long long** m, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[i][j] = 0;
        }
    }
}

void printIntMatrix(int** m, int n, const char* name) {
    cout << name << " = " << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << setw(4) << m[i][j] << " ";
        }
        cout << endl;
    }
}

void printLongMatrix(long long** m, int n, const char* name) {
    cout << name << " = " << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << setw(6) << m[i][j] << " ";
        }
        cout << endl;
    }
}

long long checksum(long long** m, int n) {
    long long sum = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sum += m[i][j];
        }
    }
    return sum;
}

double multiplyMatrix(int** a, int** b, long long** c, int n, int threadCount, omp_sched_t sched, int chunk) {
    if (threadCount > 0) {
        omp_set_num_threads(threadCount);
    }
    omp_set_schedule(sched, chunk);
    clearMatrix(c, n);

    double start = omp_get_wtime();

    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            long long sum = 0;
            for (int k = 0; k < n; k++) {
                sum += 1LL * a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }

    double end = omp_get_wtime();
    return end - start;
}

int main(int argc, char* argv[]) {
    int n = 512;
    int threadCount = 4;
    int chunk = 1;

    cout << "Input matrix size n: ";
    cin >> n;
    cout << "Input thread count: ";
    cin >> threadCount;
    cout << "Input chunk size: ";
    cin >> chunk;

    int** a = new int*[n];
    int** b = new int*[n];
    long long** c = new long long*[n];

    for (int i = 0; i < n; i++) {
        a[i] = new int[n];
        b[i] = new int[n];
        c[i] = new long long[n];
    }

    fillMatrix(a, n);
    fillMatrix(b, n);

    cout << "Matrix size n = " << n << endl;
    cout << "Threads = " << threadCount << endl;
    cout << endl;

    double t1 = multiplyMatrix(a, b, c, n, threadCount, omp_sched_static, chunk);
    cout << "schedule(static," << chunk << ") time = " << fixed << setprecision(6) << t1 << " s" << endl;

    double t2 = multiplyMatrix(a, b, c, n, threadCount, omp_sched_dynamic, chunk);
    cout << "schedule(dynamic," << chunk << ") time = " << fixed << setprecision(6) << t2 << " s" << endl;

    double t3 = multiplyMatrix(a, b, c, n, threadCount, omp_sched_guided, chunk);
    cout << "schedule(guided," << chunk << ") time = " << fixed << setprecision(6) << t3 << " s" << endl;

    double bestTime = t1;
    const char* bestName = "static";

    if (t2 < bestTime) {
        bestTime = t2;
        bestName = "dynamic";
    }
    if (t3 < bestTime) {
        bestTime = t3;
        bestName = "guided";
    }

    cout << "Best schedule = " << bestName << endl;
    cout << "Result checksum = " << checksum(c, n) << endl;

    if (n <= 8) {
        cout << endl;
        printIntMatrix(a, n, "A");
        printIntMatrix(b, n, "B");
        printLongMatrix(c, n, "C");
    }

    for (int i = 0; i < n; i++) {
        delete[] a[i];
        delete[] b[i];
        delete[] c[i];
    }
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
