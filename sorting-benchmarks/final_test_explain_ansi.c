#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>  // 引入布尔类型支持

/*
 * 排序算法测试程序
 * 功能：测试冒泡排序、选择排序和快速排序在不同数据分布下的性能和稳定性
 * 统计指标：执行时间、比较次数、赋值次数、内存占用、递归深度（仅快速排序）
 */

// 高斯分布随机数生成函数（使用Box-Muller变换）
// 功能：生成符合高斯分布（正态分布）的随机数
// 参数：mean-均值，variance-方差
// 原理：通过均匀分布随机数转换为高斯分布
double GaussRand(double mean, double variance) {
    static double u, v;  // 静态变量保存中间结果
    static int phase = 0; // 相位标志，用于交替生成两个随机数
    double z;

    if (phase == 0) {
        // 生成两个[0,1)范围内的均匀分布随机数
        u = (rand() + 1.0) / (RAND_MAX + 1.0);
        v = (rand() + 1.0) / (RAND_MAX + 1.0);
        // Box-Muller变换公式，将均匀分布转换为高斯分布
        z = sqrt(-2.0 * log(u)) * sin(2.0 * M_PI * v);
    } else {
        // 利用对称性，避免重复计算
        z = sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
    }
    phase = 1 - phase; // 切换相位
    // 返回带均值和方差的高斯分布随机数
    return mean + z * sqrt(variance);
}

// 时间测量函数
// 功能：获取程序执行时间（秒）
// 参数：restart-是否重置计时器
double gettime(int restart) {
    double c = 1.0 / CLOCKS_PER_SEC; // 时钟周期到秒的转换系数
    static clock_t t = 0;            // 静态变量保存起始时间
    if (restart || t == 0) t = clock(); // 重置或初始化计时器
    return c * (clock() - t);          // 返回经过的时间
}

// 内存使用监控（用于空间复杂度分析）
static size_t maxMemoryUsage = 0;        // 最大内存使用量
static size_t currentMemoryUsage = 0;    // 当前内存使用量
static size_t quickSortMaxMemoryUsage = 0; // 快速排序专用最大内存使用量

// 自定义内存分配函数，用于监控内存使用
void* monitoredMalloc(size_t size) {
    void* ptr = malloc(size); // 调用标准内存分配函数
    if (ptr) {
        currentMemoryUsage += size; // 更新当前内存使用量
        // 更新全局最大内存使用量
        if (currentMemoryUsage > maxMemoryUsage) {
            maxMemoryUsage = currentMemoryUsage;
        }
        // 更新快速排序专用最大内存使用量
        if (currentMemoryUsage > quickSortMaxMemoryUsage) {
            quickSortMaxMemoryUsage = currentMemoryUsage;
        }
    }
    return ptr;
}

// 自定义内存释放函数，用于监控内存使用
void monitoredFree(void* ptr, size_t size) {
    if (ptr) {
        currentMemoryUsage -= size; // 更新当前内存使用量
        free(ptr); // 调用标准内存释放函数
    }
}

// 重置内存监控数据
void resetMemoryMonitor() {
    maxMemoryUsage = 0;
    currentMemoryUsage = 0;
    quickSortMaxMemoryUsage = 0;
}

// 获取当前最大内存使用量
size_t getMaxMemoryUsage() {
    return maxMemoryUsage;
}

// 获取快速排序的最大内存使用量
size_t getQuickSortMaxMemoryUsage() {
    size_t mem = quickSortMaxMemoryUsage;
    quickSortMaxMemoryUsage = 0; // 重置以便下次测试
    return mem;
}

// 快速排序递归深度监控
static int quickMaxDepth = 0; // 记录最大递归深度

// 带原始索引的元素结构（用于稳定性测试）
typedef struct {
    int value;          // 元素值
    int original_index; // 原始索引（用于判断排序后相同元素的相对顺序是否改变）
} Element;

// 交换两个元素
void swapElements(Element *a, Element *b) {
    Element temp = *a;
    *a = *b;
    *b = temp;
}

/*
 * 冒泡排序算法（稳定排序）
 * 伪代码：
 * function bubbleSort(array, size):
 *     for i from 0 to size-2:
 *         swapped = false
 *         for j from 0 to size-i-2:
 *             if array[j] > array[j+1]:
 *                 交换 array[j] 和 array[j+1]
 *                 swapped = true
 *         if not swapped:
 *             break
 *     return array
 * 特点：相邻元素比较，稳定排序，时间复杂度O(n2)
 */
void I_Bubble(Element *a, int size, long long *cmpCount, long long *assignCount) {
    int i, j, swapped;
    for (i = 0; i < size - 1; i++) {
        swapped = 0;
        for (j = 0; j < size - i - 1; j++) {
            (*cmpCount)++; // 记录比较次数
            if (a[j].value > a[j + 1].value) {
                swapElements(&a[j], &a[j + 1]); // 交换元素
                (*assignCount) += 3; // 一次交换相当于3次赋值操作
                swapped = 1;
            }
        }
        if (swapped == 0) break; // 如果没有交换发生，说明已排序完成
    }
}

/*
 * 选择排序算法（不稳定排序）
 * 伪代码：
 * function selectionSort(array, size):
 *     for i from 0 to size-2:
 *         minIndex = i
 *         for j from i+1 to size-1:
 *             if array[j] < array[minIndex]:
 *                 minIndex = j
 *         if minIndex != i:
 *             交换 array[i] 和 array[minIndex]
 *     return array
 * 特点：每次选择最小元素，不稳定排序，时间复杂度O(n2)
 */
void I_Select(Element *a, int size, long long *cmpCount, long long *assignCount) {
    int i, j, k;
    for (i = 0; i < size - 1; i++) {
        k = i; // 记录当前最小值的索引
        for (j = i + 1; j < size; j++) {
            (*cmpCount)++; // 记录比较次数
            if (a[j].value < a[k].value) k = j; // 更新最小值索引
        }
        if (k != i) {
            swapElements(&a[i], &a[k]); // 交换当前位置和最小值位置的元素
            (*assignCount) += 3; // 一次交换相当于3次赋值操作
        }
    }
}

/*
 * 三数取中法选择基准值（快速排序辅助函数）
 * 功能：从三个位置中选择中间值作为基准值，并将其放到数组末尾
 * 伪代码：
 * function medianOfThree(array, low, high):
 *     mid = low + (high - low) / 2
 *     if array[mid] < array[low]: 交换 array[mid] 和 array[low]
 *     if array[high] < array[low]: 交换 array[high] 和 array[low]
 *     if array[high] < array[mid]: 交换 array[high] 和 array[mid]
 *     交换 array[mid] 和 array[high-1]
 *     return array[high-1]
 * 目的：避免最坏情况，提高快速排序性能
 */
int medianOfThree(Element *a, int low, int high, long long *cmpCount, long long *assignCount) {
    int mid = low + (high - low) / 2;
    (*cmpCount)++;
    if (a[mid].value < a[low].value) {
        swapElements(&a[mid], &a[low]);
        (*assignCount) += 3;
    }
    (*cmpCount)++;
    if (a[high].value < a[low].value) {
        swapElements(&a[high], &a[low]);
        (*assignCount) += 3;
    }
    (*cmpCount)++;
    if (a[high].value < a[mid].value) {
        swapElements(&a[high], &a[mid]);
        (*assignCount) += 3;
    }
    swapElements(&a[mid], &a[high - 1]);
    (*assignCount) += 3;
    return a[high - 1].value;
}

/*
 * 快速排序算法（不稳定排序）
 * 伪代码：
 * function quickSort(array, low, high, depth):
 *     if depth > maxDepth: update maxDepth
 *     while low < high:
 *         if high - low < 10:
 *             对当前子数组使用插入排序（小数组优化）
 *             return
 *         pivot = medianOfThree(array, low, high)
 *         left = low, right = high - 1
 *         while true:
 *             left++ until array[left] >= pivot
 *             right-- until array[right] <= pivot
 *             if left < right: 交换 array[left] 和 array[right]
 *             else: break
 *         交换 array[high-1] 和 array[left]
 *         if 左子数组长度 < 右子数组长度:
 *             quickSort(左子数组, depth+1)
 *             low = left + 1
 *         else:
 *             quickSort(右子数组, depth+1)
 *             high = left - 1
 * 特点：分治思想，平均时间复杂度O(nlogn)，最坏O(n2)，不稳定排序
 */
void I_Qsort(Element *a, int low, int high, long long *cmpCount, long long *assignCount, int depth) {
    if (depth > quickMaxDepth) quickMaxDepth = depth; // 更新最大递归深度
    
    while (low < high) {
        // 小数组优化：当子数组长度小于10时使用插入排序
        if (high - low < 10) {
            int i, j;
            for (i = low + 1; i <= high; i++) {
                Element temp = a[i];
                (*assignCount)++;
                j = i;
                while (j > low) {
                    (*cmpCount)++;
                    if (!(a[j - 1].value > temp.value)) break; // 找到插入位置
                    a[j] = a[j - 1];
                    (*assignCount)++;
                    j--;
                }
                a[j] = temp;
                (*assignCount)++;
            }
            return;
        }

        int pivot = medianOfThree(a, low, high, cmpCount, assignCount); // 三数取中选择基准值
        int left = low, right = high - 1;

        // 分区过程：将数组分为小于基准值和大于基准值两部分
        while (1) {
            do { left++; (*cmpCount)++; } while (a[left].value < pivot); // 找到第一个大于等于基准值的元素
            do { right--; (*cmpCount)++; } while (a[right].value > pivot); // 找到第一个小于等于基准值的元素
            (*cmpCount)++;
            if (left < right) {
                swapElements(&a[left], &a[right]); // 交换元素
                (*assignCount) += 3;
            } else {
                break; // 分区完成
            }
        }

        swapElements(&a[high - 1], &a[left]); // 将基准值放到正确位置
        (*assignCount) += 3;

        // 递归排序子数组（优化：先处理较短的子数组，减少递归深度）
        if (left - low < high - left) {
            I_Qsort(a, low, left - 1, cmpCount, assignCount, depth + 1);
            low = left + 1;
        } else {
            I_Qsort(a, left + 1, high, cmpCount, assignCount, depth + 1);
            high = left - 1;
        }
    }
}

// 获取当前最大递归深度并重置
int getQuickMaxDepth() {
    int depth = quickMaxDepth;
    quickMaxDepth = 0; // 重置以便下次测试
    return depth;
}

/*
 * 测试排序算法的稳定性
 * 功能：检查排序后相同值元素的相对顺序是否与原始顺序一致
 * 原理：遍历数组，若发现相同值元素的原始索引顺序颠倒，则不稳定
 */
bool isSortStable(Element *a, int size) {
    for (int i = 1; i < size; i++) {
        if (a[i].value == a[i-1].value) {
            if (a[i].original_index < a[i-1].original_index) {
                return false; // 发现顺序颠倒，不稳定
            }
        }
    }
    return true; // 所有相同元素顺序保持不变，稳定
}

/*
 * 测试排序算法的函数
 * 功能：对指定排序算法进行性能测试
 * 流程：创建数据副本 -> 执行排序 -> 记录时间和操作次数 -> 测试稳定性 -> 释放内存
 */
double testSortAlgorithm(void (*sortFunc)(Element*, int, long long*, long long*), 
                         Element* data, int size, long long *cmpCount, long long *assignCount, bool *isStable) {
    resetMemoryMonitor(); // 重置内存监控
    
    // 创建数据副本，避免修改原始数据
    Element* copy = (Element*)monitoredMalloc(size * sizeof(Element));
    if (!copy) {
        printf("内存分配失败！\n");
        exit(1);
    }
    
    for (int i = 0; i < size; i++) {
        copy[i] = data[i]; // 复制原始数据
    }

    *cmpCount = 0;       // 初始化比较次数
    *assignCount = 0;    // 初始化赋值次数
    double startTime = gettime(1); // 开始计时
    
    sortFunc(copy, size, cmpCount, assignCount); // 执行排序算法
    
    double endTime = gettime(0); // 结束计时
    
    *isStable = isSortStable(copy, size); // 测试稳定性
    
    monitoredFree(copy, size * sizeof(Element)); // 释放内存
    return endTime - startTime; // 返回执行时间
}

/*
 * 测试快速排序的函数（重载版本）
 * 功能：专门测试快速排序，同时记录递归深度和内存占用
 */
double testQuickSort(Element* data, int size, long long *cmpCount, long long *assignCount, 
                     size_t *memoryUsage, bool *isStable) {
    resetMemoryMonitor(); // 重置内存监控
    quickMaxDepth = 0;     // 重置递归深度监控
    
    // 创建数据副本
    Element* copy = (Element*)monitoredMalloc(size * sizeof(Element));
    if (!copy) {
        printf("内存分配失败！\n");
        exit(1);
    }
    
    for (int i = 0; i < size; i++) {
        copy[i] = data[i]; // 复制原始数据
    }

    *cmpCount = 0;       // 初始化比较次数
    *assignCount = 0;    // 初始化赋值次数
    double startTime = gettime(1); // 开始计时
    
    I_Qsort(copy, 0, size - 1, cmpCount, assignCount, 1); // 执行快速排序
    
    double endTime = gettime(0); // 结束计时
    
    *memoryUsage = getQuickSortMaxMemoryUsage(); // 获取内存使用量
    *isStable = isSortStable(copy, size); // 测试稳定性
    
    monitoredFree(copy, size * sizeof(Element)); // 释放内存
    return endTime - startTime; // 返回执行时间
}

int main() {
    srand(time(NULL)); // 初始化随机数生成器

    int baseSize = 1024;  // 基础数据规模
    int numTests = 7;     // 测试数据规模数量（2^0到2^6倍）
    int distributions = 4; // 数据分布类型数量

    // 数据分布类型名称
    const char* distNames[] = {"正序数组", "逆序数组", "均匀分布", "正态分布"};

    // 对每种数据分布进行测试
    for (int dist = 0; dist < distributions; dist++) {
        printf("==================== %s 排序测试结果 ====================\n", distNames[dist]);
        // 输出表头（包含各项统计指标）
        printf("%-12s\t%-16s\t%-10s\t%-10s\t%-10s\t%-10s\t%-16s\t%-10s\t%-10s\t%-10s\t%-10s\t%-16s\t%-10s\t%-10s\t%-10s\t%-10s\t%-10s\n", 
               "数据规模", "冒泡排序(秒)", "比较", "赋值", "内存(字节)", "稳定性", 
               "选择排序(秒)", "比较", "赋值", "内存(字节)", "稳定性", 
               "快速排序(秒)", "比较", "赋值", "递归深度", "内存(字节)", "稳定性");
        printf("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");

        // 对每种数据规模进行测试
        for (int test = 0; test < numTests; test++) {
            int size = baseSize * (1 << test); // 计算当前数据规模（2的test次方倍）
            printf("%-12d\t", size); // 输出数据规模

            // 分配内存存储原始数据
            Element* original = (Element*)monitoredMalloc(size * sizeof(Element));
            if (!original) {
                printf("内存分配失败！\n");
                return 1;
            }

            // 根据分布类型初始化数组
            switch (dist) {
                case 0: // 正序数组
                    for (int i = 0; i < size; i++) {
                        original[i].value = i / 2; // 故意创建重复值以便测试稳定性
                        original[i].original_index = i; // 记录原始索引
                    }
                    break;
                case 1: // 逆序数组
                    for (int i = 0; i < size; i++) {
                        original[i].value = (size - i - 1) / 2; // 故意创建重复值
                        original[i].original_index = i;
                    }
                    break;
                case 2: // 均匀分布
                    for (int i = 0; i < size; i++) {
                        original[i].value = rand() % (size / 2); // 生成0到size/2之间的随机数，增加重复概率
                        original[i].original_index = i;
                    }
                    break;
                case 3: // 正态分布
                    for (int i = 0; i < size; i++) {
                        original[i].value = (int)GaussRand(size/4, size/8); // 生成以size/4为均值的正态分布随机数
                        original[i].original_index = i;
                    }
                    break;
            }

            // 定义变量存储测试结果
            long long cmpBubble, assignBubble, cmpSelect, assignSelect, cmpQuick, assignQuick;
            double timeBubble, timeSelect, timeQuick;
            size_t memBubble, memSelect, memQuick;
            bool stableBubble, stableSelect, stableQuick;
            int depthQuick;

            // 测试冒泡排序
            timeBubble = testSortAlgorithm(I_Bubble, original, size, &cmpBubble, &assignBubble, &stableBubble);
            memBubble = getMaxMemoryUsage();
            printf("%.6f\t%10lld\t%10lld\t%10zu\t%-10s\t", timeBubble, cmpBubble, assignBubble, memBubble, stableBubble ? "稳定" : "不稳定");

            // 测试选择排序
            timeSelect = testSortAlgorithm(I_Select, original, size, &cmpSelect, &assignSelect, &stableSelect);
            memSelect = getMaxMemoryUsage();
            printf("%.6f\t%10lld\t%10lld\t%10zu\t%-10s\t", timeSelect, cmpSelect, assignSelect, memSelect, stableSelect ? "稳定" : "不稳定");

            // 测试快速排序
            timeQuick = testQuickSort(original, size, &cmpQuick, &assignQuick, &memQuick, &stableQuick);
            depthQuick = getQuickMaxDepth();
            printf("%.6f\t%10lld\t%10lld\t%10d\t%10zu\t%-10s\n", 
                   timeQuick, cmpQuick, assignQuick, depthQuick, memQuick, stableQuick ? "稳定" : "不稳定");

            monitoredFree(original, size * sizeof(Element)); // 释放原始数据内存
        }
        printf("\n"); // 每组数据分布测试后换行
    }

    return 0;
}