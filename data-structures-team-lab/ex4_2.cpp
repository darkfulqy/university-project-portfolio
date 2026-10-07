#include <iostream>
#include <numeric>   // for std::gcd
using namespace std;

struct Result {
    long long leftCount;
    long long rightCount;
};

// Recursive version: move from the target node back to its parent each time.
Result countPathRecursiveImpl(long long a, long long b) {
    if (a <= 0 || b <= 0) {
        return {-1, -1};
    }

    if (a == 1 && b == 1) {
        return {0, 0};
    }

    if (a > b) {
        Result parent = countPathRecursiveImpl(a - b, b);
        if (parent.leftCount == -1) {
            return parent;
        }
        parent.leftCount++;
        return parent;
    }

    if (b > a) {
        Result parent = countPathRecursiveImpl(a, b - a);
        if (parent.leftCount == -1) {
            return parent;
        }
        parent.rightCount++;
        return parent;
    }

    return {-1, -1};
}

Result countPathRecursive(long long a, long long b) {
    if (a <= 0 || b <= 0 || std::gcd(a, b) != 1) {
        return {-1, -1};
    }

    return countPathRecursiveImpl(a, b);
}

// 朴素版：每次只减一次
Result countPathSlow(long long a, long long b) {
    Result res = {0, 0};

    // 非法情况：题目通常默认输入合法，这里只是补保护
    if (a <= 0 || b <= 0 || std::gcd(a, b) != 1) {
        return {-1, -1};
    }

    while (!(a == 1 && b == 1)) {
        if (a > b) {
            a -= b;
            res.leftCount++;
        } else if (b > a) {
            b -= a;
            res.rightCount++;
        } else {
            // 若 a == b 且不等于 (1,1)，说明不是树中合法结点
            return {-1, -1};
        }
    }

    return res;
}

// 优化版：一次跳过连续多步
Result countPathFast(long long a, long long b) {
    Result res = {0, 0};

    if (a <= 0 || b <= 0 || std::gcd(a, b) != 1) {
        return {-1, -1};
    }

    while (a != 1 && b != 1) {
        if (a > b) {
            // 连续回退若干次左分支
            long long k = (a - 1) / b;
            res.leftCount += k;
            a -= k * b;
        } else if (b > a) {
            // 连续回退若干次右分支
            long long k = (b - 1) / a;
            res.rightCount += k;
            b -= k * a;
        } else {
            return {-1, -1};
        }
    }

    // 收尾：其中一个已经是 1
    if (a == 1) {
        res.rightCount += (b - 1);
    } else {
        res.leftCount += (a - 1);
    }

    return res;
}

int main() {
    int n;
    cin >> n;

    while (n--) {
        long long a, b;
        cin >> a >> b;

        // 交题时建议用优化版
        Result ans = countPathFast(a, b);

        cout << ans.leftCount << " " << ans.rightCount << '\n';

        // 如果你想测试两个函数是否一致，可以临时改成下面这样：
        /*
        Result ans0 = countPathRecursive(a, b);
        Result ans1 = countPathSlow(a, b);
        Result ans2 = countPathFast(a, b);
        cout << "recursive: " << ans0.leftCount << " " << ans0.rightCount << '\n';
        cout << "slow:      " << ans1.leftCount << " " << ans1.rightCount << '\n';
        cout << "fast:      " << ans2.leftCount << " " << ans2.rightCount << '\n';
        */
    }

    return 0;
}





































































































            
