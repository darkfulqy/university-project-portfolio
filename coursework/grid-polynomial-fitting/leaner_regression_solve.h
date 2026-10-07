#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>
#include <string>
using namespace std;

//基本结构体
struct Point {
    double x;
    double y;
};

struct GridCell {
    int id;      // 原始序号 0~15
    int row;     // 行
    int col;     // 列

    // 角点
    Point p[4];

    // 四条边上的两个端点
    Point top[2];    // 上边：右上 左上
    Point bottom[2]; // 下边：右下 左下
    Point left[2];   // 左边：左上 左下
    Point right[2];  // 右边：右上 右下
};

//高斯消元法
vector<double> gauss_solve(vector<vector<double>> A, vector<double> b) {
    int n = A.size();
    for (int i = 0; i < n; i++) A[i].push_back(b[i]); // 构造增广矩阵

    for (int i = 0; i < n; i++) {
        // 选主元
        double maxv = fabs(A[i][i]);
        int maxRow = i;
        for (int k = i + 1; k < n; k++) {
            if (fabs(A[k][i]) > maxv) {
                maxv = fabs(A[k][i]);
                maxRow = k;
            }
        }
        swap(A[i], A[maxRow]);

        // 归一化主元行
        double diag = A[i][i];
        if (fabs(diag) < 1e-12) {
            cerr << "矩阵奇异！" << endl;
            exit(1);
        }
        for (int j = i; j <= n; j++) A[i][j] /= diag;

        // 消元
        for (int k = 0; k < n; k++) {
            if (k == i) continue;
            double factor = A[k][i];
            for (int j = i; j <= n; j++)
                A[k][j] -= factor * A[i][j];
        }
    }

    // 取结果
    vector<double> x(n);
    for (int i = 0; i < n; i++) x[i] = A[i][n];
    return x;
}

// 多项式最小二乘拟合
// 输入: x, y 数据；degree 多项式阶数
// 输出: 系数向量 coeff[0..degree] 以及 R²
double polyfit(const vector<double> &x, const vector<double> &y, int degree,
               vector<double> &coeff, const string &savefile = "") {
    int n = x.size();
    vector<vector<double>> A(degree + 1, vector<double>(degree + 1, 0.0));
    vector<double> b(degree + 1, 0.0);

    // 构造正规方程 A * coeff = b
    for (int i = 0; i <= degree; i++) {
        for (int j = 0; j <= degree; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++)
                sum += pow(x[k], i + j);
            A[i][j] = sum;
        }
        double sum_y = 0;
        for (int k = 0; k < n; k++)
            sum_y += y[k] * pow(x[k], i);
        b[i] = sum_y;
    }

    coeff = gauss_solve(A, b);

    // 计算R²
    double y_mean = 0;
    for (auto v : y) y_mean += v;
    y_mean /= n;

    double ss_tot = 0, ss_res = 0;
    for (int i = 0; i < n; i++) {
        double y_pred = 0;
        for (int j = 0; j <= degree; j++)
            y_pred += coeff[j] * pow(x[i], j);
        ss_tot += pow(y[i] - y_mean, 2);
        ss_res += pow(y[i] - y_pred, 2);
    }
    double r2 = 1 - ss_res / ss_tot;

    // 保存拟合结果（如果需要）
    if (!savefile.empty()) {
        ofstream fout(savefile);
        for (int i = 0; i < n; i++) {
            double y_pred = 0;
            for (int j = 0; j <= degree; j++)
                y_pred += coeff[j] * pow(x[i], j);
            fout << x[i] << " " << y_pred << endl;
        }
        fout.close();
    }

    return r2;
}

// 构造 5 行 5 列数据 
// cells:所有网格单元的信息（行、列、四个顶点、四条边）
void build_lines_from_file(const string &filename,
                           vector<vector<Point>> &row_lines,
                           vector<vector<Point>> &col_lines,
                           vector<GridCell> &cells) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << " 无法打开文件：" << filename << endl;
        exit(1);
    }

    row_lines.assign(5, vector<Point>()); // 5 条横线
    col_lines.assign(5, vector<Point>()); // 5 条竖线
    cells.clear();

    string line;
    while (getline(fin, line)) {
        if (line.empty()) continue;

        stringstream ss(line);
        GridCell cell;
        ss >> cell.id;  // 第一列是网格编号

        double x1, y1, x2, y2, x3, y3, x4, y4;
        if (!(ss >> x1 >> y1 >> x2 >> y2 >> x3 >> y3 >> x4 >> y4)) {
            cerr << "行解析失败，跳过: " << line << endl;
            continue;
        }

   
        cell.p[0] = {x1, y1}; 
        cell.p[1] = {x2, y2}; 
        cell.p[2] = {x3, y3}; 
        cell.p[3] = {x4, y4}; 

        
        cell.col = cell.id / 4;  
        cell.row = cell.id % 4;  

        // 由四个角点得到四条边
        Point RD = cell.p[0];
        Point RU = cell.p[1];
        Point LU = cell.p[2];
        Point LD = cell.p[3];

        
        cell.top[0] = RU;
        cell.top[1] = LU;
        
        cell.bottom[0] = RD;
        cell.bottom[1] = LD;
        
        cell.left[0] = LU;
        cell.left[1] = LD;
       
        cell.right[0] = RU;
        cell.right[1] = RD;

        int r = cell.row; 
        int c = cell.col; 

        // 当前格的上边属于第 r 条横线
        row_lines[r].push_back(cell.top[0]);
        row_lines[r].push_back(cell.top[1]);
        // 当前格的下边属于第 r+1 条横线
        row_lines[r + 1].push_back(cell.bottom[0]);
        row_lines[r + 1].push_back(cell.bottom[1]);

        // 当前格的左边属于第 c 条竖线
        col_lines[c].push_back(cell.left[0]);
        col_lines[c].push_back(cell.left[1]);
        // 当前格的右边属于第 c+1 条竖线
        col_lines[c + 1].push_back(cell.right[0]);
        col_lines[c + 1].push_back(cell.right[1]);

        cells.push_back(cell);
    }

    fin.close();
}
