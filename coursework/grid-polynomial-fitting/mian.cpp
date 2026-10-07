#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include "leaner_regression_solve.h"
using namespace std;

int main() {
    vector<vector<Point>> row_lines;
    vector<vector<Point>> col_lines;
    vector<GridCell> cells;

    build_lines_from_file("data.txt", row_lines, col_lines, cells);

    cout << "读取到网格数量: " << cells.size() << endl;

    int degree;
    cout << "请输入拟合多项式的阶数 : ";
    cin >> degree;
    if (degree < 1) {
        cerr << "阶数必须 >= 1" << endl;
        return 1;
    }

    const double canvas_w = 256.0; 
    const double canvas_h = 256.0; 

    //  横向网格线 
    cout << "\n横向网格线拟合结果 (y = a0 + a1*x + ...):" << endl;
    for (int i = 0; i < 5; ++i) {
        vector<double> xs, ys;
        for (const auto &pt : row_lines[i]) {
            xs.push_back(pt.x);
            ys.push_back(pt.y);
        }

        vector<double> coeff;
        double r2 = polyfit(xs, ys, degree, coeff);

        cout << "Row line " << i << ": y = ";
        cout << fixed << setprecision(6);
        for (int j = 0; j <= degree; ++j) {
            if (j > 0) cout << " + ";
            cout << coeff[j] << "*x^" << j;
        }
        cout << "    (R²=" << r2 << ")" << endl;

        // 保存拟合曲线 
        stringstream fname;
        fname << "row_fit_" << i << ".txt";
        ofstream fout(fname.str());
        int steps = 300;
        for (int k = 0; k <= steps; ++k) {
            double x = canvas_w * k / steps;
            double y = 0;
            for (int j = 0; j <= degree; ++j)
                y += coeff[j] * pow(x, j);
            fout << x << " " << y << "\n";
        }
        fout.close();
    }

    cout << "\n-----------分割线-----------" << endl;

    // 纵向网格线
    cout << "纵向网格线拟合结果 (x = b0 + b1*y + ...):" << endl;
    for (int j = 0; j < 5; ++j) {
        vector<double> xs, ys;
        for (const auto &pt : col_lines[j]) {
            ys.push_back(pt.y);
            xs.push_back(pt.x);
        }

        vector<double> coeff;
        double r2 = polyfit(ys, xs, degree, coeff);

        cout << "Col line " << j << ": x = ";
        cout << fixed << setprecision(6);
        for (int k = 0; k <= degree; ++k) {
            if (k > 0) cout << " + ";
            cout << coeff[k] << "*y^" << k;
        }
        cout << "    (R²=" << r2 << ")" << endl;

        // 保存拟合曲线
        stringstream fname;
        fname << "col_fit_" << j << ".txt";
        ofstream fout(fname.str());
        int steps = 300;
        for (int k = 0; k <= steps; ++k) {
            double y = canvas_h * k / steps;
            double x = 0;
            for (int p = 0; p <= degree; ++p)
                x += coeff[p] * pow(y, p);
            fout << x << " " << y << "\n";
        }
        fout.close();
    }

    cout << "\n 所有拟合曲线已保存 (degree = " << degree << ")" << endl;
    return 0;
}
