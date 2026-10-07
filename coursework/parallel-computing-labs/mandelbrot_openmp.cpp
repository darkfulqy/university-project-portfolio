#include <omp.h>
#include <windows.h>
#include <gdiplus.h>

#include <iostream>

using namespace std;
using namespace Gdiplus;

int GetEncoderClsid(const WCHAR* format, CLSID* pClsid) {
    UINT num = 0;
    UINT size = 0;
    GetImageEncodersSize(&num, &size);
    if (size == 0) {
        return -1;
    }

    ImageCodecInfo* pImageCodecInfo = (ImageCodecInfo*)(malloc(size));
    if (pImageCodecInfo == NULL) {
        return -1;
    }

    GetImageEncoders(num, size, pImageCodecInfo);
    for (UINT j = 0; j < num; ++j) {
        if (wcscmp(pImageCodecInfo[j].MimeType, format) == 0) {
            *pClsid = pImageCodecInfo[j].Clsid;
            free(pImageCodecInfo);
            return j;
        }
    }

    free(pImageCodecInfo);
    return -1;
}

int main() {
    int width, height, maxIter, threadCount;

    cout << "Input image width: ";
    cin >> width;
    cout << "Input image height: ";
    cin >> height;
    cout << "Input max iterations: ";
    cin >> maxIter;
    cout << "Input thread count: ";
    cin >> threadCount;

    int** image = new int*[height];
    for (int i = 0; i < height; i++) {
        image[i] = new int[width];
    }

    omp_set_num_threads(threadCount);

    double start = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double cx = (double)x / width * 3.5 - 2.5;
            double cy = (double)y / height * 2.0 - 1.0;

            double zx = 0.0;
            double zy = 0.0;
            int iter = 0;

            while (zx * zx + zy * zy <= 4.0 && iter < maxIter) {
                double temp = zx * zx - zy * zy + cx;
                zy = 2.0 * zx * zy + cy;
                zx = temp;
                iter++;
            }

            image[y][x] = iter;
        }
    }

    double end = omp_get_wtime();

    long long totalIter = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            totalIter += image[y][x];
        }
    }

    GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR gdiplusToken;
    GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

    Bitmap bitmap(width, height, PixelFormat24bppRGB);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int color = image[y][x] * 255 / maxIter;
            bitmap.SetPixel(x, y, Color(color, color, color));
        }
    }

    CLSID pngClsid;
    if (GetEncoderClsid(L"image/png", &pngClsid) != -1) {
        bitmap.Save(L"output\\mandelbrot.png", &pngClsid, NULL);
    }

    GdiplusShutdown(gdiplusToken);

    cout << endl;
    cout << "Done" << endl;
    cout << "Threads = " << threadCount << endl;
    cout << "Time = " << end - start << " s" << endl;
    cout << "Total iterations = " << totalIter << endl;
    cout << "Image saved to output\\mandelbrot.png" << endl;

    for (int i = 0; i < height; i++) {
        delete[] image[i];
    }
    delete[] image;

    return 0;
}
