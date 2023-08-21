#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define BLOCK_SIZE 15
#define OFFSET (BLOCK_SIZE/2)
#define DISPARITY_RANGE 64

unsigned char left_window[BLOCK_SIZE][BLOCK_SIZE] = { 0 };
unsigned char right_window[BLOCK_SIZE][BLOCK_SIZE] = { 0 };

Mat leftImage = imread("C:\\Users\\ben\\Desktop\\hm2.png", IMREAD_GRAYSCALE);
Mat rightImage = imread("C:\\Users\\ben\\Desktop\\hm1.png", IMREAD_GRAYSCALE);

int calculate_ssd_of_blocks() {
    int diff = 0;
    int ssd = 0;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            diff = left_window[i][j] - right_window[i][j];
            ssd += diff * diff;
        }
    }
    return ssd;
}
void load_left_window(int row, int col,Mat img) {
    for (int x = 0; x < BLOCK_SIZE; x++) {
        for (int y = 0; y < BLOCK_SIZE; y++) {
            left_window[x][y] = img.at<uchar>(row + x - OFFSET, col + y - OFFSET);
        }
    }
}

void load_right_window(int row, int col, Mat img) {
    for (int x = 0; x < BLOCK_SIZE; x++) {
        for (int y = 0; y < BLOCK_SIZE; y++) {
            right_window[x][y] = img.at<uchar>(row + x - OFFSET, col + y - OFFSET);
        }
    }
}

int main() {
    Mat depthMap = leftImage.clone();
    int cols = leftImage.cols;
    int rows = leftImage.rows;


    for (int i = OFFSET; i < rows - OFFSET; i++) {//search all the pixels except borders 
        for (int j = OFFSET; j < cols - OFFSET; j++) {
            int best_ssd = 255 * 255 * BLOCK_SIZE * BLOCK_SIZE;
            int match = 0;

            load_left_window(i, j, leftImage);

            for (int disp = 0; disp < DISPARITY_RANGE; disp++) { //compare left block with right block
                int ssd = 0;
                if (j + disp < cols - OFFSET) {
                    load_right_window(i, j + disp, rightImage);
                    ssd = calculate_ssd_of_blocks();
                    if (ssd < best_ssd) {
                        best_ssd = ssd;
                        match = disp;
                    }
                }
            }
            depthMap.at<uchar>(i,j) = static_cast<uchar>(match);
        }
    }

    Mat croppedDepthMap(depthMap(Range(OFFSET, rows - OFFSET), Range(OFFSET, cols - OFFSET)));
    Mat normalizedDepthMap;
    normalize(croppedDepthMap, normalizedDepthMap, 255, 0, cv::NORM_MINMAX);
    normalizedDepthMap.convertTo(normalizedDepthMap, CV_8UC1);
    imshow("Normalize Derinlik Haritasi", normalizedDepthMap);
    imshow("Derinlik Haritasi", croppedDepthMap);
    imwrite("test.jpg", normalizedDepthMap);
    waitKey(0);

    return 0;
}