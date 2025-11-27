#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>

using namespace std;
using namespace cv;

// 이동 경로를 저장할 구조체
struct TransformParam {
    double dx;
    double dy;
};

int main(int argc, char **argv)
{
    if (argc != 2) {
        cout << "Usage: ./CVProject <video_file_path>" << endl;
        return -1;
    }

    VideoCapture cap;
    if (cap.open(argv[1]) == 0) {
        cout << "no such file!" << endl;
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    int delay = 1000/fps;

    Mat frame, prev_gray, curr_gray;
    Mat stabilized_frame;

    vector<Point2f> prevPoints, currPoints;
    
    // 누적 경로(Trajectory) 변수
    double acc_x = 0.0;
    double acc_y = 0.0;
    
    // 부드러운 경로 생성을 위한 버퍼 (최근 30프레임 저장)
    const int SMOOTHING_RADIUS = 30; 
    deque<TransformParam> trajectory_buffer;
    
    // 특징점 검출 파라미터
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    int maxCorners = 200; 
    double k = 0.04;
    
    Size winSize(21, 21); // Optical Flow 윈도우 크기

    bool initialization = true;

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, curr_gray, COLOR_BGR2GRAY);

        if (initialization)
        {
            goodFeaturesToTrack(curr_gray, prevPoints, maxCorners, qualityLevel, minDistance, Mat(), blockSize, false, k);
            stabilized_frame = frame.clone();
            curr_gray.copyTo(prev_gray);
            initialization = false;
        }
        else
        {
            vector<uchar> status;
            vector<float> err;
            
            // Optical Flow 계산 
            if (prevPoints.size() > 0) {
                calcOpticalFlowPyrLK(prev_gray, curr_gray, prevPoints, currPoints, status, err, winSize, 3);
            }

            // 프레임 간 이동량 계산 (평균)
            double sum_dx = 0.0;
            double sum_dy = 0.0;
            int count = 0;
            vector<Point2f> valid_curr;

            for (size_t i = 0; i < status.size(); i++) {
                if (status[i]) {
                    double dx = currPoints[i].x - prevPoints[i].x;
                    double dy = currPoints[i].y - prevPoints[i].y;
                    
                    // 노이즈 제거
                    if (abs(dx) < 50 && abs(dy) < 50) {
                        sum_dx += dx;
                        sum_dy += dy;
                        count++;
                        valid_curr.push_back(currPoints[i]);
                    }
                }
            }

            double dx = 0.0;
            double dy = 0.0;
            
            if (count > 0) {
                dx = sum_dx / count;
                dy = sum_dy / count;
                prevPoints = valid_curr;
            } else {
                goodFeaturesToTrack(curr_gray, prevPoints, maxCorners, qualityLevel, minDistance, Mat(), blockSize, false, k);
            }

            // 경로 누적
            acc_x += dx;
            acc_y += dy;

            trajectory_buffer.push_back({acc_x, acc_y});
            if(trajectory_buffer.size() > SMOOTHING_RADIUS) {
                trajectory_buffer.pop_front();
            }

            // 경로 계산
            double smooth_x = 0.0;
            double smooth_y = 0.0;
            
            for(TransformParam t : trajectory_buffer) {
                smooth_x += t.dx;
                smooth_y += t.dy;
            }
            
            if (!trajectory_buffer.empty()) {
                smooth_x /= trajectory_buffer.size();
                smooth_y /= trajectory_buffer.size();
            }

            // 보정값 적용 (Smoothed - Actual)
            double diff_x = smooth_x - acc_x;
            double diff_y = smooth_y - acc_y;

            Mat M = Mat::eye(3, 3, CV_64F); 
            M.at<double>(0, 2) = diff_x;
            M.at<double>(1, 2) = diff_y;

            warpPerspective(frame, stabilized_frame, M, frame.size());
        }

        curr_gray.copyTo(prev_gray);

        // 결과 비교 화면 생성
        Mat comparison;
        hconcat(frame, stabilized_frame, comparison);
        
        putText(comparison, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        putText(comparison, "Stabilized", Point(frame.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        imshow("Video Stabilization Project", comparison);

        if (waitKey(delay) == 27) break;
    }

    return 0;
}
