#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;
//이전 프레임에서 현재 프레임으로 얼마나 움직였는지(dx, dy) 저장
struct TransformParam {
    double dx;
    double dy;
    TransformParam() : dx(0), dy(0) {}
    TransformParam(double _dx, double _dy) : dx(_dx), dy(_dy) {}
};
// dx, dy를 계속 더해서, 카메라가 시작점부터 현재까지 어떻게 움직여왔는지(궤적) 저장
struct Trajectory {
    double x;
    double y;
    Trajectory() : x(0), y(0) {}
    Trajectory(double _x, double _y) : x(_x), y(_y) {}
};

int main() {
    VideoCapture cap("sampleVideo2.mp4");
    if (!cap.isOpened()) {
        return -1;
    }

    int w = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    int n_frames = (int)cap.get(CAP_PROP_FRAME_COUNT);

    cout << "Video Info: " << w << "x" << h << ", " << n_frames << " frames" << endl;

    Mat prev, curr, prev_gray, curr_gray;

    cap >> prev; //비디오를 열고 첫 번째 프레임을 읽어 흑백으로 변환
    if (prev.empty()) return -1;

    cvtColor(prev, prev_gray, COLOR_BGR2GRAY);

    vector<TransformParam> transforms;
    vector<Trajectory> trajectory;
    vector<Trajectory> smoothed_trajectory;

    Trajectory current_trajectory(0, 0);
    trajectory.push_back(current_trajectory);

    vector<Point2f> prev_pts, curr_pts;

    goodFeaturesToTrack(prev_gray, prev_pts, 200, 0.01, 30);

    cout << "Step 1: Analyzing Motion..." << endl;

    for (int i = 1; i < n_frames; i++) {
        bool success = cap.read(curr);
        if (!success) break;

        if (i % 10 == 0) {
            cout << "Processing frame: " << i << " / " << n_frames << "\r";
        }

        cvtColor(curr, curr_gray, COLOR_BGR2GRAY);

        if (prev_pts.size() < 50) { //영상에서 추적하기 좋은 점(코너 등) 찾기. 점 개수 50미만 떨어질 시 다시 찾아서 보충.
            goodFeaturesToTrack(prev_gray, prev_pts, 200, 0.01, 30);
        }

        vector<uchar> status;
        vector<float> err;

        if (prev_pts.size() > 0) { //이전 프레임의 특징점(prev_pts)들이 현재 프레임(curr_pts)에서 어디로 이동했는지 계산
            calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts, status, err);
        }

        //모든 점들의 이동 거리 평균 구하기
        double sum_dx = 0.0;
        double sum_dy = 0.0;
        int valid_count = 0;

        vector<Point2f> p_prev, p_curr;
        for (size_t k = 0; k < status.size(); k++) {
            if (status[k]) {
                p_prev.push_back(prev_pts[k]);
                p_curr.push_back(curr_pts[k]);

                sum_dx += (curr_pts[k].x - prev_pts[k].x);
                sum_dy += (curr_pts[k].y - prev_pts[k].y);
                valid_count++;
            }
        }

        double dx = 0.0;
        double dy = 0.0;

        if (valid_count > 0) {
            dx = sum_dx / valid_count;
            dy = sum_dy / valid_count;
        }

        transforms.push_back(TransformParam(dx, dy));
        // 매 프레임의 움직임을 누적하여 저장. 
        current_trajectory.x += dx;
        current_trajectory.y += dy;
        trajectory.push_back(current_trajectory);

        prev = curr.clone();
        prev_gray = curr_gray.clone();
        prev_pts = p_curr;
    }

    int radius = 30; //원도우 크기. 앞 뒤 30프레임 참조.

    for (size_t i = 0; i < trajectory.size(); i++) {
        double sum_x = 0, sum_y = 0;
        int count = 0;

        for (int j = -radius; j <= radius; j++) {
            int idx = (int)i + j;
            if (idx >= 0 && idx < trajectory.size()) {
                sum_x += trajectory[idx].x;
                sum_y += trajectory[idx].y;
                count++;
            }
        }
        // 주변 radius 범위 내의 값들을 다 더해서 평균 구함 (이동 평균 필터)
        smoothed_trajectory.push_back(Trajectory(sum_x / count, sum_y / count));
    }

    cap.set(CAP_PROP_POS_FRAMES, 0);
    Mat frame, stabilized_frame, combined;

    cap >> frame;

    hconcat(frame, frame, combined);
    if (combined.cols > 1920) resize(combined, combined, Size(combined.cols / 2, combined.rows / 2));
    imshow("Original vs Stabilized", combined);
    waitKey(1);

    for (size_t i = 0; i < transforms.size() && i < smoothed_trajectory.size() - 1; i++) {
        cap >> frame;
        if (frame.empty()) break;
        // 보정값 개선하여 이미지를 튀는 픽셀만큼 반대로 밀기
        double diff_x = smoothed_trajectory[i + 1].x - trajectory[i + 1].x;
        double diff_y = smoothed_trajectory[i + 1].y - trajectory[i + 1].y;

        int shift_x = (int)diff_x;
        int shift_y = (int)diff_y;

        stabilized_frame = Mat::zeros(frame.size(), frame.type());

        int src_x = 0;
        int src_y = 0;
        int src_w = w;
        int src_h = h;

        if (shift_x > 0) src_w -= shift_x;
        else             src_x = -shift_x;

        if (shift_y > 0) src_h -= shift_y;
        else             src_y = -shift_y;

        src_w = min(src_w, w - src_x);
        src_h = min(src_h, h - src_y);

        int dst_x = 0;
        int dst_y = 0;

        if (shift_x > 0) dst_x = shift_x;
        else             dst_x = 0;

        if (shift_y > 0) dst_y = shift_y;
        else             dst_y = 0;

        if (src_w > 0 && src_h > 0 && dst_x + src_w <= w && dst_y + src_h <= h) {
            Rect srcRect(src_x, src_y, src_w, src_h);
            Rect dstRect(dst_x, dst_y, src_w, src_h);

            Mat src_roi = frame(srcRect);
            Mat dst_roi = stabilized_frame(dstRect);
            //shift_x, shift_y에 따라 srcRect(원본에서 가져올 영역)와 dstRect(붙여넣을 영역) 계산
            src_roi.copyTo(dst_roi);
        }
        //결과 출력
        hconcat(frame, stabilized_frame, combined);

        if (combined.cols > 1920) {
            resize(combined, combined, Size(combined.cols / 2, combined.rows / 2));
        }

        putText(combined, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        putText(combined, "Stabilized", Point(combined.cols / 2 + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        imshow("Original vs Stabilized", combined);

        if (waitKey(10) == 27) break;
    }

    return 0;
}