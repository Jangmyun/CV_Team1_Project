#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// 프레임 간 카메라 움직임(이동 + 회전)을 담는 구조체
struct TransformParam {
    TransformParam() {}
    TransformParam(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
    }
    double dx;  
    double dy;  
    double da;  

    void getTransform(Mat &T) const {
        T.at<double>(0,0) = cos(da);
        T.at<double>(0,1) = -sin(da);
        T.at<double>(1,0) = sin(da);
        T.at<double>(1,1) = cos(da);
        T.at<double>(0,2) = dx;
        T.at<double>(1,2) = dy;
    }
};

// 누적된 카메라 궤적을 담는 구조체
struct Trajectory {
    Trajectory() {}
    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }
    double x;  
    double y;  
    double a;  
};

// 프레임 간 변환 정보들을 누적해 카메라 궤적을 생성
vector<Trajectory> cumsum(const vector<TransformParam> &transforms) {
    vector<Trajectory> trajectory;
    trajectory.reserve(transforms.size());

    double a = 0.0;
    double x = 0.0;
    double y = 0.0;

    for (size_t i = 0; i < transforms.size(); i++) {
        x += transforms[i].dx;
        y += transforms[i].dy;
        a += transforms[i].da;
        trajectory.push_back(Trajectory(x, y, a));
    }
    return trajectory;
}

// 궤적을 이동 평균으로 부드럽게 만드는 함수
vector<Trajectory> smooth(const vector<Trajectory> &trajectory, int radius) {
    vector<Trajectory> smoothed_trajectory;
    smoothed_trajectory.reserve(trajectory.size());

    int size = static_cast<int>(trajectory.size());

    for (int i = 0; i < size; i++) {
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_a = 0.0;
        int count = 0;

        for (int j = -radius; j <= radius; j++) {
            int idx = i + j;
            if (idx >= 0 && idx < size) {
                sum_x += trajectory[idx].x;
                sum_y += trajectory[idx].y;
                sum_a += trajectory[idx].a;
                count++;
            }
        }

        smoothed_trajectory.push_back(
            Trajectory(sum_x / count, sum_y / count, sum_a / count)
        );
    }
    return smoothed_trajectory;
}

// 외곽의 검은 영역을 줄이기 위한 스케일 보정
void fixBorder(Mat &frame_stabilized) {
    Mat T = getRotationMatrix2D(
        Point2f(static_cast<float>(frame_stabilized.cols) / 2.0f,
                static_cast<float>(frame_stabilized.rows) / 2.0f),
        0.0,
        1.04
    );
    warpAffine(frame_stabilized, frame_stabilized, T, frame_stabilized.size());
}

int main(int argc, char** argv) {
    string inputPath = "sampleVideo3.mp4";
    if (argc > 1) {
        inputPath = argv[1];
    }

    VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open video file: " << inputPath << endl;
        return -1;
    }

    int n_frames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
    int w = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0.0) {
        fps = 30.0;
    }

    cout << "Input video: " << inputPath << endl;
    cout << "Frames: " << n_frames << ", Size: " << w << "x" << h << ", FPS: " << fps << endl;

    string outputPath = "video_out.mp4";
    int fourcc = VideoWriter::fourcc('m','p','4','v');
    VideoWriter out(outputPath, fourcc, fps, Size(2 * w, h));
    if (!out.isOpened()) {
        cerr << "Error: Cannot open output video file for write: " << outputPath << endl;
        return -1;
    }

    Mat curr, curr_gray;
    Mat prev, prev_gray;

    if (!cap.read(prev)) {
        cerr << "Error: Cannot read the first frame from video." << endl;
        return -1;
    }
    cvtColor(prev, prev_gray, COLOR_BGR2GRAY);

    vector<TransformParam> transforms;
    transforms.reserve(max(n_frames - 1, 1));

    Mat last_T = Mat::eye(2, 3, CV_64F);

    for (int i = 1; i < n_frames; i++) {
        vector<Point2f> prev_pts, curr_pts;

        goodFeaturesToTrack(prev_gray, prev_pts, 200, 0.01, 30);

        if (!cap.read(curr)) {
            cout << "End of video reached at frame " << i << endl;
            break;
        }
        cvtColor(curr, curr_gray, COLOR_BGR2GRAY);

        if (prev_pts.empty()) {
            cout << "No features to track at frame " << i << endl;
            prev_gray = curr_gray.clone();
            continue;
        }

        vector<uchar> status;
        vector<float> err;
        calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts, status, err);

        vector<Point2f> prev_pts_filtered, curr_pts_filtered;
        for (size_t k = 0; k < status.size(); k++) {
            if (status[k]) {
                prev_pts_filtered.push_back(prev_pts[k]);
                curr_pts_filtered.push_back(curr_pts[k]);
            }
        }

        if (prev_pts_filtered.size() < 10) {
            cout << "Too few tracked points at frame " << i << endl;
            prev_gray = curr_gray.clone();
            continue;
        }

        Mat T = estimateAffinePartial2D(prev_pts_filtered, curr_pts_filtered);

        if (T.empty()) {
            T = last_T.clone();
        } else {
            T.convertTo(T, CV_64F);
            last_T = T.clone();
        }

        double dx = T.at<double>(0, 2);
        double dy = T.at<double>(1, 2);
        double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));

        transforms.push_back(TransformParam(dx, dy, da));

        prev_gray = curr_gray.clone();

        cout << "Frame: " << i << " / " << n_frames
             << " - Tracked points: " << prev_pts_filtered.size() << endl;
    }

    if (transforms.empty()) {
        cerr << "Error: No transforms estimated. Check input video / feature detection." << endl;
        return -1;
    }

    vector<Trajectory> trajectory = cumsum(transforms);
    vector<Trajectory> smoothed_trajectory = smooth(trajectory, 30);

    vector<TransformParam> transforms_smooth;
    transforms_smooth.reserve(transforms.size());

    for (size_t i = 0; i < transforms.size(); i++) {
        double diff_x = smoothed_trajectory[i].x - trajectory[i].x;
        double diff_y = smoothed_trajectory[i].y - trajectory[i].y;
        double diff_a = smoothed_trajectory[i].a - trajectory[i].a;

        double dx = transforms[i].dx + diff_x;
        double dy = transforms[i].dy + diff_y;
        double da = transforms[i].da + diff_a;

        transforms_smooth.push_back(TransformParam(dx, dy, da));
    }

    cap.set(CAP_PROP_POS_FRAMES, 0);

    Mat T(2, 3, CV_64F);
    Mat frame, frame_stabilized, frame_out;

    namedWindow("Before and After", WINDOW_NORMAL);

    int max_index = static_cast<int>(transforms_smooth.size());

    for (int i = 0; i < max_index; i++) {
        if (!cap.read(frame)) {
            cout << "End of video reached while applying transforms at frame " << i << endl;
            break;
        }

        transforms_smooth[i].getTransform(T);

        /*
        void cv::warpAffine(
            cv::InputArray src,
            cv::OutputArray dst,
            cv::InputArray M,
            cv::Size dsize,
            int flags = cv::INTER_LINEAR,
            int borderMode = cv::BORDER_CONSTANT,
            const cv::Scalar& borderValue = cv::Scalar()
        );
        */
        /*
        This function applies an Affine Transformation—including translation, rotation, 
        and scaling—to the input image using a $2 \times 3$ transformation matrix, 
        generating a warped output image.
        */
        warpAffine(frame, frame_stabilized, T, frame.size());

        /*
        void cv::fixBorder(
            cv::InputOutputArray dst,
            int borderType
        );
        */
       /*
       It is used to correct artifacts that may occur around the image borders 
       when using the warpAffine or warpPerspective functions. 
       Specifically, it compensates for unwanted effects (e.g., errors in edge pixel values) 
       that arise when the borders are not correctly handled after an image transformation.
       */
        fixBorder(frame_stabilized);

        /*
        void cv::hconcat(
            const cv::Mat* src,
            size_t nsrc,
            cv::OutputArray dst
        );
        */
       /*
       The function is used to combine several input images horizontally to create a single larger image. 
       It performs the operation of stitching the images side-by-side along the same row.
       */
        hconcat(frame, frame_stabilized, frame_out);

        imshow("Before and After", frame_out);
        out.write(frame_out);

        char key = static_cast<char>(waitKey(1000/fps));
        if (key == 27) {
            break;
        }
    }

    cout << "Stabilized video saved to: " << outputPath << endl;
    return 0;
}
