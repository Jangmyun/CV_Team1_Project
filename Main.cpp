#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// -----------------------------
// Step 3: Motion Estimation Struct
// -----------------------------
struct TransformParam {
    TransformParam() {}
    TransformParam(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
    }
    double dx;
    double dy;
    double da; // angle (radians)

    void getTransform(Mat &T) const {
        // Reconstruct transformation matrix from dx, dy, da
        T.at<double>(0,0) = cos(da);
        T.at<double>(0,1) = -sin(da);
        T.at<double>(1,0) = sin(da);
        T.at<double>(1,1) = cos(da);
        T.at<double>(0,2) = dx;
        T.at<double>(1,2) = dy;
    }
};

// -----------------------------
// Step 4: Trajectory Struct
// -----------------------------
struct Trajectory {
    Trajectory() {}
    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }
    double x;
    double y;
    double a; // angle
};

// -----------------------------
// Step 4.1: Cumulative Sum Function
// -----------------------------
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

// -----------------------------
// Step 4.2: Smooth Trajectory (Moving Average)
// -----------------------------
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

// -----------------------------
// Fix border by scaling slightly
// -----------------------------
void fixBorder(Mat &frame_stabilized) {
    Mat T = getRotationMatrix2D(
        Point2f(static_cast<float>(frame_stabilized.cols) / 2.0f,
                static_cast<float>(frame_stabilized.rows) / 2.0f),
        0.0,
        1.04 // scale
    );
    warpAffine(frame_stabilized, frame_stabilized, T, frame_stabilized.size());
}

// -----------------------------
// main()
// -----------------------------
int main(int argc, char** argv) {
    // Input video path (argv[1] 우선, 없으면 기본 파일명)
    string inputPath = "sampleVideo1.mp4";
    if (argc > 1) {
        inputPath = argv[1];
    }

    // Step 1: Open input video
    VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open video file: " << inputPath << endl;
        return -1;
    }

    // Get frame count (주의: 일부 코덱은 frame count가 정확하지 않을 수 있음)
    int n_frames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));

    // Get width and height of video stream
    int w = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

    // Get FPS
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0.0) {
        // macOS에서 FPS 못 읽어오면 기본값 세팅
        fps = 30.0;
    }

    cout << "Input video: " << inputPath << endl;
    cout << "Frames: " << n_frames << ", Size: " << w << "x" << h << ", FPS: " << fps << endl;

    // Step 1.1: Set up output video
    string outputPath = "video_out.mp4";
    int fourcc = VideoWriter::fourcc('m','p','4','v');
    VideoWriter out(outputPath, fourcc, fps, Size(2 * w, h));
    if (!out.isOpened()) {
        cerr << "Error: Cannot open output video file for write: " << outputPath << endl;
        return -1;
    }

    // Step 2: Read the first frame and convert to grayscale
    Mat curr, curr_gray;
    Mat prev, prev_gray;

    if (!cap.read(prev)) {
        cerr << "Error: Cannot read the first frame from video." << endl;
        return -1;
    }
    cvtColor(prev, prev_gray, COLOR_BGR2GRAY);

    // Step 3: Motion estimation between frames
    vector<TransformParam> transforms;
    transforms.reserve(max(n_frames - 1, 1)); // rough reserve

    // Initialize last_T as identity (2x3)
    Mat last_T = Mat::eye(2, 3, CV_64F);

    // Optical flow variables
    for (int i = 1; i < n_frames; i++) {
        vector<Point2f> prev_pts, curr_pts;

        // Detect good features to track in previous frame
        goodFeaturesToTrack(prev_gray, prev_pts, 200, 0.01, 30);

        // Read next frame
        if (!cap.read(curr)) {
            cout << "End of video reached at frame " << i << endl;
            break;
        }
        cvtColor(curr, curr_gray, COLOR_BGR2GRAY);

        // If no points were found in previous frame, skip
        if (prev_pts.empty()) {
            cout << "No features to track at frame " << i << endl;
            prev_gray = curr_gray.clone();
            continue;
        }

        // Calculate optical flow
        vector<uchar> status;
        vector<float> err;
        calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts, status, err);

        // Filter only valid points
        vector<Point2f> prev_pts_filtered, curr_pts_filtered;
        for (size_t k = 0; k < status.size(); k++) {
            if (status[k]) {
                prev_pts_filtered.push_back(prev_pts[k]);
                curr_pts_filtered.push_back(curr_pts[k]);
            }
        }

        if (prev_pts_filtered.size() < 10) {
            // 너무 적게 추적되면 transform 추정이 불안정해짐
            cout << "Too few tracked points at frame " << i << endl;
            prev_gray = curr_gray.clone();
            continue;
        }

        // Estimate 2D affine transform with partial constraints (rotation + translation + possible scale)
        Mat T = estimateAffinePartial2D(prev_pts_filtered, curr_pts_filtered);

        // Handle failure of transform estimation
        if (T.empty()) {
            // use last good transform
            T = last_T.clone();
        } else {
            // ensure type is CV_64F
            T.convertTo(T, CV_64F);
            last_T = T.clone();
        }

        // Extract translation
        double dx = T.at<double>(0, 2);
        double dy = T.at<double>(1, 2);

        // Extract rotation (assuming no shear / minimal scale diff)
        double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));

        transforms.push_back(TransformParam(dx, dy, da));

        // Move to next frame
        prev_gray = curr_gray.clone();

        cout << "Frame: " << i << " / " << n_frames
             << " - Tracked points: " << prev_pts_filtered.size() << endl;
    }

    if (transforms.empty()) {
        cerr << "Error: No transforms estimated. Check input video / feature detection." << endl;
        return -1;
    }

    // Step 4: Calculate smooth motion between frames
    vector<Trajectory> trajectory = cumsum(transforms);
    vector<Trajectory> smoothed_trajectory = smooth(trajectory, 30); // radius 30

    // Step 4.3: Generate smoothed transforms
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

    // Step 5: Apply smoothed camera motion to frames
    // Reset capture to first frame
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

        // Apply affine warp to the frame
        warpAffine(frame, frame_stabilized, T, frame.size());

        // Fix border artifacts
        fixBorder(frame_stabilized);

        // Concatenate original and stabilized frames side by side
        hconcat(frame, frame_stabilized, frame_out);

        // Optionally resize if too wide
        if (frame_out.cols > 1920) {
            resize(frame_out, frame_out,
                   Size(frame_out.cols / 2, frame_out.rows / 2));
        }

        imshow("Before and After", frame_out);
        out.write(frame_out);

        // Press Esc to quit early
        char key = static_cast<char>(waitKey(10));
        if (key == 27) { // ESC
            break;
        }
    }

    cout << "Stabilized video saved to: " << outputPath << endl;
    return 0;
}