#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/videostab.hpp>

using namespace std;
using namespace cv;
using namespace cv::videostab;

int main(int argc, char** argv) {
  if (argc < 2) {
    cout << "Usage: " << argv[0] << " <input_video> [output_video]" << endl;
    return 1;
  }

  string inputPath = argv[1];
  string outputPath = (argc >= 3) ? argv[2] : "stabilized_video.avi";

  try {
    Ptr<VideoFileSource> src = makePtr<VideoFileSource>(inputPath);

    OnePassStabilizer* stabilizer = new OnePassStabilizer();

    Ptr<MotionEstimatorRansacL2> motionEstimator =
        makePtr<MotionEstimatorRansacL2>(MM_AFFINE);

    Ptr<KeypointBasedMotionEstimator> kBasedEst =
        makePtr<KeypointBasedMotionEstimator>(motionEstimator);

    kBasedEst->setDetector(GFTTDetector::create(1000));

    stabilizer->setMotionEstimator(kBasedEst);

    stabilizer->setMotionFilter(makePtr<GaussianMotionFilter>(15));

    stabilizer->setFrameSource(src);
    // sliding window: window를 이동시키며 window 내에서 obj detection
    stabilizer->setRadius(15);
    stabilizer->setTrimRatio(0.1);  // 경계 trim 10%
    stabilizer->setBorderMode(BORDER_REPLICATE);

    // 설정이 끝났으므로 Frame Source 용도로만 사용하기 위한 type casting
    Ptr<IFrameSource> stabilizedFrames(dynamic_cast<IFrameSource*>(stabilizer));

    VideoWriter videoWriter;
    Mat frame;
    int frameCount = 0;

    while (!(frame = stabilizedFrames->nextFrame()).empty()) {
      frameCount++;
      cout << frameCount << '\n';

      if (!videoWriter.isOpened()) {
        videoWriter.open(outputPath, VideoWriter::fourcc('X', 'V', 'I', 'D'),
                         src->fps(), frame.size());

        if (!videoWriter.isOpened()) {  // VideoWriter open 실패
          cerr << "Video writer open failed" << endl;
          return 1;
        }
      }

      videoWriter << frame;

      imshow("Output", frame);
      if (waitKey(33) == 27) break;
    }

    cout << "Total frames: " << frameCount << endl;
    cout << "Saved " << outputPath << endl;

    videoWriter.release();
    stabilizedFrames.release();
    destroyAllWindows();
  } catch (const exception e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }

  return 0;
}