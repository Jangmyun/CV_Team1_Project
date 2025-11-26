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
  }

  return 0;
}