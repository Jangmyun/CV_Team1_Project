#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
  if (argc < 2) {
    cout << "Usage: " << argv[0] << " <input_video> [output_video]" << endl;
    return 1;
  }

  string inputPath = argv[1];
  string outputPath = (argc >= 3) ? argv[2] : "stabilized_video.avi";

  return 0;
}