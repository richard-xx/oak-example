#include <chrono>
#include <iomanip>
#include <iostream>
// Includes common necessary includes for development using depthai library
#include "depthai/depthai.hpp"
#include <cmrc/cmrc.hpp>
CMRC_DECLARE(models);
/*
The code is the same as for Tiny-yolo-V3, the only difference is the blob file.
The blob was compiled following this tutorial:
https://github.com/TNTWEN/OpenVINO-YOLOV4
*/

static const std::vector<std::string> labelMap = {
    "person",        "bicycle",       "car",           "motorbike",
    "aeroplane",     "bus",           "train",         "truck",
    "boat",          "traffic light", "fire hydrant",  "stop sign",
    "parking meter", "bench",         "bird",          "cat",
    "dog",           "horse",         "sheep",         "cow",
    "elephant",      "bear",          "zebra",         "giraffe",
    "backpack",      "umbrella",      "handbag",       "tie",
    "suitcase",      "frisbee",       "skis",          "snowboard",
    "sports ball",   "kite",          "baseball bat",  "baseball glove",
    "skateboard",    "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",           "fork",          "knife",
    "spoon",         "bowl",          "banana",        "apple",
    "sandwich",      "orange",        "broccoli",      "carrot",
    "hot dog",       "pizza",         "donut",         "cake",
    "chair",         "sofa",          "pottedplant",   "bed",
    "diningtable",   "toilet",        "tvmonitor",     "laptop",
    "mouse",         "remote",        "keyboard",      "cell phone",
    "microwave",     "oven",          "toaster",       "sink",
    "refrigerator",  "book",          "clock",         "vase",
    "scissors",      "teddy bear",    "hair drier",    "toothbrush"};

dai::Pipeline createPipeline(std::vector<uint8_t> &nnData) {

  uint8_t NumClasses = 80;

  dai::OpenVINO::Blob model(nnData);
  auto input_dim = model.networkInputs.begin()->second.dims;
  int nnWidth = (int)input_dim.at(0);
  int nnHeight = (int)input_dim.at(1);

  auto first_output = model.networkOutputs.begin();
  std::string output_name = first_output->first;
  auto output_dim = first_output->second.dims.at(2);
  if (output_name.find("yolov6") == -1) {
    NumClasses = output_dim - 5;
  } else {
    NumClasses = int(output_dim / 3) - 5;
  }

  // Create pipeline
  dai::Pipeline pipeline;

  // Define sources and outputs
  auto monoLeft = pipeline.create<dai::node::MonoCamera>();
  auto monoRight = pipeline.create<dai::node::MonoCamera>();
  auto spatialDetectionNetwork =
      pipeline.create<dai::node::YoloSpatialDetectionNetwork>();
  auto stereo = pipeline.create<dai::node::StereoDepth>();
  auto imageManip = pipeline.create<dai::node::ImageManip>();

  auto xoutImg = pipeline.create<dai::node::XLinkOut>();
  auto xoutNN = pipeline.create<dai::node::XLinkOut>();
  auto xoutDepth = pipeline.create<dai::node::XLinkOut>();

  xoutImg->setStreamName("image");
  xoutNN->setStreamName("detections");
  xoutDepth->setStreamName("depth");

  // Properties
  monoLeft->setResolution(
      dai::MonoCameraProperties::SensorResolution::THE_800_P);
  monoLeft->setBoardSocket(dai::CameraBoardSocket::LEFT);
  monoRight->setResolution(
      dai::MonoCameraProperties::SensorResolution::THE_800_P);
  monoRight->setBoardSocket(dai::CameraBoardSocket::RIGHT);

  imageManip->initialConfig.setResize(nnWidth, nnHeight);
  // The NN model expects BGR input. By default ImageManip output type would be
  // same as input (gray in this case)
  imageManip->initialConfig.setFrameType(dai::ImgFrame::Type::BGR888p);
  imageManip->setMaxOutputFrameSize(nnWidth * nnHeight * 3);

  // setting node configs
  stereo->setDefaultProfilePreset(
      dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
  // Align depth map to the perspective of RGB camera, on which inference is
  // done
  stereo->setDepthAlign(
      dai::RawStereoDepthConfig::AlgorithmControl::DepthAlign::RECTIFIED_LEFT);
  stereo->setOutputSize(monoLeft->getResolutionWidth(),
                        monoLeft->getResolutionHeight());

  // Network specific settings
  spatialDetectionNetwork->setBlob(model);
  spatialDetectionNetwork->setConfidenceThreshold(0.5f);

  // Yolo specific settings
  spatialDetectionNetwork->setNumClasses(NumClasses);
  spatialDetectionNetwork->setCoordinateSize(4);
  spatialDetectionNetwork->setAnchors({});
  spatialDetectionNetwork->setAnchorMasks({});
  spatialDetectionNetwork->setIouThreshold(0.5f);

  // spatial specific parameters
  spatialDetectionNetwork->setBoundingBoxScaleFactor(1);
  spatialDetectionNetwork->setDepthLowerThreshold(0);
  spatialDetectionNetwork->setDepthUpperThreshold(50000);

  // Linking
  monoLeft->out.link(stereo->left);
  monoRight->out.link(stereo->right);

  stereo->rectifiedLeft.link(imageManip->inputImage);
  stereo->depth.link(spatialDetectionNetwork->inputDepth);

  imageManip->out.link(xoutImg->input);
  imageManip->out.link(spatialDetectionNetwork->input);

  spatialDetectionNetwork->out.link(xoutNN->input);
  spatialDetectionNetwork->passthroughDepth.link(xoutDepth->input);

  return pipeline;
}

int main(int argc, char **argv) {
  using namespace std::chrono;
  std::string nnPath;
  std::vector<uint8_t> blobData;

  // If path to blob specified, use that
  if (argc > 1) {
    nnPath = std::string(argv[1]);
  }

  // Print which blob we are using
  printf("Using blob at path: %s\n", nnPath.c_str());

  // create model blob
  if (nnPath.empty()) {
    auto fs1 = cmrc::models::get_filesystem();
    auto nnPathCMRC = fs1.open("yolov6n.blob");
    blobData = std::vector<uint8_t>(nnPathCMRC.begin(), nnPathCMRC.end());
  } else {
    std::ifstream stream(nnPath, std::ios::in | std::ios::binary);
    if (!stream.is_open()) {
      throw std::runtime_error("Cannot load blob, file at path " + nnPath +
                               " doesn't exist.");
    }
    blobData = std::vector<std::uint8_t>(std::istreambuf_iterator<char>(stream),
                                         std::istreambuf_iterator<char>());
  }

  // Connect to device and start pipeline
  dai::Device device(createPipeline(blobData));

  // Output queues will be used to get the rgb frames and nn data from the
  // outputs defined above
  auto imageQueue = device.getOutputQueue("image", 4, false);
  auto detectQueue = device.getOutputQueue("detections", 4, false);
  auto depthQueue = device.getOutputQueue("depth", 4, false);

  cv::Mat frame;
  cv::Mat depthFrameColor;

  std::vector<dai::SpatialImgDetection> detections;
  auto startTime = steady_clock::now();
  float counter = 0;
  float fps = 0;
  auto color2 = cv::Scalar(255, 255, 255);

  // Add bounding boxes and text to the frame and show it to the user
  auto displayFrame = [](const std::string &name, const cv::Mat &frame,
                         const cv::Mat &depthFrameColor,
                         std::vector<dai::SpatialImgDetection> &detections) {
    auto color = cv::Scalar(255, 0, 0);

    // nn data, being the bounding box locations, are in <0..1> range - they
    // need to be normalized with frame width/height
    for (auto &detection : detections) {

      auto roiData = detection.boundingBoxMapping;
      auto roi = roiData.roi;
      roi = roi.denormalize(depthFrameColor.cols, depthFrameColor.rows);
      auto topLeft = roi.topLeft();
      auto bottomRight = roi.bottomRight();
      auto xmin = (int)topLeft.x;
      auto ymin = (int)topLeft.y;
      auto xmax = (int)bottomRight.x;
      auto ymax = (int)bottomRight.y;
      cv::rectangle(depthFrameColor,
                    cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)),
                    color, cv::FONT_HERSHEY_SIMPLEX);

      auto x1 = int(detection.xmin * frame.cols);
      auto y1 = int(detection.ymin * frame.rows);
      auto x2 = int(detection.xmax * frame.cols);
      auto y2 = int(detection.ymax * frame.rows);

      uint32_t labelIndex = detection.label;
      std::string labelStr = std::to_string(labelIndex);
      if (labelIndex < labelMap.size()) {
        labelStr = labelMap[labelIndex];
      }
      cv::putText(frame, labelStr, cv::Point(x1 + 10, y1 + 20),
                  cv::FONT_HERSHEY_TRIPLEX, 0.5, 255);
      std::stringstream confStr;
      confStr << std::fixed << std::setprecision(2)
              << detection.confidence * 100;
      cv::putText(frame, confStr.str(), cv::Point(x1 + 10, y1 + 40),
                  cv::FONT_HERSHEY_TRIPLEX, 0.5, 255);
      cv::rectangle(frame, cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)),
                    color, cv::FONT_HERSHEY_SIMPLEX);

      std::stringstream depthX;
      depthX << "X: " << (int)detection.spatialCoordinates.x << " mm";
      cv::putText(frame, depthX.str(), cv::Point(x1 + 10, y1 + 50),
                  cv::FONT_HERSHEY_TRIPLEX, 0.5, 255);
      std::stringstream depthY;
      depthY << "Y: " << (int)detection.spatialCoordinates.y << " mm";
      cv::putText(frame, depthY.str(), cv::Point(x1 + 10, y1 + 65),
                  cv::FONT_HERSHEY_TRIPLEX, 0.5, 255);
      std::stringstream depthZ;
      depthZ << "Z: " << (int)detection.spatialCoordinates.z << " mm";
      cv::putText(frame, depthZ.str(), cv::Point(x1 + 10, y1 + 80),
                  cv::FONT_HERSHEY_TRIPLEX, 0.5, 255);
    }

    // Show the frame
    cv::imshow(name, frame);
    cv::imshow("depth", depthFrameColor);
  };

  while (true) {
    std::shared_ptr<dai::ImgFrame> imageQueueData;
    std::shared_ptr<dai::ImgFrame> depthQueueData;
    std::shared_ptr<dai::SpatialImgDetections> detectQueueData;

    imageQueueData = imageQueue->get<dai::ImgFrame>();
    depthQueueData = depthQueue->get<dai::ImgFrame>();
    detectQueueData = detectQueue->get<dai::SpatialImgDetections>();

    if (imageQueueData) {
      frame = imageQueueData->getCvFrame();

      cv::normalize(depthQueueData->getFrame(), depthFrameColor, 255, 0,
                    cv::NORM_INF, CV_8UC1);
      cv::equalizeHist(depthFrameColor, depthFrameColor);
      cv::applyColorMap(depthFrameColor, depthFrameColor, cv::COLORMAP_HOT);

      std::stringstream fpsStr;
      fpsStr << "NN fps: " << std::fixed << std::setprecision(2) << fps;
      cv::putText(frame, fpsStr.str(),
                  cv::Point(2, (int)imageQueueData->getHeight() - 4),
                  cv::FONT_HERSHEY_TRIPLEX, 0.4, color2);
    }

    if (detectQueueData) {
      counter++;
      auto currentTime = steady_clock::now();
      auto elapsed = duration_cast<duration<float>>(currentTime - startTime);
      if (elapsed > seconds(1)) {
        fps = counter / elapsed.count();
        counter = 0;
        startTime = currentTime;
      }
      detections = detectQueueData->detections;
    }

    if (!frame.empty()) {
      displayFrame("image", frame, depthFrameColor, detections);
    }

    int key = cv::waitKey(1);
    if (key == 'q' || key == 'Q') {
      return 0;
    }
  }
  return 0;
}