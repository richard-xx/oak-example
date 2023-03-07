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
  auto nnWidth = input_dim.at(0);
  auto nnHeight = input_dim.at(1);

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
  auto detectionNetwork = pipeline.create<dai::node::YoloDetectionNetwork>();
  auto imageManip = pipeline.create<dai::node::ImageManip>();
  auto xoutImg = pipeline.create<dai::node::XLinkOut>();
  auto xoutNN = pipeline.create<dai::node::XLinkOut>();

  xoutImg->setStreamName("image");
  xoutNN->setStreamName("detections");

  // Properties
  monoLeft->setResolution(
      dai::MonoCameraProperties::SensorResolution::THE_800_P);
  monoLeft->setBoardSocket(dai::CameraBoardSocket::LEFT);
  monoLeft->setFps(120);

  imageManip->initialConfig.setResize(nnWidth, nnHeight);
  // The NN model expects BGR input. By default ImageManip output type would be
  // same as input (gray in this case)
  imageManip->initialConfig.setFrameType(dai::ImgFrame::Type::BGR888p);
  imageManip->setMaxOutputFrameSize(nnWidth * nnHeight * 3);

  // Network specific settings
  detectionNetwork->setBlob(model);
  detectionNetwork->setConfidenceThreshold(0.5f);

  // Yolo specific settings
  detectionNetwork->setNumClasses(NumClasses);
  detectionNetwork->setCoordinateSize(4);
  detectionNetwork->setAnchors({});
  detectionNetwork->setAnchorMasks({});
  detectionNetwork->setIouThreshold(0.5f);

  // Linking
  monoLeft->out.link(imageManip->inputImage);
  imageManip->out.link(xoutImg->input);
  imageManip->out.link(detectionNetwork->input);
  detectionNetwork->out.link(xoutNN->input);
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

  cv::Mat frame;
  std::vector<dai::ImgDetection> detections;
  auto startTime = steady_clock::now();
  int counter = 0;
  float fps = 0;
  auto color2 = cv::Scalar(255, 255, 255);

  // Add bounding boxes and text to the frame and show it to the user
  auto displayFrame = [](const std::string &name, const cv::Mat &frame,
                         std::vector<dai::ImgDetection> &detections) {
    auto color = cv::Scalar(255, 0, 0);
    // nn data, being the bounding box locations, are in <0..1> range - they
    // need to be normalized with frame width/height
    for (auto &detection : detections) {
      int x1 = detection.xmin * frame.cols;
      int y1 = detection.ymin * frame.rows;
      int x2 = detection.xmax * frame.cols;
      int y2 = detection.ymax * frame.rows;

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
    }

    // Show the frame
    cv::imshow(name, frame);
  };

  while (true) {
    std::shared_ptr<dai::ImgFrame> imageQueueData;
    std::shared_ptr<dai::ImgDetections> detectQueueData;

    imageQueueData = imageQueue->tryGet<dai::ImgFrame>();
    detectQueueData = detectQueue->tryGet<dai::ImgDetections>();

    if (imageQueueData) {

      frame = imageQueueData->getCvFrame();
      std::stringstream fpsStr;
      fpsStr << "NN fps: " << std::fixed << std::setprecision(2) << fps;
      cv::putText(frame, fpsStr.str(),
                  cv::Point(2, imageQueueData->getHeight() - 4),
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
      displayFrame("image", frame, detections);
    }

    int key = cv::waitKey(1);
    if (key == 'q' || key == 'Q') {
      return 0;
    }
  }
  return 0;
}