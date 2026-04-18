/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

#include <atomic>
#include <chrono>
#include <fins/node.hpp>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>
#include <onnxruntime_cxx_api.h>

class YOLOHandPose : public fins::Node {
public:
  void define() override {
    set_name("YOLOHandPose");
    set_description("YOLO hand pose detection node.");
    set_category("Vision>AI");

    register_input<cv::Mat>("image", &YOLOHandPose::on_image);
    register_output<cv::Mat>("image");
    register_parameter<std::string>("model_path", &YOLOHandPose::update_model_path, "model/yolov8n-pose.onnx");
  }

  void initialize() override {
    std::lock_guard<std::mutex> lock(mutex_);
    init_session();
  }

  void run() override { }
  void pause() override { }
  void reset() override { }

  void on_image(const cv::Mat &input_img, fins::AcqTime acq_time) {
    if (input_img.empty()) return;

    cv::Mat result_img;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!session_) {
        send("image", input_img, acq_time);
        return;
      }
      result_img = inference(input_img);
    }
    send("image", result_img, acq_time);
  }

  void update_model_path(const std::string &model_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_path_ = model_path;
    init_session();
  }

private:
  void init_session() {
    try {
      session_.reset();
      env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLOHandPose");
      Ort::SessionOptions session_options;
      
      session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
      session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

      try {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        logger->info("YOLOHandPose: Using CUDA execution provider");
      } catch (...) {
        logger->warn("YOLOHandPose: CUDA failed, using CPU");
      }

      session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), session_options);
      logger->info("YOLOHandPose: Loaded model from {}", model_path_);
    } catch (const std::exception& e) {
      logger->error("Failed to load handpose model: {}", e.what());
      session_.reset();
    }
  }

  cv::Mat inference(const cv::Mat& inputImg) {
    cv::Mat resized;
    cv::Size targetSize(640, 640);
    cv::resize(inputImg, resized, targetSize);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    cv::Mat blob = cv::dnn::blobFromImage(resized);

    std::vector<int64_t> inputShape = {1, 3, 640, 640};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, (float*)blob.data, blob.total(), inputShape.data(), inputShape.size());

    const char* inputNames[] = {"images"};
    const char* outputNames[] = {"output0"};

    auto outputs = session_->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);
    float* rawOutput = outputs[0].GetTensorMutableData<float>();
    auto outputShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    // YOLOv8-pose output: [1, 56, 8400] (56 = 4 box + 1 score + 17*3 keypoints)
    int dimensions = outputShape[1]; 
    int num_anchors = outputShape[2]; 
    
    cv::Mat outputMat(dimensions, num_anchors, CV_32F, rawOutput);
    outputMat = outputMat.t(); 

    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<cv::Point>> keypoints_list;

    float ratio_x = (float)inputImg.cols / 640.0f;
    float ratio_y = (float)inputImg.rows / 640.0f;

    for (int i = 0; i < outputMat.rows; ++i) {
        float score = outputMat.at<float>(i, 4);
        if (score > 0.5) {
            float cx = outputMat.at<float>(i, 0);
            float cy = outputMat.at<float>(i, 1);
            float w = outputMat.at<float>(i, 2);
            float h = outputMat.at<float>(i, 3);

            int left = static_cast<int>((cx - 0.5 * w) * ratio_x);
            int top = static_cast<int>((cy - 0.5 * h) * ratio_y);
            int width = static_cast<int>(w * ratio_x);
            int height = static_cast<int>(h * ratio_y);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(score);

            // Extract 17 keypoints (for person pose, hands use similar structure)
            std::vector<cv::Point> kpts;
            for (int k = 0; k < 17; ++k) {
                float kx = outputMat.at<float>(i, 5 + k * 3) * ratio_x;
                float ky = outputMat.at<float>(i, 5 + k * 3 + 1) * ratio_y;
                kpts.push_back(cv::Point(kx, ky));
            }
            keypoints_list.push_back(kpts);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.5, indices);

    cv::Mat resultImg = inputImg.clone();
    for (int idx : indices) {
        cv::rectangle(resultImg, boxes[idx], cv::Scalar(255, 0, 0), 2);
        for (const auto& pt : keypoints_list[idx]) {
            cv::circle(resultImg, pt, 3, cv::Scalar(0, 0, 255), -1);
        }
    }
    return resultImg;
  }

  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::string model_path_ = "model/yolov8n-pose.onnx";
  std::mutex mutex_;
};

EXPORT_NODE(YOLOHandPose)