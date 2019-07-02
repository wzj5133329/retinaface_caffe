#include "anchor_generator.h"
#include<sys/time.h>

uint64_t current_timestamp();
class Detector {
 public:
  Detector(const std::string& model_file,
           const std::string& weights_file,          
		   const float confidence,
       const float nms,
		   const std::string& gpu_mode);

  std::vector<Anchor> Detect(cv::Mat& img);
 private:
  std::shared_ptr<caffe::Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  float confidence_threshold;
  float nms_threshold;

  float ratio_w=0.0;
  float ratio_h=0.0;
private:
	void preprocess(const  cv::Mat& img, std::vector<cv::Mat>* input_channels);
	void wrapInputLayer(std::shared_ptr<caffe::Net<float> > net_, std::vector<cv::Mat>* input_channels);

 
};