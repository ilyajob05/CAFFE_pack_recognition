/*
 * predictor.h
 *
 *  Created on: Feb 3, 2017
 *      Author: ilya
 */

#ifndef PREDICTOR_H_
#define PREDICTOR_H_

#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>

//#define CPU_ONLY

using namespace std;
using namespace cv;
using namespace caffe;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;
typedef std::pair<cv::Rect, int> PredictionPosition;



class Classifier {
 public:
  std::vector< std::pair<int,float> > Classify(const std::vector<cv::Mat>& img);

  Classifier(const string& model_file,
             const string& trained_file,
             const string& label_file);


 private:
  std::vector< std::vector<float> > Predict(const std::vector<cv::Mat>& img, int size);
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  std::vector<string> labels_;
  Mat mean_;

  void WrapInputLayer(std::vector<cv::Mat>* input_channels, int nImages);

  void Preprocess(const std::vector<cv::Mat>& img,
                  std::vector<cv::Mat>* input_channels, int nImages);

  void SetMean(const string& mean_file);

};



#endif /* PREDICTOR_H_ */
