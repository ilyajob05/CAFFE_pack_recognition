/*
 * predictor.cpp
 *
 *  Created on: Feb 3, 2017
 *      Author: ilya
 */

#include "predictor.h"
#include <iostream>



Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& label_file) {
cout << "set mode cpu/gpu...";
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif
cout << "complete" << endl;

  /* Load the network. */
	cout << "open net...";
  net_.reset(new Net<float>(model_file, TEST));
  cout << "complete" << endl;

  cout << "load weight...";
  net_->CopyTrainedLayersFrom(trained_file);
  cout << "complete" << endl;

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  cout << "load mean...";
  /// todo:
  ///SetMean(mean_file);
  cout << "complete" << endl;

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}


static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}


std::vector< std::pair<int,float> > Classifier::Classify(const std::vector<cv::Mat>& img) {
  std::vector< std::vector<float> > output = Predict(img, img.size());

  std::vector<std::pair <int, float>> predictions;
  for ( int i = 0 ; i < output.size(); i++ )
  {
    std::vector<int> maxN = Argmax(output[i], 1);
    int idx = maxN[0];
    predictions.push_back(std::make_pair(idx, output[i][idx]));
  }
  return predictions;
}



/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}



std::vector< std::vector<float> > Classifier::Predict(const std::vector<cv::Mat>& img, int nImages)
{
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(nImages, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels, nImages);

  Preprocess(img, &input_channels, nImages);

  net_->Forward();

  /* Copy the output layer to a std::vector */

  Blob<float>* output_layer = net_->output_blobs()[0];
  std::vector <std::vector<float> > ret;
  for (int i = 0; i < nImages; i++)
  {
    const float* begin = output_layer->cpu_data() + i*output_layer->channels();
	//const float* begin = output_layer->gpu_data() + i*output_layer->channels();
    const float* end = begin + output_layer->channels();
    ret.push_back( std::vector<float>(begin, end) );
  }
  return ret;
}


/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels, int nImages) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels()* nImages; ++i) 
  {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const std::vector<cv::Mat>& img,
                            std::vector<cv::Mat>* input_channels, int num)
{
  for (int i = 0; i < num; i++)
  {
	cv::Mat tmpImg;

	if (img[i].channels() == 3 && num_channels_ == 1)
	{
		cv::cvtColor(img[i], tmpImg, cv::COLOR_BGR2GRAY);
	}
	else if (img[i].channels() == 4 && num_channels_ == 1)
	{
		cv::cvtColor(img[i], tmpImg, cv::COLOR_BGRA2GRAY);
	}
	else if (img[i].channels() == 4 && num_channels_ == 3)
	{
		cv::cvtColor(img[i], tmpImg, cv::COLOR_BGRA2BGR);
	}
	else if (img[i].channels() == 1 && num_channels_ == 3)
	{
		cv::cvtColor(img[i], tmpImg, cv::COLOR_GRAY2BGR);
	}
	else
	{
		tmpImg = img[i];
	}

	if (tmpImg.size() != input_geometry_)
	{
		cout << "resize" << endl;
		cv::resize(tmpImg, tmpImg, input_geometry_, CV_INTER_LANCZOS4);
	}

	if (num_channels_ == 3)
	{
		tmpImg.convertTo(tmpImg, CV_32FC3);
	}
	else
	{
		tmpImg.convertTo(tmpImg, CV_32FC1);
	}

	vector<cv::Mat> channels;
	cv::split(tmpImg, channels);

	for (uint j = 0; j < channels.size(); j++)
	{
	   channels[j].copyTo((*input_channels)[i*num_channels_+j]);
	}
  }

}



