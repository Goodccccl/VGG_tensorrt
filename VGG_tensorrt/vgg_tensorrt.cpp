//#include"utils.h"
//#include<opencv.hpp>
//#include<logger.h>
//#include<cuda_runtime.h>
//
//using namespace std;
//
//int main(int argc, char** argv)
//{
//	cv::CommandLineParser parser(argc, argv,
//		{
//			"{model 	|| tensorrt model file	   }"
//			"{size      || image (h, w), eg: 640   }"
//			"{batch_size|| batch size              }"
//			"{video     || video's path			   }"
//			"{img       || image's path			   }"
//			"{cam_id    || camera's device id	   }"
//			"{show      || if show the result	   }"
//			"{savePath  || save path, can be ignore}"
//		});
//
//	// path
//	std::string model_path = "../../data/yolov8/yolov8n.trt";
//	std::string image_path = "../../data/bus.jpg";
//
//	// set parameters
//	size_t batch_size = 8;
//	int img_size = 48;
//
//	// load model 
//	std::vector<unsigned char> trt_file = utils::loadModel(model_path);
//	if (trt_file.empty())
//	{
//		sample::gLogError << "trt File is empty!" << endl;
//		return -1;
//	}
//	std::unique_ptr<nvinfer1::IRuntime> runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
//	if (runtime == nullptr)
//	{
//		return -1;
//	}
//	std::unique_ptr<nvinfer1::ICudaEngine> engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(trt_file.data(), trt_file.size()));
//	if (engine == nullptr)
//	{
//		return -1;
//	}
//	std::unique_ptr<nvinfer1::IExecutionContext> context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
//	if (context == nullptr)
//	{
//		return -1;
//	}
//	nvinfer1::Dims m_output_dims = context->getBindingDimensions(1);
//	int m_total_objects = m_output_dims.d[2];
//	assert(batch_size <= m_output_dims.d[0]);
//	int output_area = 1;
//	for (int i = 1; i < m_output_dims.nbDims; i++)
//	{
//		if (m_output_dims.d[i] != 0)
//		{
//			output_area *= m_output_dims.d[i];
//		}
//	}
//	// malloc
//	float* m_output_src_device;
//	float* m_output_src_transpose_device;
//	cudaMalloc(&m_output_src_device, batch_size * output_area * sizeof(float));
//	cudaMalloc(&m_output_src_transpose_device, batch_size * output_area * sizeof(float));
//	
//
//	cv::Mat frame;
//	std::vector<cv::Mat> imgs_batch;
//	imgs_batch.reserve(batch_size);
//	frame = cv::imread(image_path);
//	context->enqueue()
//}