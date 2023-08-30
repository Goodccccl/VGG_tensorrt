//#include<iostream>
//#include<NvInfer.h>
//#include<NvInferRuntimeCommon.h>
//#include<vector>
//#include<fstream>
//#include<string>
//#include"utils.h"
//#include"logger.h"
//#include<cuda_runtime.h>
//#include <opencv.hpp>
//
//int main()
//{
//	std::string image_path = "F:/Workprojects/TongFu_Bump/test/NG_bump_1_L0.tif_2023_3_8_10_5_12_445_178_s1_0.907_s2_0.855_s3_0.836_s4_0.456_rErr_1.190.bmp";
//
//	auto engine_data = utils::loadModel("F:/C++ projects/VGG_tensorrt/VGG_tensorrt/MyVGG.trt");
//	auto runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
//	auto engine = (runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
//
//
//	// 创建cuda流，确定这个batch的推理是独立的
//	cudaStream_t stream = nullptr;
//	cudaStreamCreate(&stream);
//	auto context = engine->createExecutionContext();
//
//	int input_batch = 1;
//	int input_channel = 3;
//	int input_height = 48;
//	int input_width = 48;
//
//	// 准备好input_data_host和input_data_device，分别表示内存中的数据指针和显存中的数据指针
//	// 一会儿将预处理过的图像数据搬运到GPU
//	int input_numel = input_batch * input_channel * input_height * input_width;
//	float* input_data_host = nullptr;
//	float* input_data_device = nullptr;
//
//	cudaMallocHost(&input_data_host, input_numel * sizeof(float));
//	cudaMalloc(&input_data_device, input_numel * sizeof(float));
//
//	auto src_image = utils::preprocess(image_path, input_height, input_width);
//
//	cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream);
//	const int num_class = 1;
//	float output_data_host[num_class];
//	float* output_data_device = nullptr;
//	cudaMalloc(&output_data_device, sizeof(output_data_host));
//
//	auto input_dims = engine->getBindingDimensions(0);
//	input_dims.d[0] = input_batch;
//	context->setBindingDimensions(0, input_dims);
//	// 用一个指针数组bindings指定input和output在gpu中的指针
//	float* bindings[] = { input_data_device, output_data_device };
//	context->enqueueV2((void**)bindings, stream, nullptr);
//	cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
//	cudaStreamSynchronize(stream);
//
//	std::cout << output_data_host[0] << std::endl;
//
//	cudaStreamDestroy(stream);
//	cudaFreeHost(input_data_host);
//	cudaFree(input_data_device);
//	cudaFree(output_data_device);
//}