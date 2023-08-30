#include<iostream>
#include<NvInfer.h>
#include<NvInferRuntimeCommon.h>
#include<vector>
#include<fstream>
#include<string>
#include"utils.h"
#include"logger.h"



int main()
{
	std::string trt_path = "F:/C++ projects/VGG_tensorrt/VGG_tensorrt/MyVGG.trt";		// trtģ�͵�ַ
	std::string img_path = "F:/C++ projects/VGG_tensorrt/VGG_tensorrt/2.bmp";		// Ԥ��ͼƬ�ļ���λ��

	const int batchSize = 1;				// Ԥ��batch_size
	const int input_channel = 3;			// �����ͨ����
	const int input_height = 48;
	const int input_width = 48;				// ����ĸߺͿ�
	const int outputSize = 1;				

	// ����engine
	std::vector<unsigned char> model_data = utils::loadModel(trt_path);
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());	// ���������runtimeʵ�����ӿ�
	nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(model_data.data(), model_data.size());	// �����л�ģ�͵õ�engine
	nvinfer1::IExecutionContext *context = engine->createExecutionContext();		// ����context�������

	// �������������ڴ�
	//void* buffers[2];	// Ϊ���롢������û�����
	const char* INPUT_BLOB_NAME = "input";
	const char* OUTPUT_BLOB_NAME = "output";
	const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
	const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);	// �õ�engine����������������

	// cuda malloc
	void* input_mem{ nullptr };
	cudaMalloc(&input_mem, batchSize * input_channel * input_height * input_width * sizeof(float));
	void* output_mem{ nullptr };
	cudaMalloc(&output_mem, batchSize * outputSize * sizeof(float));

	std::vector<float> outputData(batchSize * outputSize);
	//float outputData[4];

	// ����ͼƬԤ����
	float* inputData = utils::preprocess(img_path, input_height, input_width);
	/*std::cout << *inputData << std::endl;*/
	//for (int i = 0; i < 3 * input_height * input_width; i++)
	//{
	//	std::cout << i <<":"<< *&inputData[i] << std::endl;
	//}

	// ����cuda��
	cudaStream_t stream;
	cudaStreamCreate(&stream);
		// cudaMemcpy ͬ��ִ��	cudaMemcpyAsync �첽ִ��	���������ݴ�cpu������gpu��
	cudaMemcpyAsync(input_mem, inputData, batchSize * input_channel * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice, stream);

	// ִ������
	void* bindings[] = { input_mem, output_mem };
	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < outputSize; i++)
	{
		//context->enqueue(batchSize, buffers, stream, nullptr);
		context->enqueueV2(bindings, stream, nullptr);
	}
	auto end = std::chrono::system_clock::now(); // ����ʱ��
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
	// �������gpu������cpu��
	cudaMemcpyAsync(outputData.data(), output_mem, batchSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
	std::cout << outputData[0] << std::endl;

	std::cout << "Top Predictions:" << std::endl;
	for (int i = 0; i < 1; i++)
	{
		float maxVal = -0.5f;
		float maxIndex = -1;
		for (int j = 0; j < outputSize; j++)
		{
			if (outputData[j] > maxVal)
			{
				maxVal = outputData[j];
				maxIndex = j;
			}
		}
		std::cout << "Class " << maxIndex << ": " << maxVal << std::endl;
		outputData[maxIndex] = -1.0f; // �������÷���Ϊ������ȷ�������ظ����
	}

	// �ͷ�
	context->destroy();
	engine->destroy();
	runtime->destroy();
}