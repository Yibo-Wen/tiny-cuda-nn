#include <tiny-cuda-nn/common_device.h>

#include <tiny-cuda-nn/config.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace tcnn;
using precision_t = network_precision_t;

GPUMemory<float> load_noise(const std::string& filename, int& width, int& height, int& depth) {
	// width * height * depth
	std::ifstream infile(filename);
	if(!infile) {
		std::cerr << "Error: failed to open file " << filename << std::endl;
	}

	infile >> width >> height >> depth;
	int resolution = width * height * depth;

	if (!resolution) {
		std::cerr << "Error: Invalid input dimension" << std::endl;
	}

	float* out = new float[resolution];
	for (int i = 0; i < resolution; ++i) {
		infile >> out[i];
	}

	GPUMemory<float> result(resolution * 4);
	result.copy_from_host(out);
	free(out); // release memory of image data

	return result;
}

template <typename T>
void save_noise(const T* noise, int width, int height, int depth, const std::string& filename) {
	std::vector<float> noise_host(width * height * depth);
	CUDA_CHECK_THROW(cudaMemcpy(noise_host.data(), noise, noise_host.size() * sizeof(float), cudaMemcpyDeviceToHost));

	std::ofstream outfile(filename);
	if(!outfile) {
		std::cerr << "Error: failed to create file " << filename << std::endl;
	}

	outfile << width << " " << height << " " << depth << std::endl;
    int resolution = width * height * depth;
	for (int i = 0; i < resolution; i++) {
		outfile << noise_host[i] << " ";
	}
	outfile << std::endl;
	outfile.close();
}

template <typename T>
void save_noise_slice(const T* noise_slice, int width, int height, const std::string& filename) {
	std::vector<float> noise_host(width * height);
	CUDA_CHECK_THROW(cudaMemcpy(noise_host.data(), noise_slice, noise_host.size() * sizeof(float), cudaMemcpyDeviceToHost));

	std::ofstream outfile(filename);
	if(!outfile) {
		std::cerr << "Error: failed to create file " << filename << std::endl;
	}

	outfile << width << " " << height << std::endl;
	for (int i = 0; i < width * height; i++) {
		outfile << noise_host[i] << " ";
	}
	outfile << std::endl;
	outfile.close();
}

__global__ void eval_noise(uint32_t n_elements, cudaTextureObject_t texture, float* __restrict__ xs_and_ys_and_zs, float* __restrict__ result) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;
	uint32_t input_idx = i * 3;
	result[i] = tex3D<float>(texture, xs_and_ys_and_zs[input_idx], xs_and_ys_and_zs[input_idx+1], xs_and_ys_and_zs[input_idx+2]);
}

int main(int argc, char* argv[]) {
	try {
		uint32_t compute_capability = cuda_compute_capability();
		if (compute_capability < MIN_GPU_ARCH) {
			std::cerr
				<< "Warning: Insufficient compute capability " << compute_capability << " detected. "
				<< "This program was compiled for >=" << MIN_GPU_ARCH << " and may thus behave unexpectedly." << std::endl;
		}

		if (argc < 2) {
			std::cout << "USAGE: " << argv[0] << " " << "path-to-image.jpg [path-to-optional-config.json]" << std::endl;
			std::cout << "Sample EXR files are provided in 'data/images'." << std::endl;
			return 0;
		}

		json config = {
			{"loss", {
				{"otype", "RelativeL2"}
			}},
			{"optimizer", {
				{"otype", "Adam"},
				// {"otype", "Shampoo"},
				{"learning_rate", 1e-2},
				{"beta1", 0.9f},
				{"beta2", 0.99f},
				{"l2_reg", 0.0f},
				// The following parameters are only used when the optimizer is "Shampoo".
				{"beta3", 0.9f},
				{"beta_shampoo", 0.0f},
				{"identity", 0.0001f},
				{"cg_on_momentum", false},
				{"frobenius_normalization", true},
			}},
			{"encoding", {
				{"otype", "OneBlob"},
				{"n_bins", 32},
			}},
			{"network", {
				{"otype", "FullyFusedMLP"},
				// {"otype", "CutlassMLP"},
				{"n_neurons", 64},
				{"n_hidden_layers", 4},
				{"activation", "ReLU"},
				{"output_activation", "None"},
			}},
		};

		if (argc >= 3) {
			std::cout << "Loading custom json config '" << argv[2] << "'." << std::endl;
			std::ifstream f{argv[2]};
			config = json::parse(f, nullptr, true, /*skip_comments=*/true);
		}

		// First step: load an 3D noise txt
		int width, height, depth;
		GPUMemory<float> noise = load_noise(argv[1], width, height, depth);

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		cudaArray* d_tex_array = nullptr;
		cudaExtent extent = make_cudaExtent(width, height, depth);
		cudaMalloc3DArray(&d_tex_array, &channelDesc, extent);

		// Second step: create a cuda texture out of this noise. It'll be used to generate training data efficiently on the fly
		// copy data to texture
	    cudaMemcpy3DParms copyParams = {0};
	    copyParams.srcPtr = make_cudaPitchedPtr(noise.data(), width * sizeof(float), height, depth);
	    copyParams.dstArray = d_tex_array;
	    copyParams.extent = extent;
	    copyParams.kind = cudaMemcpyDeviceToDevice;
	    cudaMemcpy3D(&copyParams);

	    noise.free_memory();

	    cudaResourceDesc texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));
        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array  = d_tex_array;

	    // set texture parameters
		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.normalizedCoords = true;
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.addressMode[2] = cudaAddressModeClamp;

		cudaTextureObject_t texture;
		CUDA_CHECK_THROW(cudaCreateTextureObject(&texture, &texRes, &texDesc, NULL));

		// Third step: sample a reference image to dump to disk. Visual comparison of this reference image and the learned
		//             function will be eventually possible.

		int sampling_width = width;
		int sampling_height = height;

		float sampling_depth_slice = 0.5;

		// Uncomment to fix the resolution of the training task independent of input image
		// int sampling_width = 1024;
		// int sampling_height = 1024;

		uint32_t n_coords = sampling_width * sampling_height;
		uint32_t n_coords_padded = next_multiple(n_coords, batch_size_granularity);

		GPUMemory<float> sampled_noise_slice(n_coords);
		GPUMemory<float> xs_and_ys_and_zs(n_coords_padded * 3);

		std::vector<float> host_xs_and_ys_and_zs(n_coords * 3);
		for (int y = 0; y < sampling_height; ++y) {
			for (int x = 0; x < sampling_width; ++x) {
				int idx = (y * sampling_width + x) * 3;
				host_xs_and_ys_and_zs[idx+0] = (float)(x + 0.5) / (float)sampling_width;
				host_xs_and_ys_and_zs[idx+1] = (float)(y + 0.5) / (float)sampling_height;
				host_xs_and_ys_and_zs[idx+2] = sampling_depth_slice;
			}
		}

		xs_and_ys_and_zs.copy_from_host(host_xs_and_ys_and_zs.data());

		linear_kernel(eval_noise, 0, nullptr, n_coords, texture, xs_and_ys_and_zs.data(), sampled_noise_slice.data());

		save_noise_slice(sampled_noise_slice.data(), sampling_width, sampling_height, "reference.txt");

		// Fourth step: train the model by sampling the above image and optimizing an error metric

		// Various constants for the network and optimization
		const uint32_t batch_size = 1 << 18;
		const uint32_t n_training_steps = argc >= 4 ? atoi(argv[3]) : 10000000;
		const uint32_t n_input_dims = 3; // 3-D noise coordinate
		const uint32_t n_output_dims = 1; // noise value

		cudaStream_t inference_stream;
		CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
		cudaStream_t training_stream = inference_stream;

		default_rng_t rng{1337};

		// Auxiliary matrices for training
		GPUMatrix<float> training_target(n_output_dims, batch_size);
		GPUMatrix<float> training_batch(n_input_dims, batch_size);

		// Auxiliary matrices for evaluation
		GPUMatrix<float> prediction(n_output_dims, n_coords_padded);
		GPUMatrix<float> inference_batch(xs_and_ys_and_zs.data(), n_input_dims, n_coords_padded);

		json encoding_opts = config.value("encoding", json::object());
		json loss_opts = config.value("loss", json::object());
		json optimizer_opts = config.value("optimizer", json::object());
		json network_opts = config.value("network", json::object());

		std::shared_ptr<Loss<precision_t>> loss{create_loss<precision_t>(loss_opts)};
		std::shared_ptr<Optimizer<precision_t>> optimizer{create_optimizer<precision_t>(optimizer_opts)};
		std::shared_ptr<NetworkWithInputEncoding<precision_t>> network = std::make_shared<NetworkWithInputEncoding<precision_t>>(n_input_dims, n_output_dims, encoding_opts, network_opts);

		auto trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(network, optimizer, loss);

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		float tmp_loss = 0;
		uint32_t tmp_loss_counter = 0;

		std::cout << "Beginning optimization with " << n_training_steps << " training steps." << std::endl;

		uint32_t interval = 10;

		for (uint32_t i = 0; i < n_training_steps; ++i) {
			bool print_loss = i % interval == 0;
			bool visualize_learned_func = argc < 5 && i % interval == 0;

			// Compute reference values at random coordinates
			{
				generate_random_uniform<float>(training_stream, rng, batch_size * n_input_dims, training_batch.data());
				linear_kernel(eval_noise, 0, training_stream, batch_size, texture, training_batch.data(), training_target.data());
			}

			// Training step
			{
				auto ctx = trainer->training_step(training_stream, training_batch, training_target);

				if (i % std::min(interval, (uint32_t)100) == 0) {
					tmp_loss += trainer->loss(training_stream, *ctx);
					++tmp_loss_counter;
				}
			}

			// Debug outputs
			{
				if (print_loss) {
					std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
					std::cout << "Step#" << i << ": " << "loss=" << tmp_loss/(float)tmp_loss_counter << " time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

					tmp_loss = 0;
					tmp_loss_counter = 0;
				}

				if (visualize_learned_func) {
					network->inference(inference_stream, inference_batch, prediction);
					auto filename = fmt::format("{}.txt", i);
					std::cout << "Writing '" << filename << "'... ";
					save_noise_slice(prediction.data(), sampling_width, sampling_height, filename); // MODIFIED
					std::cout << "done." << std::endl;
				}

				// Don't count visualizing as part of timing
				// (assumes visualize_learned_pdf is only true when print_loss is true)
				if (print_loss) {
					begin = std::chrono::steady_clock::now();
				}
			}

			if (print_loss && i > 0 && interval < 1000) {
				interval *= 10;
			}
		}

		// Dump final image if a name was specified
		if (argc >= 5) {
			network->inference(inference_stream, inference_batch, prediction);
			save_noise_slice(prediction.data(), sampling_width, sampling_height, argv[4]);
		}

		free_all_gpu_memory_arenas();

		// If only the memory arenas pertaining to a single stream are to be freed, use
		//free_gpu_memory_arena(stream);
	} catch (std::exception& e) {
		std::cout << "Uncaught exception: " << e.what() << std::endl;
	}

	return EXIT_SUCCESS;
}