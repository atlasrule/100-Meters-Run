/*

	-	The Hundred Meters Race	    -

	The code is intended to simulate the race of 100 runners on a 100 meters run.

	Number of runners can be set.

	In each clock cycle each runner's speed updated as a PRNG between 1-5 m/s.
	And calculates its position.

	When the first runner finishes the race, positions printed.
	
	Ranks printed out when runners completes the track.

	The program can be rewritten as shared memory which provides better parallelism when data transaction between chaces needed.
	Using shared memory may actually slow down performance if the data is only used once and there is no reuse of the data among different threads within a block.

	The number 256 used in updatePositions<<<(NUM_RUNNERS + 255) / 256, 256>>>(runners_d, deltaTime)
	represents the number of threads per block, set it as power of 2
	It's usually best to launch as many threads as possible in each thread block to take full advantage of the GPU parallelism.

*/


#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>

#define NUM_RUNNERS 100


class Runner {
	public:
		__device__ __host__ Runner() {
			speed = curand_uniform(&rng) * 4.0f + 1.0f; // Random speed between 1 and 5 m/s
			position = 0.0f;
		}

		__device__ __host__ void updatePosition(float deltaTime) {
			position += speed * deltaTime;
		}

		__device__ __host__ float getPosition() const {
			return position;
		}

	private:
		float speed;
		float position;
		curandState rng;
};


__global__ void updatePositions(Runner* runners, float deltaTime) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < NUM_RUNNERS) {
		runners[i].updatePosition(deltaTime);
	}
}

void wait(int milliseconds) {
	clock_t endwait = clock() + milliseconds * CLOCKS_PER_SEC / 1000;
	while (clock() < endwait) {}
}


int main() {

	// Allocate memory for the runners on the host and device
	Runner* runners_h = new Runner[NUM_RUNNERS];
	Runner* runners_d;
	cudaMalloc((void**)&runners_d, NUM_RUNNERS * sizeof(Runner));
	cudaMemcpy(runners_d, runners_h, NUM_RUNNERS * sizeof(Runner), cudaMemcpyHostToDevice);

	// Initialize the CUDA random number generator
	curandGenerator_t rng;
	curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(rng, time(NULL));

	// Initialize the race variables
	float timeElapsed = 0.0f;
	clock_t lastUpdateTime = clock();

	while (true) {
		// Calculate the time elapsed since the last update
		clock_t now = clock();
		float deltaTime = (float)(now - lastUpdateTime) / CLOCKS_PER_SEC;
		timeElapsed += deltaTime;

		// Check if any runner has finished the race
		for (int i = 0; i < NUM_RUNNERS; i++) {
			if (runners_h[i].getPosition() >= 100.0f) {
				// Print the positions of all runners when the winner finished
				std::cout << "Race finished in " << std::setprecision(2) << std::fixed << timeElapsed << " s\n";
				std::cout << "Runner " << std::setw(3) << i + 1 << " is the winner!\n";
				std::cout << "Positions:\n";
				for (int j = 0; j < NUM_RUNNERS; j++) {
					std::cout << "Runner " << std::setw(3) << j + 1 << ": " << std::setw(7) << runners_h[j].getPosition() << " m\n";
				}
				std::cout << std::endl;
				// Free the memory and exit
				delete[] runners_h;
				cudaFree(runners_d);
				curandDestroyGenerator(rng);
				return 0;
			}
		}

		// Update the positions of the runners on the device
		updatePositions <<<(NUM_RUNNERS + 255) / 256, 256 >>> (runners_d, deltaTime);
		cudaDeviceSynchronize();

		// Copy the updated positions back to the host
		cudaMemcpy(runners_h, runners_d, NUM_RUNNERS * sizeof(Runner), cudaMemcpyDeviceToHost);

		// Wait for one second before the next update
		wait(1000);

		// Update the last update time
		lastUpdateTime = now;
	}

	// This point should never be reached
	return 0;
}