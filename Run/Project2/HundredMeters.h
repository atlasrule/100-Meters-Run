#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cuda_runtime.h>

#define NUM_THREADS_PER_BLOCK 256

class Runner {
public:
	float speed;
	float position;
	bool finished;

	__device__ Runner(float s) : speed(s), position(0), finished(false) {}
};

__global__ void race(Runner *runners, int num_runners) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < num_runners) {
		curandState_t state;
		curand_init(clock64(), tid, 0, &state);
		float rand_speed = curand_uniform(&state) * 4 + 1;
		runners[tid] = Runner(rand_speed);

		while (!runners[tid].finished) {
			runners[tid].position += runners[tid].speed;
			__syncthreads();
			if (runners[tid].position >= 100) {
				runners[tid].finished = true;
				int finished = 0;
				for (int i = 0; i < num_runners; i++) {
					if (runners[i].finished) {
						finished++;
					}
				}
				if (finished == 1) {
					printf("First runner reached the finish line with speed: %f m/s\n", runners[tid].speed);
					printf("Current positions:\n");
					for (int i = 0; i < num_runners; i++) {
						printf("Runner %d: %f\n", i + 1, runners[i].position);
					}
				}
				__syncthreads();
			}
		}
	}
}

__global__ void sort_runners(Runner *runners, int num_runners) {
	for (int i = 0; i < num_runners - 1; i++) {
		for (int j = 0; j < num_runners - i - 1; j++) {
			if (runners[j].position < runners[j + 1].position) {
				std::swap(runners[j], runners[j + 1]);
			}
		}
	}

	printf("Race finished!\n");
	printf("Final standings:\n");
	for (int i = 0; i < num_runners; i++) {
		printf("%d. Runner with speed %f m/s, finished in %f seconds\n", i + 1, runners[i].speed, runners[i].position / runners[i].speed);
	}
}

int main() {
	int num_runners = 100;
	Runner *dev_runners;
	cudaMalloc((void**)&dev_runners, num_runners * sizeof(Runner));

	race << <(num_runners + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK >> > (dev_runners, num_runners);
	sort_runners << <1, 1 >> > (dev_runners, num_runners);

	cudaFree(dev_runners);
	return 0;
}
