# 100-Meters-Run
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
