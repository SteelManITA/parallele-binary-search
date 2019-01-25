#include <functional>

/* divisione intera per eccesso a/b */
__host__ __device__ __forceinline__
int div_up(int a, int b)
{
	return (a + b - 1)/b;
}

__device__ __forceinline__
int getId()
{
    return threadIdx.x + blockIdx.x*blockDim.x;
}

void error(const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

void cudaCheck(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %d - %s\n", msg,
            err, cudaGetErrorString(err));
        exit(1);
    }
}

char* concat(const char *s1, const char *s2)
{
    char *s = new char[strlen(s1) + strlen(s2) + 1];
    s[0] = '\0';
    return strcat(strcat(s, s1), s2);
}

void cudaRunEvent(
    const char *eventName,
    std::function<void()> event,
    unsigned int transferredBytes
) {
    float runtime;
    double bandwidth;
    cudaError_t err;

    cudaEvent_t start, end;
    err = cudaEventCreate(&start);
    cudaCheck(err, "create start");
    err = cudaEventCreate(&end);
    cudaCheck(err, "create end");

    // start event
    err = cudaEventRecord(start);
    cudaCheck(err, concat("start ", eventName));
    event();
    err = cudaEventRecord(end);
    cudaCheck(err, concat("end ", eventName));
    err = cudaEventSynchronize(end);
    cudaCheck(err, concat(concat("end ", eventName), " sync"));
    err = cudaEventElapsedTime(&runtime, start, end);
    cudaCheck(err, concat("elapsed time ", eventName));
    bandwidth = (transferredBytes)/1.0e6/runtime;

    printf("%-10s:\t%-10.9gms\t%-10.9gGB/s\n", eventName, runtime, bandwidth);
}