//Nvidia官方给的程序
//可以看到自己电脑的GPU是否能运行cuda程序，以及gpu的性能

///********************************************************************
//*  InitCUDA.cu
//*  This is a init CUDA of the CUDA program.
//*********************************************************************/
//#include <stdio.h>
//#include <stdlib.h>
//#include <cuda_runtime.h>
//
///************************************************************************/
///* Init CUDA                                                            */
///************************************************************************/
//int main(void)
//{
//	int count = 0;
//	int i = 0;
//
//	cudaGetDeviceCount(&count);
//	if(count == 0) {
//		fprintf(stderr, "There is no device./n");
//		return false;
//	}
//
//	for(i = 0; i < count; i++) {
//		cudaDeviceProp prop;
//		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess)
//		{                
//			printf("name:%s\n",                                         prop.name);
//			printf("totalGlobalMem:%u\n",                   prop.totalGlobalMem);
//			printf("sharedMemPerBlock:%u\n",                prop.sharedMemPerBlock);
//			printf("regsPerBlock:%d\n",                           prop.regsPerBlock);
//			printf("warpSize:%d\n",                               prop.warpSize);
//			printf("memPitch:%u\n",                               prop.memPitch);
//			printf("maxThreadsPerBlock:%d\n",               prop.maxThreadsPerBlock);
//			printf("maxThreadsDim:x %d, y %d, z %d\n",      prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
//			printf("maxGridSize:x %d, y %d, z %d\n",      prop.maxGridSize[0],prop.maxGridSize[0],prop.maxGridSize[0]);
//			printf("totalConstMem:%u\n",                    prop.totalConstMem);
//			printf("major:%d\n",                                  prop.major);
//			printf("minor:%d\n",                                  prop.minor);
//			printf("clockRate:%d\n",                              prop.clockRate);
//			printf("textureAlignment:%u\n",                       prop.textureAlignment);
//
//			if(prop.major >= 1) {
//				break;
//			}
//
//		}
//	}
//	if(i == count) {
//		fprintf(stderr, "There is no device supporting CUDA 1.x./n");
//		return false;
//	}
//	cudaSetDevice(i);
//
//	printf("CUDA initialized./n");
//	return true;
//}