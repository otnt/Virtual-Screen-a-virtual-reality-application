
#include "cudaWarpPerspective.h"

//一些全局变量
Mat pdfImg;
extern uchar* gpuDataSrc;
uchar* gpuDataImg;
float* gpuDataMapmap;
int* gpuDataRowAndCol;

bool InitCUDA()
{
    int count;

    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "There is no device./n");
        return false;
    }

    int i;
    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if(prop.major >= 1) {
                break;
            }
        }
    }

    if(i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x./n");
        return false;
    }

    cudaSetDevice(i);

    return true;
}

//data是最终要存储的图片，imgData是贴上去的图片
__global__ void CudaWarpPerspectiveLeftHalf(uchar* srcData,uchar* imgData,float* Mdata,int*gpuDataRowAndCol,int imgChannels)//int srcRows,int srcCols,int imgRows,int imgCols)
{
	int tid = threadIdx.x;//tid表示列col
	int bid = blockIdx.x;//bid表示行row

				int newX = int((Mdata[0]*float(tid)+Mdata[1]*float(bid)+Mdata[2])/(Mdata[6]*float(tid)+Mdata[7]*float(bid)+Mdata[8]));
				int newY = int((Mdata[3]*float(tid)+Mdata[4]*float(bid)+Mdata[5])/(Mdata[6]*float(tid)+Mdata[7]*float(bid)+Mdata[8]));

				if(newX<0 || newX>gpuDataRowAndCol[3]-1 || newY<0 || newY>gpuDataRowAndCol[2]-1){
					return;
				}
				else{
					uchar *pdata = srcData + bid*gpuDataRowAndCol[1]*3 + tid*3;

					uchar *pdataImg  = imgData + newY*gpuDataRowAndCol[3]*imgChannels + newX*imgChannels;//  pdfImg.ptr<uchar>(newY);

					//如果是四通道，透明部分，则推出
					if(imgChannels == 4 && *(pdataImg+3) == 0){
						return;
					}
					//pdataImg += 3*newX;
					*pdata++ = *pdataImg ++;
					*pdata++ = *pdataImg ++;
					*pdata = *pdataImg;
				}
}

__global__ void CudaWarpPerspectiveRightHalf(uchar* srcData,uchar* imgData,float* Mdata,int*gpuDataRowAndCol,int imgChannels)//int srcRows,int srcCols,int imgRows,int imgCols)
{
	int tid = threadIdx.x;//tid表示列col
	tid += gpuDataRowAndCol[1]/2;
	int bid = blockIdx.x;//bid表示行row

				int newX = int((Mdata[0]*float(tid)+Mdata[1]*float(bid)+Mdata[2])/(Mdata[6]*float(tid)+Mdata[7]*float(bid)+Mdata[8]));
				int newY = int((Mdata[3]*float(tid)+Mdata[4]*float(bid)+Mdata[5])/(Mdata[6]*float(tid)+Mdata[7]*float(bid)+Mdata[8]));

				if(newX<0 || newX>gpuDataRowAndCol[3]-1 || newY<0 || newY>gpuDataRowAndCol[2]-1){
					return;
				}
				else{
					uchar *pdata = srcData + bid*gpuDataRowAndCol[1]*3 + tid*3;

					uchar *pdataImg  = imgData + newY*gpuDataRowAndCol[3]*imgChannels + newX*imgChannels;//  pdfImg.ptr<uchar>(newY);

					//如果是四通道，透明部分，则推出
					if(imgChannels == 4 && *(pdataImg+3) == 0){
						return;
					}
					//pdataImg += 3*newX;
					*pdata++ = *pdataImg ++;
					*pdata++ = *pdataImg ++;
					*pdata = *pdataImg;
				}
}

bool cudaCopyPdfOrVideoImg()
{
	cudaMalloc((void**) &gpuDataImg, pdfImg.channels()*pdfImg.rows*pdfImg.cols*sizeof(uchar));
	cudaMemcpy(gpuDataImg, pdfImg.data, pdfImg.channels()*pdfImg.rows*pdfImg.cols*sizeof(uchar), cudaMemcpyHostToDevice);
	return true;
}

extern bool isVideo;
bool myCudaWarpInit(const char* pdfPicName,Mat src)
{
	if(!isVideo){
		//读入图片
		pdfImg = imread(pdfPicName,-1);
		//复制数据
		//copy 叠加上去的图
		cudaMalloc((void**) &gpuDataImg, pdfImg.channels()*pdfImg.rows*pdfImg.cols*sizeof(uchar));
		cudaMemcpy(gpuDataImg, pdfImg.data, pdfImg.channels()*pdfImg.rows*pdfImg.cols*sizeof(uchar), cudaMemcpyHostToDevice);
	}
	//copy 两个图的row和col
	int cpuDataRowAndCol[4] = {src.rows,src.cols,pdfImg.rows,pdfImg.cols};
	cudaMalloc((void**) &gpuDataRowAndCol,4*sizeof(int));
	cudaMemcpy(gpuDataRowAndCol,cpuDataRowAndCol,4*sizeof(int),cudaMemcpyHostToDevice);
	
	return true;
}

void myCudaRelease()
{
	cudaFree(gpuDataImg);
	cudaFree(gpuDataRowAndCol);
}

void cudaReleasePdfOrVideoImg()
{
	cudaFree(gpuDataImg);
}

void myCudaWarp(Mat src,const Point* dest)
{
		const int NUM = src.rows*src.cols ;
		const int THREAD_NUM = src.cols/2;//cols数量超出最大thread限制（1024），故/2
		const int BLOCK_NUM = src.rows;	

		//对应矩阵
		CvPoint2D32f psrc[4],pdst[4];

		/*psrc[0].x = 0;psrc[0].y = 0;
		psrc[1].x = pdfImg.cols-1;psrc[1].y = 0;
		psrc[2].x = pdfImg.cols-1;psrc[2].y = pdfImg.rows-1;
		psrc[3].x = 0;psrc[3].y = pdfImg.rows-1;*/

		//再往里缩一点，这样可以遮住四个标志物
		//int offsetC = (double)src.rows*0.2143, offsetR = (double)src.rows*0.2143;
		int offsetC =pdfImg.cols*0.1, offsetR = pdfImg.rows*0.2;
		psrc[0].x = 0+offsetC;psrc[0].y = 0+offsetR;
		psrc[1].x = pdfImg.cols-1-offsetC;psrc[1].y = 0+offsetR;
		psrc[2].x = pdfImg.cols-1-offsetC;psrc[2].y = pdfImg.rows-1-offsetR;
		psrc[3].x = 0+offsetC;psrc[3].y = pdfImg.rows-1-offsetR;

		for (int i = 0; i < 4; i++)
		{
			pdst[i].x = dest[i].x;
			pdst[i].y = dest[i].y;
		}
		//算出变换矩阵
		CvMat *mapmapcv = cvCreateMat(3,3,CV_32FC1);
			//获得从dst到src的映射，而非从src到dst的映射！这样后面可以省略一步invert
		cvGetPerspectiveTransform(pdst,psrc,mapmapcv);
		Mat mapmap = mapmapcv;
		float* pdatamap = (float*)mapmap.data;
			
		//copy 变换矩阵
		cudaMalloc((void**) &gpuDataMapmap , 9*sizeof(float));
		cudaMemcpy(gpuDataMapmap,pdatamap,9*sizeof(float),cudaMemcpyHostToDevice);
		
		//运算
		CudaWarpPerspectiveLeftHalf<<<BLOCK_NUM, THREAD_NUM>>>(gpuDataSrc,gpuDataImg,gpuDataMapmap,gpuDataRowAndCol,pdfImg.channels());
		CudaWarpPerspectiveRightHalf<<<BLOCK_NUM, THREAD_NUM>>>(gpuDataSrc,gpuDataImg,gpuDataMapmap,gpuDataRowAndCol,pdfImg.channels());

		//copy 回来
		cudaMemcpy(src.data, gpuDataSrc, NUM * 3 *sizeof(uchar), cudaMemcpyDeviceToHost);

	return;
}

void cudaCopyFrame(Mat frame)
{
	//复制数据
	//copy 要画的图（被叠加的图）
	cudaMalloc((void**) &gpuDataSrc, frame.rows*frame.cols * frame.channels() *sizeof(uchar));
	cudaMemcpy(gpuDataSrc, frame.data, frame.rows*frame.cols  * frame.channels() *sizeof(uchar), cudaMemcpyHostToDevice);
}