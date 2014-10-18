
#include "cudaFindDarkestPoint.h"

extern uchar* gpuDataSrc;
int *gpuSearchSize, *gpuSearchSizeResult;

__global__ void getPatchValue(uchar*gpuSrcData,uchar*gpuSmallSeq,int sx,int sy,int rows,int cols,int subSize,int beginKernalSize)
{
	int tid = threadIdx.x;//tid表示列col
	int bid = blockIdx.x;//bid表示行row

	int px=sx-subSize/2 + tid;
	int py =sy-subSize/2 + bid;

	//基于先验知识，边缘点可以不做检测，直接返回一个极大值
	if(py - beginKernalSize/2 <= 0 || py + beginKernalSize/2 >= rows-1
		|| px - beginKernalSize/2 <= 0 || px + beginKernalSize/2 >= cols-1){
		gpuSmallSeq[bid*subSize+tid]  = 0;
		return;
	}

	float result = 0.0;
	for(int i=-beginKernalSize/2;i<beginKernalSize/2;i++)
				{
					uchar* pdata_ori = gpuSrcData;
					pdata_ori += (py+i)*cols*3 + (px-beginKernalSize/2)*3;//src.ptr<uchar>(py+i);
					//pdata_ori += px-beginKernalSize/2;
					for(int j=-beginKernalSize/2;j<beginKernalSize/2;j++)
					{
						//Gray = 0.212671 * R + 0.715160 * G + 0.072169 * B +0 *A 
						result += (float)*pdata_ori * 0.072169;
						pdata_ori++;
						result += (float)*pdata_ori * 0.715160;
						pdata_ori++;
						result += (float)*pdata_ori * 0.212671;
						pdata_ori++;
					}
				}
	
	result = (double)(255*beginKernalSize*beginKernalSize - result)/(double)(beginKernalSize*beginKernalSize);
	gpuSmallSeq[bid*subSize+tid]  = (int)result;
	return;
}


//在src的sx和sy邻域（subSize x subSize块内）,寻找beginKernalSize x beginKernalSize块最黑的坐标，返回src中的坐标
Point getDarkestPoint(Mat src,int sx,int sy,int subSize,int beginKernalSize,double ththreshold,double *histoGood)
{
	//----------------------使用GPU计算这些beginKernalSize x beginKernalSize块的像素值之和，归一化到0~255，便于显示，也不会太影响精度----------------------------
	const int NUM = subSize*subSize ;
	const int THREAD_NUM = subSize;
	const int BLOCK_NUM = subSize;	

	uchar *smallSeq1 = new uchar[subSize*subSize];
	uchar *gpuSmallSeq1;
	cudaMalloc((void**) &gpuSmallSeq1, subSize*subSize*sizeof(uchar));
	getPatchValue<<<BLOCK_NUM, THREAD_NUM>>>(gpuDataSrc,gpuSmallSeq1,sx,sy,src.rows,src.cols,subSize,beginKernalSize);
	//copy 回来
	cudaMemcpy(smallSeq1, gpuSmallSeq1, subSize*subSize*sizeof(uchar), cudaMemcpyDeviceToHost);
	//-----------------------------------GPU运算完毕----------------------------------------------

	//------------------------------------对这张新图求最黑的点，对所有最黑的点取位置平均---------------------------------
	//还是自己的算法好用啊T T
	int maxSmallSeq = 0;
	for(int i=0;i<subSize*subSize;i++){
		if(maxSmallSeq < smallSeq1[i])
			maxSmallSeq = smallSeq1[i];
	}
	int countSmallSeq = 0,rowrow = 0,colcol = 0;
	for(int i=0;i<subSize*subSize;i++){
		if(maxSmallSeq == smallSeq1[i]){
			countSmallSeq ++;
			rowrow += i/subSize;
			colcol += i%subSize;
		}
	}
	countSmallSeq = max(countSmallSeq,1);
	rowrow /= countSmallSeq;
	colcol /= countSmallSeq;
	//蛋疼，矫正
	swap(rowrow,colcol);
	//--------------------------------------------求点结束------------------------------------------------------

	Mat forShowBlackCamShiftMat(subSize,subSize,CV_8UC1);
	forShowBlackCamShiftMat.data = (uchar*)smallSeq1;

	//-----------------------------------求以最黑点为中心的histogram---------------------------------------
	//之前似乎将x视作col，将y视作row
	int px = sx - subSize/2 + colcol, py = sy - subSize/2 + rowrow;
	int roiSize = subSize;//roi- roiSize * roiSize

	double *cpuHistogram1 = new double[256];
	memset(cpuHistogram1,0,256*sizeof(double));

	int countcount = 0;

	for(int i=-roiSize/2;i<roiSize/2;i++)
	{
		//py+i可能会越界
		int tmp = py+i;
		if(tmp<0)
			continue;//前几行保留黑色
		else if (tmp > src.rows-1) 
			break;//最后几行保留黑色

		uchar* pdata_ori = src.ptr<uchar>(tmp);

		//px-roiSize/2可能小于0,这几列跳过，保留黑色
		int continueCols = 0;
		if(px-roiSize/2 < 0) 
			continueCols = roiSize/2 -px;
		//px+roiSize/2可能越界，这几列也保留黑色
		int breakCols = roiSize;
		if(px+roiSize/2 > src.cols-1)
			breakCols = roiSize-((px+roiSize/2)-(src.cols -1));
		pdata_ori += (max)(px-roiSize/2,0);
		for(int j=-roiSize/2;j<roiSize/2;j++)
		{
			while(continueCols && continueCols--)//跳过头几列
			{
				pdata_ori++;
				continue;
			}
			if(breakCols-- == 0)//跳过最后几列
				break;
			cpuHistogram1[*pdata_ori] ++;
			countcount ++;
			pdata_ori++;
		}
	}
	//直方图归一化
	for (int i = 0; i < 256; i++){
		cpuHistogram1[i] /= countcount;
	}
	//--------------------------------------------求histogram结束---------------------------------------------

	//--------------------------------histogram匹配-----------------------------------------
	double match = 0;
	for (int i = 0; i < 256; i++){
		match += (min)(histoGood[i],cpuHistogram1[i]);
	}

	//如果match超过0.6，则更新直方图
	if(match >= 0.6){
		for (int i=0;i<256;i++){
			histoGood[i] = cpuHistogram1[i];
		}
	}

	circle(forShowBlackCamShiftMat,Point(rowrow,colcol),5,Scalar(0),-1,8,0);
	namedWindow("追踪点",0);
	imshow("追踪点",forShowBlackCamShiftMat);
	imwrite("追踪点.jpg",forShowBlackCamShiftMat);
	waitKey(1);
	//------------------------------匹配结束--------------------------------------

	//-------------------------------返回点----------------------------------------------------
	cudaFree(gpuSmallSeq1);
	delete []smallSeq1;
	delete []cpuHistogram1;


	if(match > ththreshold)
	{
		return Point(sx - subSize/2 + rowrow, sy - subSize/2 + colcol);
	}
	else
		return Point(-1,-1);

}

//在src的sx和sy邻域（subSize x subSize块内）,寻找beginKernalSize x beginKernalSize块最黑的坐标，返回src中的坐标
Point getFirstDarkestPoint(Mat src,int sx,int sy,int subSize,int beginKernalSize,double ththreshold,double* frameHistogram)
{
	//----------------------使用GPU计算这些beginKernalSize x beginKernalSize块的像素值之和，归一化到0~255，便于显示，也不会太影响精度----------------------------
	const int NUM = subSize*subSize ;
	const int THREAD_NUM = subSize;
	const int BLOCK_NUM = subSize;	

	uchar *smallSeq1 = new uchar[subSize*subSize];
	uchar *gpuSmallSeq1;
	cudaMalloc((void**) &gpuSmallSeq1, subSize*subSize*sizeof(uchar));
	getPatchValue<<<BLOCK_NUM, THREAD_NUM>>>(gpuDataSrc,gpuSmallSeq1,sx,sy,src.rows,src.cols,subSize,beginKernalSize);
	//copy 回来
	cudaMemcpy(smallSeq1, gpuSmallSeq1, subSize*subSize*sizeof(uchar), cudaMemcpyDeviceToHost);

	//--------------------------------------------求点结束------------------------------------------------------

	//-------------------------------------找到最黑点--------------------------------------------------
	//还是自己的算法好用啊T T
	int maxSmallSeq = 0;
	for(int i=0;i<subSize*subSize;i++){
		if(maxSmallSeq < smallSeq1[i])
			maxSmallSeq = smallSeq1[i];
	}
	int countSmallSeq = 0,rowrow = 0,colcol = 0;
	for(int i=0;i<subSize*subSize;i++){
		if(maxSmallSeq == smallSeq1[i]){
			countSmallSeq ++;
			rowrow += i/subSize;
			colcol += i%subSize;
		}
	}
	countSmallSeq = max(countSmallSeq,1);
	rowrow /= countSmallSeq;
	colcol /= countSmallSeq;
	//蛋疼，矫正
	swap(rowrow,colcol);

	Mat forShowBlackCamShiftMat(subSize,subSize,CV_8UC1);
	forShowBlackCamShiftMat.data = smallSeq1;
	//-----------------------------------找点结束------------------------------------------------

	//-----------------------------------获得直方图--------------------------------------------
	//之前似乎将x视作col，将y视作row
	int px = sx - subSize/2 + colcol, py = sy - subSize/2 + rowrow;
	int roiSize = subSize;//roi- roiSize * roiSize

	double *cpuHistogram1 = new double[256];
	memset(cpuHistogram1,0,256*sizeof(double));

	int countcount = 0;

	for(int i=-roiSize/2;i<roiSize/2;i++)
	{
		//py+i可能会越界
		int tmp = py+i;
		if(tmp<0)
			continue;//前几行保留黑色
		else if (tmp > src.rows-1) 
			break;//最后几行保留黑色

		uchar* pdata_ori = src.ptr<uchar>(tmp);

		//px-roiSize/2可能小于0,这几列跳过，保留黑色
		int continueCols = 0;
		if(px-roiSize/2 < 0) 
			continueCols = roiSize/2 -px;
		//px+roiSize/2可能越界，这几列也保留黑色
		int breakCols = roiSize;
		if(px+roiSize/2 > src.cols-1)
			breakCols = roiSize-((px+roiSize/2)-(src.cols -1));
		pdata_ori += (max)(px-roiSize/2,0);
		for(int j=-roiSize/2;j<roiSize/2;j++)
		{
			while(continueCols && continueCols--)//跳过头几列
			{
				pdata_ori++;
				//pdata_roi++;
				continue;
			}
			if(breakCols-- == 0)//跳过最后几列
				break;
			cpuHistogram1[*pdata_ori] ++;
			countcount ++;
			//*pdata_roi = *pdata_ori;
			//pdata_roi++;
			pdata_ori++;
		}
	}
	//直方图归一化
	for (int i = 0; i < 256; i++){
		cpuHistogram1[i] /= countcount;
	}

	//第一帧得到的直方图，作为后续匹配的标准
	for (int i = 0; i < 256; i++){
		frameHistogram[i] = cpuHistogram1[i];
	}

	circle(forShowBlackCamShiftMat,Point(rowrow,colcol),5,Scalar(0),-1,8,0);
	namedWindow("追踪点",0);
	imshow("追踪点",forShowBlackCamShiftMat);
	waitKey(1);
	//-------------------------------------直方图获取结束----------------------------------------------


	//------------------------------------------------------------------------------------------
	//返回的点
	Point returnPoint(sx - subSize/2 + rowrow, sy - subSize/2 + colcol);

	cudaFree(gpuSmallSeq1);
	delete []smallSeq1;
	delete []cpuHistogram1;

	return returnPoint;
}

void cudaGetSearchSizeInit()
{
	cudaMalloc((void**) &gpuSearchSize, 2*4*sizeof(int));//4个点，每个点有xy坐标
	cudaMalloc((void**) &gpuSearchSizeResult,1*sizeof(int));
}

void cudaSearchSizeRelease()
{
	cudaFree(gpuSearchSize);
	cudaFree(gpuSearchSizeResult);
}

__global__ void cudaSearchSizeCompute(int* gpuPoints,int *result)
{
	int x1,y1,x2,y2,subsubSize;

	x1 = gpuPoints[0],y1 = gpuPoints[1];
	x2 = gpuPoints[2],y2 = gpuPoints[3];
	int disdis1 =  int(  sqrt(  float((x1-x2)*(x1-x2))+float((y1-y2)*(y1-y2))  )  );

	x1 = gpuPoints[2],y1 = gpuPoints[3];
	x2 = gpuPoints[4],y2 = gpuPoints[5];
	int disdis2 =  int(  sqrt(  float((x1-x2)*(x1-x2))+float((y1-y2)*(y1-y2))  )  );

	subsubSize = min(disdis1,disdis2);
	subsubSize = (double)(subsubSize)*0.3214;//基于测量得到的结果
	subsubSize = max(1,subsubSize);

	*result = subsubSize;
}

void cudaGetSearchSizeCompute(Point* p)
{
	int cpuPoints[8] = 
	{
		p[0].x,p[0].y,
		p[1].x,p[1].y,
		p[2].x,p[2].y,
		p[3].x,p[3].y
	};
	cudaMemcpy(gpuSearchSize,cpuPoints,2*4*sizeof(int),cudaMemcpyHostToDevice);
	cudaSearchSizeCompute<<<1,1>>>(gpuSearchSize,gpuSearchSizeResult);
}

int cudaSearchSizeRetrive()
{
	int cpuSearchSizeResult;
	cudaMemcpy(&cpuSearchSizeResult,gpuSearchSizeResult,1*sizeof(int),cudaMemcpyDeviceToHost);
	return cpuSearchSizeResult;
}
