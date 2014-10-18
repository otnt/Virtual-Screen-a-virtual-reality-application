
//尝试对hough检测出来的直线，判断其距离和角度，对于一定范围内的线，归到一条直线内

#include <stdio.h>
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"  
#include<opencv2/legacy/legacy.hpp>
#include <opencv2\opencv.hpp>
#include <time.h>

#include "cudaWarpPerspective.h"
#include "cudaFindDarkestPoint.h"
#include "label.h"
#include "maxListClass.h"
#include "getFirstFunc.h"
#include "otsu.h"


#include <cstdlib>
#include <algorithm>
#include <vector>
#include <iterator>
#include <ctime>
#include <functional>
#include <fstream>

#include "myKinect.h"

#include "mpg123.h"



#define smoothFrameNum 5//时域均滑帧数
#define smoothSigma  5//时域均滑系数


using namespace cv;
using namespace std;

//一些全局变量
uchar* gpuDataSrc;//每读入一帧，就将它立刻送到gpu，便于做warp叠加
Mat src;//处理每一帧的灰度图
Mat frame;//每一帧新读入的图片
const int widthColor = 1280;//kinect彩色图的宽和高
const int heightColor = 960;
bool isVideo = 0;//0--pdf,1--video
char *NAME;//图片或视频的名字

//-----------------------------------------------


int main(int argc, char** argv)
{

	//必须3个参数，第2个参数为1-视频/0-图片，第3个参数为视频或照片的名称
	printf("请确保你的电脑能运行CUDA程序，并且安装了CUDA SDK！否则此程序无法运行~~\n");
	if(argc != 3)
	{
		printf("argc num:%d\n",argc);system("pause");
		printf("必须2个参数，第一个参数为‘1’-视频/‘0’-图片，第二个参数为视频或图片的名称（需要包含路径和格式后缀）!\n");
		system("pause");
		exit(1);
	}
	else
	{
		isVideo = int(argv[1][0]);
		NAME = argv[2];
	}

	//测试视频信号
	const char* filename = "pix.mp4";
	CvCapture *capture = cvCreateFileCapture(filename);

	//是否为第一帧？如果是，那么使用全局搜索四个点，否则，跟踪已经搜索到的四个点
	bool isFirst = true;
	Point firstPoint[4];

	//子领域大小：subSize x subSize,在这个范围内搜偶多
	const int subSize = 100;

	//对搜索到的点标上颜色，测试用
	Scalar pointColor[4] = {Scalar(255,0,0),Scalar(0,255,0),Scalar(0,0,255),Scalar(255,255,0)};

	//输出视频
	namedWindow("avi",0);

	//输出帧初始化
	frame.create(heightColor,widthColor,CV_8UC3);

	//显示的pdf图片/video名称
	//const char*pdfPicName = "Tulips.jpg";
	const char*pdfPicName = NAME;
	const char*showVideoName = NAME;
	CvCapture *showVideoCapture = cvCreateFileCapture(showVideoName);
	extern Mat pdfImg;
	IplImage*tmpIplImage;
	IplImage*showVideoIplImage;

	//KINECT配置
	initKinect();

	//CUDA初始化
	InitCUDA();
	if(isVideo){
		showVideoIplImage = cvQueryFrame(showVideoCapture);
		pdfImg = showVideoIplImage;
	}
	myCudaWarpInit(pdfPicName,frame);
	cudaGetSearchSizeInit();


	//视频同时保存，调试用
	IplImage kinectOutput;
	CvVideoWriter *writer = 0;
	CvVideoWriter *writerDepth = 0;
	int isColor = 1;
	int fps     = 24;  // or 30
	int frameW  = widthColor; // 744 for firewire cameras
	int frameH  = heightColor; // 480 for firewire cameras
	writer=cvCreateVideoWriter("videoForTest.avi",CV_FOURCC('D','I','V','X'),fps,cvSize(frameW,frameH),isColor);
	//writer=cvCreateVideoWriter("videoForTest.avi",CV_FOURCC('D','I','V','X'),fps,cvSize(512,292),isColor);


	//每k帧，取1帧，发现适当丢弃1帧效果可能反而更好
	const int passFrameNumConst = 2;
	int passFrameNum = passFrameNumConst;

	//保存上一次的firstPoint位置，用于在仅丢失一给点时，尝试恢复位置
	Point preFirstPoint[4];

	//连续k帧同一点丢失，那么重来
	int continuiousLost[4];
	memset(continuiousLost,0,sizeof(int)*4);
	const int continuiousLostNum = 24 / passFrameNumConst;

	//为每个点创造直方图矢量，增强自适应性
	double frameHistogram[4][256];

	//跟踪黑点的搜索范围
	int subsubSize;

	//初始窗口创建
	namedWindow("大津法二值分割",0);
	namedWindow("追踪点",0);
	namedWindow("极值图",0);
	namedWindow("avi",0);
	
	resizeWindow("avi",widthColor/2,heightColor/2);
	resizeWindow("追踪点",200,200);
	resizeWindow("大津法二值分割",widthColor/4,heightColor/4);
	resizeWindow("极值图",widthColor/4,heightColor/4);
	
	moveWindow("avi",0,0);
	moveWindow("追踪点",widthColor/2+20,0);
	moveWindow("大津法二值分割",widthColor/2+20+200+20,0);
	moveWindow("极值图",widthColor/2+20+200+20,heightColor/4+40);

	//while(tmpIplImage = cvQueryFrame(capture))
	while(1)
	{
		//frame = tmpIplImage;

		getKinectColorData(frame,widthColor,heightColor);
		Sleep(1);

		kinectOutput = frame;
		cvWriteFrame(writer,&kinectOutput);

		cvtColor(frame,src,CV_BGR2GRAY);

		//把frame传送到cuda内
		cudaFree(gpuDataSrc);
		cudaCopyFrame(frame);

		if(isFirst){
			if(getFirst(frame,firstPoint)){
				//给四个点标号
				labelPoints(firstPoint);
				//判断是否是矩形，对于第一帧，要求是一个矩形才能执行下去！
				if(isARectangle(firstPoint)){
					cvtColor(frame,src,CV_BGR2GRAY);
					//------------------对第一帧的点也要做寻找最黑点-------------------------
					//求出subsubSize
					int heightOrWidth1 = disCompute(firstPoint[0].x,firstPoint[0].y,firstPoint[1].x,firstPoint[1].y);
					int heightOrWidth2 = disCompute(firstPoint[2].x,firstPoint[2].y,firstPoint[1].x,firstPoint[1].y);
					subsubSize = min(heightOrWidth1,heightOrWidth2);
					subsubSize = (double)(subsubSize)*0.3214;//基于现实测量得到的结果
					subsubSize = max(1,subsubSize);
					//子块直方图匹配阈值
					double subThreshold = 0.5;
					double* frameHistogramThis = new double[256];

					for(int subi=0;subi<4;subi++){
						Point firstPointClone = firstPoint[subi];
						bool hasFount = false;

						//使用爬坡法寻找极点
						const int searchNum = 1;
						int findDarkestPointOffset[searchNum*2] = 
						{
							0,0
						};
						for(int i=0;i<searchNum;i++){
							firstPoint[subi] = getFirstDarkestPoint(src,firstPointClone.x+findDarkestPointOffset[i*2+0],
								firstPointClone.y+findDarkestPointOffset[i*2+1],subsubSize,subsubSize/10,subThreshold,frameHistogramThis);
							if(firstPoint[subi].x > -1){
								for(int ff=0;ff<256;ff++){
									frameHistogram[subi][ff] = frameHistogramThis[ff];
								}
								hasFount = true;
								break;
							}
						}
					}
					delete []frameHistogramThis;

					//备份第一帧的点
					for(int i=0;i<4;i++){
						preFirstPoint[i] = firstPoint[i];
					}

					/*for(int i=0;i<4;i++){
					circle(frame,firstPoint[i],10,pointColor[i],-1,8.0);
					}*/

					isFirst = false;			
				}
			}
		}
		else{
			//原本是每2帧丢弃1帧，现在觉得没有必要了
			/*if(passFrameNum --){
			cudaFree(gpuDataSrc);
			continue;
			}
			else{passFrameNum = passFrameNumConst;}*/

			//强制丢失flag
			bool lostlost[4];
			memset(lostlost,0,4*sizeof(bool));

			//搜索区域在每次结束时送给gpu，每次到来时，直接从gpu拿
			subsubSize = cudaSearchSizeRetrive();

			//子块直方图匹配阈值
			double subThreshold = 0.5;

			//-----------------并行求darkestPoint---------------------------
			
			//gpuGetDarkestPoint(src,firstPoint,subsubSize,subsubSize/5,subThreshold,frameHistogram);

			//-------------------------------------------------------------
			//对每个子块找交点
			for (int subi = 0; subi < 4; subi++){
				//找最黑点
				bool hasFount = true;
				//使用爬坡法寻找极点
				int maxIndex = 0;
				firstPoint[subi] = getDarkestPoint(src,firstPoint[subi].x,firstPoint[subi].y,subsubSize,subsubSize/5,subThreshold,frameHistogram[subi]);


				if(firstPoint[subi].x <= 0 || firstPoint[subi].x >= src.cols-1
					|| firstPoint[subi].y <= 0 || firstPoint[subi].y >= src.rows-1){
						hasFount = false;
				}
					
				if(!hasFount){
					continuiousLost[subi] ++;
					lostlost[subi] = true;
				}
				else{
					continuiousLost[subi] = 0;
				}
			}

			//最多允许丢失2个点
			int lostPointNum = lostlost[0] + lostlost[1] + lostlost[2] + lostlost[3];
			if(lostPointNum >= 3){
				cudaFree(gpuDataSrc);
				isFirst = true;
				continue;
			}
			//连续若干帧丢失，则重来
			for(int i=0;i<4;i++){
				if(continuiousLost[i] >= continuiousLostNum){
					memset(continuiousLost,0,4*sizeof(int));
					cudaFree(gpuDataSrc);
					isFirst = true;
					break;
				}
			}
			if(isFirst){
				continue;
			}
			
			Point delta(0,0);
			int countNotLost = 0;
			for(int i=0;i<4;i++){
				if(lostlost[i] == false){
					delta += firstPoint[i] - preFirstPoint[i];
					countNotLost ++;
				}
			}
			for(int i=0;i<4;i++){
				if(lostlost[i] == true){
					firstPoint[i].x =  (double)preFirstPoint[i].x + (double)delta.x/(double)countNotLost;
					firstPoint[i].y =  (double)preFirstPoint[i].y + (double)delta.y/(double)countNotLost;
					firstPoint[i].x = max(firstPoint[i].x,0);
					firstPoint[i].y = max(firstPoint[i].y,0);
					firstPoint[i].x = min(firstPoint[i].x,src.cols-1);
					firstPoint[i].y = min(firstPoint[i].y,src.rows-1);
				}
			}

			//备份这一帧的point坐标
			for(int i=0;i<4;i++){
				preFirstPoint[i] = firstPoint[i];
			}
			
			//给位置重新做标记
			labelPoints(firstPoint);

			//附上pdf图片
			if(isVideo){
				showVideoIplImage = cvQueryFrame(showVideoCapture);
				pdfImg = showVideoIplImage;
				cudaCopyPdfOrVideoImg();
				myCudaWarp(frame,firstPoint);
				cudaReleasePdfOrVideoImg();
			}
			else{
				myCudaWarp(frame,firstPoint);
			}

			//将点表示出来
			/*for (int i=0;i<4;i++)
			{
			circle(frame,firstPoint[i],10,pointColor[i],-1,8,0);
			}*/
		}
		//如果第一帧结束，把四个点送个gpu，让它计算下一次搜索黑点的范围
		if(!isFirst){
			cudaGetSearchSizeCompute(firstPoint);
		}
		
		cudaFree(gpuDataSrc);
		imshow("avi",frame);
		if(waitKey(1) == 27){
			break;
		}
	}

	//若干释放
	cvReleaseVideoWriter(&writer);
	cvReleaseCapture( &capture );
	myCudaRelease();
	cudaSearchSizeRelease();
	destroyAllWindows();

	printf("program exit successfully!\n");
	return 0;
}
