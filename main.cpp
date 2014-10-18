
//���Զ�hough��������ֱ�ߣ��ж������ͽǶȣ�����һ����Χ�ڵ��ߣ��鵽һ��ֱ����

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



#define smoothFrameNum 5//ʱ�����֡��
#define smoothSigma  5//ʱ�����ϵ��


using namespace cv;
using namespace std;

//һЩȫ�ֱ���
uchar* gpuDataSrc;//ÿ����һ֡���ͽ��������͵�gpu��������warp����
Mat src;//����ÿһ֡�ĻҶ�ͼ
Mat frame;//ÿһ֡�¶����ͼƬ
const int widthColor = 1280;//kinect��ɫͼ�Ŀ�͸�
const int heightColor = 960;
bool isVideo = 0;//0--pdf,1--video
char *NAME;//ͼƬ����Ƶ������

//-----------------------------------------------


int main(int argc, char** argv)
{

	//����3����������2������Ϊ1-��Ƶ/0-ͼƬ����3������Ϊ��Ƶ����Ƭ������
	printf("��ȷ����ĵ���������CUDA���򣬲��Ұ�װ��CUDA SDK������˳����޷�����~~\n");
	if(argc != 3)
	{
		printf("argc num:%d\n",argc);system("pause");
		printf("����2����������һ������Ϊ��1��-��Ƶ/��0��-ͼƬ���ڶ�������Ϊ��Ƶ��ͼƬ�����ƣ���Ҫ����·���͸�ʽ��׺��!\n");
		system("pause");
		exit(1);
	}
	else
	{
		isVideo = int(argv[1][0]);
		NAME = argv[2];
	}

	//������Ƶ�ź�
	const char* filename = "pix.mp4";
	CvCapture *capture = cvCreateFileCapture(filename);

	//�Ƿ�Ϊ��һ֡������ǣ���ôʹ��ȫ�������ĸ��㣬���򣬸����Ѿ����������ĸ���
	bool isFirst = true;
	Point firstPoint[4];

	//�������С��subSize x subSize,�������Χ����ż��
	const int subSize = 100;

	//���������ĵ������ɫ��������
	Scalar pointColor[4] = {Scalar(255,0,0),Scalar(0,255,0),Scalar(0,0,255),Scalar(255,255,0)};

	//�����Ƶ
	namedWindow("avi",0);

	//���֡��ʼ��
	frame.create(heightColor,widthColor,CV_8UC3);

	//��ʾ��pdfͼƬ/video����
	//const char*pdfPicName = "Tulips.jpg";
	const char*pdfPicName = NAME;
	const char*showVideoName = NAME;
	CvCapture *showVideoCapture = cvCreateFileCapture(showVideoName);
	extern Mat pdfImg;
	IplImage*tmpIplImage;
	IplImage*showVideoIplImage;

	//KINECT����
	initKinect();

	//CUDA��ʼ��
	InitCUDA();
	if(isVideo){
		showVideoIplImage = cvQueryFrame(showVideoCapture);
		pdfImg = showVideoIplImage;
	}
	myCudaWarpInit(pdfPicName,frame);
	cudaGetSearchSizeInit();


	//��Ƶͬʱ���棬������
	IplImage kinectOutput;
	CvVideoWriter *writer = 0;
	CvVideoWriter *writerDepth = 0;
	int isColor = 1;
	int fps     = 24;  // or 30
	int frameW  = widthColor; // 744 for firewire cameras
	int frameH  = heightColor; // 480 for firewire cameras
	writer=cvCreateVideoWriter("videoForTest.avi",CV_FOURCC('D','I','V','X'),fps,cvSize(frameW,frameH),isColor);
	//writer=cvCreateVideoWriter("videoForTest.avi",CV_FOURCC('D','I','V','X'),fps,cvSize(512,292),isColor);


	//ÿk֡��ȡ1֡�������ʵ�����1֡Ч�����ܷ�������
	const int passFrameNumConst = 2;
	int passFrameNum = passFrameNumConst;

	//������һ�ε�firstPointλ�ã������ڽ���ʧһ����ʱ�����Իָ�λ��
	Point preFirstPoint[4];

	//����k֡ͬһ�㶪ʧ����ô����
	int continuiousLost[4];
	memset(continuiousLost,0,sizeof(int)*4);
	const int continuiousLostNum = 24 / passFrameNumConst;

	//Ϊÿ���㴴��ֱ��ͼʸ������ǿ����Ӧ��
	double frameHistogram[4][256];

	//���ٺڵ��������Χ
	int subsubSize;

	//��ʼ���ڴ���
	namedWindow("��򷨶�ֵ�ָ�",0);
	namedWindow("׷�ٵ�",0);
	namedWindow("��ֵͼ",0);
	namedWindow("avi",0);
	
	resizeWindow("avi",widthColor/2,heightColor/2);
	resizeWindow("׷�ٵ�",200,200);
	resizeWindow("��򷨶�ֵ�ָ�",widthColor/4,heightColor/4);
	resizeWindow("��ֵͼ",widthColor/4,heightColor/4);
	
	moveWindow("avi",0,0);
	moveWindow("׷�ٵ�",widthColor/2+20,0);
	moveWindow("��򷨶�ֵ�ָ�",widthColor/2+20+200+20,0);
	moveWindow("��ֵͼ",widthColor/2+20+200+20,heightColor/4+40);

	//while(tmpIplImage = cvQueryFrame(capture))
	while(1)
	{
		//frame = tmpIplImage;

		getKinectColorData(frame,widthColor,heightColor);
		Sleep(1);

		kinectOutput = frame;
		cvWriteFrame(writer,&kinectOutput);

		cvtColor(frame,src,CV_BGR2GRAY);

		//��frame���͵�cuda��
		cudaFree(gpuDataSrc);
		cudaCopyFrame(frame);

		if(isFirst){
			if(getFirst(frame,firstPoint)){
				//���ĸ�����
				labelPoints(firstPoint);
				//�ж��Ƿ��Ǿ��Σ����ڵ�һ֡��Ҫ����һ�����β���ִ����ȥ��
				if(isARectangle(firstPoint)){
					cvtColor(frame,src,CV_BGR2GRAY);
					//------------------�Ե�һ֡�ĵ�ҲҪ��Ѱ����ڵ�-------------------------
					//���subsubSize
					int heightOrWidth1 = disCompute(firstPoint[0].x,firstPoint[0].y,firstPoint[1].x,firstPoint[1].y);
					int heightOrWidth2 = disCompute(firstPoint[2].x,firstPoint[2].y,firstPoint[1].x,firstPoint[1].y);
					subsubSize = min(heightOrWidth1,heightOrWidth2);
					subsubSize = (double)(subsubSize)*0.3214;//������ʵ�����õ��Ľ��
					subsubSize = max(1,subsubSize);
					//�ӿ�ֱ��ͼƥ����ֵ
					double subThreshold = 0.5;
					double* frameHistogramThis = new double[256];

					for(int subi=0;subi<4;subi++){
						Point firstPointClone = firstPoint[subi];
						bool hasFount = false;

						//ʹ�����·�Ѱ�Ҽ���
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

					//���ݵ�һ֡�ĵ�
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
			//ԭ����ÿ2֡����1֡�����ھ���û�б�Ҫ��
			/*if(passFrameNum --){
			cudaFree(gpuDataSrc);
			continue;
			}
			else{passFrameNum = passFrameNumConst;}*/

			//ǿ�ƶ�ʧflag
			bool lostlost[4];
			memset(lostlost,0,4*sizeof(bool));

			//����������ÿ�ν���ʱ�͸�gpu��ÿ�ε���ʱ��ֱ�Ӵ�gpu��
			subsubSize = cudaSearchSizeRetrive();

			//�ӿ�ֱ��ͼƥ����ֵ
			double subThreshold = 0.5;

			//-----------------������darkestPoint---------------------------
			
			//gpuGetDarkestPoint(src,firstPoint,subsubSize,subsubSize/5,subThreshold,frameHistogram);

			//-------------------------------------------------------------
			//��ÿ���ӿ��ҽ���
			for (int subi = 0; subi < 4; subi++){
				//����ڵ�
				bool hasFount = true;
				//ʹ�����·�Ѱ�Ҽ���
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

			//�������ʧ2����
			int lostPointNum = lostlost[0] + lostlost[1] + lostlost[2] + lostlost[3];
			if(lostPointNum >= 3){
				cudaFree(gpuDataSrc);
				isFirst = true;
				continue;
			}
			//��������֡��ʧ��������
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

			//������һ֡��point����
			for(int i=0;i<4;i++){
				preFirstPoint[i] = firstPoint[i];
			}
			
			//��λ�����������
			labelPoints(firstPoint);

			//����pdfͼƬ
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

			//�����ʾ����
			/*for (int i=0;i<4;i++)
			{
			circle(frame,firstPoint[i],10,pointColor[i],-1,8,0);
			}*/
		}
		//�����һ֡���������ĸ����͸�gpu������������һ�������ڵ�ķ�Χ
		if(!isFirst){
			cudaGetSearchSizeCompute(firstPoint);
		}
		
		cudaFree(gpuDataSrc);
		imshow("avi",frame);
		if(waitKey(1) == 27){
			break;
		}
	}

	//�����ͷ�
	cvReleaseVideoWriter(&writer);
	cvReleaseCapture( &capture );
	myCudaRelease();
	cudaSearchSizeRelease();
	destroyAllWindows();

	printf("program exit successfully!\n");
	return 0;
}
