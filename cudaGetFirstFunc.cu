#include "getFirstFunc.h"

extern Mat src;

bool getFirst(Mat frame,Point* firstPoint)
{
		//---------------------------��򷨶�ֵ��ͼ��------------------------------
		const int rowDivision = 10,colDivision = 10;
		const int rowLength = src.rows/rowDivision,colLength = src.cols/colDivision;
		const int minimumGap = 50;

		for (int r = 0; r < rowDivision; r++)
		{
			for (int c = 0; c < colDivision; c++)
			{
				uchar* srcData = src.data;
				int otsuThreshold =	otsu(srcData,src.rows,src.cols,colLength*c,rowLength*r,colLength,rowLength,0,minimumGap);
				int beginRow = r*rowLength,beginCol = c*colLength;
				for (int rr = 0; rr < rowLength; rr++)
				{
					srcData = src.ptr<UCHAR>(beginRow+rr);
					srcData += beginCol;
					for(int cc = 0;cc<colLength;cc++)
					{
						if(*srcData > otsuThreshold)
							*srcData = 255;
						else
							*srcData = 0;
						srcData++;
					}
				}
			}
		}
		namedWindow("��򷨶�ֵ�ָ�",0);
		imshow("��򷨶�ֵ�ָ�",src);
		waitKey(1);
		//--------------------------------��򷨽���------------------------------------------------

	//------------------------canny���+hough�任----------------------------------------------------
	Canny(src, src, 50, 200,3);

	vector<Vec4i> lines;
	const int th = 30;
	const int thgap = 20;
	HoughLinesP(src, lines, 1, CV_PI/180, th, th, thgap );

	//���lines����һ������ôֱ���˳�
	//printf("line size: %d\n",lines.size());
	if(lines.size() > 1024)
		return false;
	//-----------------------canny���+hough�任����----------------------------------------------------

	//---------------------������ֱ����������չ---------------------------------
	int expandVal = 50;

	src.setTo(0);
	vector<Vec4i> concisedLines;
	Vec4i concisedPoint;
	
	for (int i=0;i<lines.size();i++){

		Vec4i l = (lines[i]);

		int x1 = l[0],y1 = l[1],x2 = l[2],y2 = l[3];

				//���ֱ��̫����ô��Ϊ���޹�ֱ�ߣ�ֱ��ȥ��
				int tooLong = src.cols / 4;
				if( pow(x1-x2,2)+pow(y1-y2,2) > pow(tooLong,2) )
					continue;

				//--------------����������չ-----------------

				int x1_x2 = x1-x2,y1_y2 = y1-y2;
				int gap = disCompute(x1,y1,x2,y2);
				float scale = float(expandVal)/float(gap);
				int x1new = int(scale*float(x1_x2) + x1);
				int y1new = int(scale*float(y1_y2) + y1);
				int x2new = int(scale*float(-1*x1_x2) + x2);
				int y2new = int(scale*float(-1*y1_y2) + y2);
				x1 = (max)(x1new,0);
				x1 = (min)(x1,src.cols-1);
				x2 = (max)(x2new,0);
				x2 = (min)(x2,src.cols-1);
				y1 = (max)(y1new,0);
				y1 = (min)(y1,src.rows-1);
				y2 = (max)(y2new,0);
				y2 = (min)(y2,src.rows-1);

				line(src,Point(x1,y1),Point(x2,y2),Scalar(255),3,CV_AA);

				concisedPoint[0] = x1;
				concisedPoint[1] = y1;
				concisedPoint[2] = x2;
				concisedPoint[3] = y2;
				concisedLines.push_back(concisedPoint);			  

	}
	//-------------------��չ��������------------------------------------

	//�ѽ������һ�£����ں�������ҵ��ĵ������ֱ��ͼ
	Mat srcCopy = src.clone();

	//-------------------���㽻��-------------------------------------------------------------------------------
	src.setTo(0);
	//�Ըղ��ҵ���ֱ�ߣ����»���ͼ�ϣ�����ʹ�õ��ӵķ�ʽ�����������Եõ���ֵ��
	for( size_t i = 0; i < concisedLines.size(); i++ )
	{
		Vec4i l = (concisedLines[i]);
		DrawLine(src,l[0],l[1],l[2],l[3],10,4);
	}
	namedWindow("��ֵͼ",0);
	imshow("��ֵͼ",src);
	imwrite("peakValue.jpg",src);
	waitKey(1);

	//����ֵ�㣬�洢maxListLength�����ĵ㣬������Щ�㣬�õ�����ֱ��ͼ������ǰ�ĸ���
	const int maxListLength = 1000;
	_maxList maxlist(maxListLength);

	for(int i=0;i<src.rows;i++)
	{
		uchar *pdata = src.ptr<uchar>(i);
		for(int j=0;j<src.cols;j++)
		{
			if(*pdata == 0){
				pdata ++;
				continue;
			}
			maxlist.insert(*pdata,i,j);
			pdata ++;
		}
	}
	maxlist.update(srcCopy,0.7);
	//------------------���㽻�����-------------------------------------------------

	if(maxlist.fgetPointNum() == 4){
		for (int i = 0; i < 4; i++){
			firstPoint[i] = maxlist.fgetpoint()[i];
		}
		return true;
	}
	else{
		return false;
	}
}