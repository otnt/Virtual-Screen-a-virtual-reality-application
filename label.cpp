
#include "label.h"

//����ֱ��(p1,p2)��(p3,p4)�Ľ���
Point getIntersectionPoint(Point p1,Point p2,Point p3,Point p4)
{
	int y1_y3 = p1.y-p3.y;
	int x1_x2 = p1.x-p2.x;
	int x3_x4 = p3.x-p4.x;
	int y3_y4 = p3.y-p4.y;
	int y1_y2 = p1.y-p2.y;

	float px = (float(y1_y3*x1_x2*x3_x4+p3.x*y3_y4*x1_x2-p1.x*x3_x4*y1_y2)/float(y3_y4*x1_x2-x3_x4*y1_y2));
	float py = float(p1.y)-float(y1_y2)/float(x1_x2)*float(p1.x-px);

	return  Point(int(px),int(py));
}


void labelPoints(Point* firstPoint)
{
	//���ĸ����ţ�˳ʱ��˳������Ϊ1
	//��һ�����ҵ����ĵ�,��������ϣ�01&23,02&13,03&12
	Point rect[4];
	int seq[12] = {0,1,2,3,0,2,1,3,0,3,1,2};
	int seqIndex = 0;
	Point tmpPoint;
	int disThres = 0.5*(double)(sqrt  ( pow(frame.rows,2)+pow(frame.cols,2) )  );//��Ļ�Խ��ߵ�һ�볤
	int selectGroup = 0;//����ѡ���˵ڼ������
	int selectGroupNum = 0;//������ѡ�е������������1����ô��Ҫ����
	float disdis;
	for (int i = 0; i < 3; i++){
		for (int r = 0; r < 4; r++){
			rect[r] = firstPoint[seq[seqIndex++]];
		}
		//����ֱ�ߵĽ����Ƿ����ı����ڣ�������naive���������ĸ�������룬���벻Ӧ����һ��threshold
		tmpPoint = getIntersectionPoint(rect[0],rect[1],rect[2],rect[3]);
		bool withinRect = true;
		for (int j = 0; j < 4; j++){
			//disdis�п����������ɼ��㲻�ȶ��������ǰ��ǰ��һ���жϣ���ʵ����жϻ����͹���
			if(tmpPoint.x <= 0 || tmpPoint.x >= frame.cols-1
				|| tmpPoint.y <= 0 || tmpPoint.y >= frame.rows-1){
					withinRect = false;
					break;
			}
			//ʵ�ڲ�������dis�ж�
			disdis = disComputef(tmpPoint.x,tmpPoint.y,rect[j].x,rect[j].y);
			if(disdis> disThres){
				withinRect = false;
				break;
			}
		}
		if(withinRect){
			selectGroup = i;
			selectGroupNum++;
		}
	}

	//�ڶ�����12һ�������ĵ����棬34һ�������ĵ����棬1��2�����
	Point line1[2],line2[2];
	line1[0] = firstPoint[seq[4*selectGroup]];
	line1[1] = firstPoint[seq[4*selectGroup+1]];
	line2[0] = firstPoint[seq[4*selectGroup+2]];
	line2[1] = firstPoint[seq[4*selectGroup+3]];
	//line�����������ǰ��
	if(line1[0].y>line1[1].y){
		Point tmpchange = line1[0];
		line1[0] = line1[1];
		line1[1] = tmpchange;
	}
	if(line2[0].y>line2[1].y){
		Point tmpchange = line2[0];
		line2[0] = line2[1];
		line2[1] = tmpchange;
	}
	//�Ƚ�����line��ͷԪ�أ���ߵ���13���ұߵ���24
	if(line1[0].x < line2[0].x){
		firstPoint[0] = line1[0];
		firstPoint[1] = line2[0];
		firstPoint[2] = line1[1];
		firstPoint[3] = line2[1];
	}
	else{
		firstPoint[0] = line2[0];
		firstPoint[1] = line1[0];
		firstPoint[2] = line2[1];
		firstPoint[3] = line1[1];
	}
	//���ˣ�˳��ȷ�����

	return;
}