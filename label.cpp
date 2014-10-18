
#include "label.h"

//返回直线(p1,p2)和(p3,p4)的交点
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
	//给四个点标号，顺时针顺序，左上为1
	//第一步：找到中心点,共三种组合：01&23,02&13,03&12
	Point rect[4];
	int seq[12] = {0,1,2,3,0,2,1,3,0,3,1,2};
	int seqIndex = 0;
	Point tmpPoint;
	int disThres = 0.5*(double)(sqrt  ( pow(frame.rows,2)+pow(frame.cols,2) )  );//屏幕对角线的一半长
	int selectGroup = 0;//最终选择了第几组组合
	int selectGroupNum = 0;//如果最后被选中的组合数不等于1，那么就要报错！
	float disdis;
	for (int i = 0; i < 3; i++){
		for (int r = 0; r < 4; r++){
			rect[r] = firstPoint[seq[seqIndex++]];
		}
		//看两直线的交点是否在四边形内：现在用naive方法：对四个点求距离，距离不应大于一个threshold
		tmpPoint = getIntersectionPoint(rect[0],rect[1],rect[2],rect[3]);
		bool withinRect = true;
		for (int j = 0; j < 4; j++){
			//disdis有可能溢出，造成计算不稳定，因此在前面前做一个判断，其实这个判断基本就够了
			if(tmpPoint.x <= 0 || tmpPoint.x >= frame.cols-1
				|| tmpPoint.y <= 0 || tmpPoint.y >= frame.rows-1){
					withinRect = false;
					break;
			}
			//实在不行再用dis判断
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

	//第二步，12一定在中心点上面，34一定在中心点下面，1在2的左边
	Point line1[2],line2[2];
	line1[0] = firstPoint[seq[4*selectGroup]];
	line1[1] = firstPoint[seq[4*selectGroup+1]];
	line2[0] = firstPoint[seq[4*selectGroup+2]];
	line2[1] = firstPoint[seq[4*selectGroup+3]];
	//line中上面的排在前面
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
	//比较两个line中头元素，左边的是13，右边的是24
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
	//至此，顺序确定完毕

	return;
}