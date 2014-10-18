#include "mainAuxiliaryFuncs.h"

void range(int &val,int maxNum,int minNum)
{
	val = (min)(val,maxNum);
	val = (max)(val,minNum);
}


int rangeWithReturn(int val,int maxNum,int minNum)
{
	val = (min)(val,maxNum);
	val = (max)(val,minNum);
	return val;
}

int disCompute(int x1,int y1,int x2,int y2)
{
	int disdis =  int(  sqrt(  float((x1-x2)*(x1-x2))+float((y1-y2)*(y1-y2))  )  );
	return disdis;
}


float disComputef(int x1,int y1,int x2,int y2)
{
	float disdis =  double(  sqrt(  double((x1-x2)*(x1-x2))+double((y1-y2)*(y1-y2))  )  );
	return disdis;
}


void disAndThetaCompute(int x1,int y1,int x2,int y2,int *dis,int *theta)
{
	//swap(x1,y1);
	//swap(x2,y2);
	int y1_y2 = y1-y2;
	int x1_x2 = x1-x2;
	*dis =abs ( int(   float(x2*y1_y2 - y2*x1_x2)/sqrt(  float(y1_y2*y1_y2  + x1_x2* x1_x2)  )  )  );

	if(y1==y2 && y1 >= 0)
	{
		*theta = 90.0;
		return;
	}
	if(y1==y2 && y1 < 0)
	{
		*theta = 270.0;
		return;
	}
	if(x1 == x2 && x1 >= 0)
	{
		*theta = 0.0;
		return;
	}
	if(x1 == x2 && x1 < 0)
	{
		*theta = 180;
		return;
	}
	if(*dis == 0)
	{
		*theta = 0.0;
		return;
	}

	float k = float( x2*y1 - x1*y2 )/float(y1_y2);
	//printf("x1: %d,x2: %d,y1: %d,y2: %d,y1-y2: %d\n",x1,x2,y1,y2,y1_y2);

	float thetaTmp= acosf( abs(float(*dis) / k)  );
	thetaTmp = thetaTmp/3.141592653 * 180;

	//printf("k: %f,thetaTmp: %f\n",k,thetaTmp);

	bool incidence = ((y1_y2 > 0 && x1_x2 > 0) || (y1_y2<0 && x1_x2<0) )? 0:1;//这里坐标和正常的坐标略有不同
	//float incidence = float(y1_y2)/float(x1_x2);

	if(incidence==1   &&  k >= 0) //theta 270~360
	{
		*theta = int(360-thetaTmp);
	}
	else if(incidence==1  && k<0) //theta 90~180
	{
		*theta = int(180-thetaTmp);
	}
	else if(incidence==0  && k>=0)//theta 0~90
	{
		*theta = int(thetaTmp);
	}
	else if(incidence==0  && k<0)//theta = 180~270
	{
		*theta = int(thetaTmp + 180);
	}
	else
	{//ASSERT//assert(0);
	}

	if(*theta < -1000)
	{system("pause");}

}


void SetPixel(Mat img, int x, int y,char color)
{
	uchar *pdata = img.ptr<uchar>(y);
	/*pdata += x*3;
	*pdata++ += color;
	*pdata++ += color;
	*pdata++ += color;*/

	pdata += x;
	*pdata++ += color;
}



void DrawLine(Mat img, int x1, int y1, int x2, int y2,char color,int thick )
{
	if(thick > 1)//连续画多条直线
	{
		thick /= 2;//左右两边各多画thick条

		int y1_y2 = y1-y2,x1_x2 = x1-x2;

		int incidence = ((y1_y2 > 0 && x1_x2 > 0) || (y1_y2<0 && x1_x2<0) )? -1:1;//这里坐标和正常的坐标略有不同
		float k = (x1_x2==0)?99999:(abs(float(y1_y2)/float(x1_x2)))*incidence;
		//k *= incidence;
		int x1new,x2new,y1new,y2new;
		for(int i=-thick;i<thick+1;i++)//2*thick+1条
		{
			x1new = x1+i;
			range(x1new,img.cols-1,0);
			x2new = x2+i;
			range(x2new,img.cols-1,0);
			if(k<99998)//不是垂直线
			{
				y1new = y1 + float(i)*k;
				range(y1new,img.rows-1,0);
				y2new = y2 + float(i)*k;
				range(y2new,img.rows-1,0);
			}
			else
			{
				y1new = y1;
				y2new = y2;
			}
			DrawLine(img,x1new,y1new,x2new,y2new,color,1);
		}
	}
	if(thick <= 1)
	{
		int dx = x2 - x1;
		int dy = y2 - y1;
		int ux = ((dx > 0) << 1) - 1;//x的增量方向，取或-1
		int uy = ((dy > 0) << 1) - 1;//y的增量方向，取或-1
		int x = x1, y = y1, eps;//eps为累加误差

		eps = 0;dx = abs(dx); dy = abs(dy); 
		if (dx > dy) 
		{
			for (x = x1; x != x2; x += ux)
			{
				SetPixel(img, x, y,color);
				eps += dy;
				if ((eps << 1) >= dx)
				{
					y += uy; eps -= dx;
				}
			}
		}
		else
		{
			for (y = y1; y != y2; y += uy)
			{
				SetPixel(img, x, y,color);
				eps += dx;
				if ((eps << 1) >= dy)
				{
					x += ux; eps -= dy;
				}
			}
		}         
	}
}

bool isARectangle(Point* p)
{
	float verysmallpixels = 50;
	//p12水平，p34水平
	if(abs(p[0].y - p[1].y) > verysmallpixels || abs(p[2].y - p[3].y) > verysmallpixels)
		return false;

	//p23垂直，p14垂直
	if(abs(p[1].x - p[2].x) > verysmallpixels || abs(p[0].x - p[3].x) > verysmallpixels)
		return false;

	return true;
}