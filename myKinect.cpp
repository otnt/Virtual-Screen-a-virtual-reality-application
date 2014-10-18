#include "myKinect.h"

// Kinect variables
INuiSensor *pNuiSensor;
HANDLE colorStream;
HANDLE nextColorFrameEvent;

HRESULT hr;

const int widthColor = 1280;
const int heightColor = 960;

bool initKinect()
{
	// Get a working kinect sensor
	hr = NuiCreateSensorByIndex(0, &pNuiSensor);
	if (FAILED(hr))
	{
		cout << "Kinect connection failed!\n";
		return false;
	}
	// Initialize sensor
	hr = pNuiSensor->NuiInitialize(	NUI_INITIALIZE_FLAG_USES_COLOR);
    if (FAILED(hr)) 
    { 
        cout<<"Kinect initialization failed!\n";
		pNuiSensor->Release(); 
		pNuiSensor->NuiShutdown(); 
        return false; 
    } 
	// Open color stream
	nextColorFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
	pNuiSensor->NuiImageStreamOpen(NUI_IMAGE_TYPE_COLOR,NUI_IMAGE_RESOLUTION_1280x960,
		0, 2, nextColorFrameEvent, &colorStream);
	return true;
}

bool getKinectColorData(Mat dest, int width, int height)
{
	if (WaitForSingleObject(nextColorFrameEvent, 0)==0)
	{
		NUI_IMAGE_FRAME colorFrame;
		hr = pNuiSensor->NuiImageStreamGetNextFrame(colorStream, 0, &colorFrame);
		if (FAILED(hr))		//no data
			return false;
		INuiFrameTexture* texture = colorFrame.pFrameTexture;
		NUI_LOCKED_RECT LockedRect;
		texture->LockRect(0, &LockedRect, NULL, 0);
		if (LockedRect.Pitch != 0)
		{
			uchar* pdata = (uchar*)dest.data;
			uchar* curr    = (uchar*)LockedRect.pBits;
			uchar* dataEnd = curr+(width*height)*4;
			//大约0.0027s/帧
			while (curr<dataEnd) //内部数据是4个字节，0-1-2是BGR，第4个现在未使用 
			{
				*pdata++ = *curr++;
				*pdata++ = *curr++;
				*pdata++ = *curr++;
				curr++;
			}
		}
		texture->UnlockRect(0);
		pNuiSensor->NuiImageStreamReleaseFrame(colorStream, &colorFrame);
		return true;
	}
	return false;
}