//OTSU方法计算图像二值化的自适应阈值 
/*
OTSU 算法可以说是自适应计算单阈值（用来转换灰度图像为二值图像）的简单高效方法。下面的代码最早由 Ryan Dibble提供，此后经过多人Joerg.Schulenburg, R.Z.Liu 等修改，补正。

转自：http://forum.assuredigit.com/display_topic_threads.asp?ForumID=8&TopicID=3480

算法对输入的灰度图像的直方图进行分析，将直方图分成两个部分，使得两部分之间的距离最大。划分点就是求得的阈值。

parameter:   *image         --- buffer for image
              rows, cols     --- size of image
              x0, y0, dx, dy   --- region of vector used for computing threshold
              vvv             --- debug option, is 0, no debug information outputed
*/
/*======================================================================*/
/*   OTSU global thresholding routine                                 */
/*   takes a 2D unsigned char array pointer, number of rows, and     */
/*   number of cols in the array. returns the value of the threshold     */
/*======================================================================*/

#include "otsu.h"



int otsu (unsigned char *image, int rows, int cols, int x0, int y0, int dx, int dy, int vvv,int minimumGap)
{
unsigned char *np;     // 图像指针
int thresholdValue=1; // 阈值
int ihist[256];         // 图像直方图，256个点

int i, j, k;       // various counters
int n, n1, n2, gmin, gmax;
double m1, m2, sum, csum, fmax, sb;

// 对直方图置零...
memset(ihist, 0, sizeof(ihist));

gmin=255; gmax=0;
// 生成直方图
for (i = y0 + 1; i < y0 + dy - 1; i++) {
  np = &image[i*cols+x0+1];
  for (j = x0 + 1; j < x0 + dx - 1; j++) {
    ihist[*np]++;
    if(*np > gmax) gmax=*np;
    if(*np < gmin) gmin=*np;
    np++; /* next pixel */
  }
}

//如果太均衡，则判断为全白
if((gmax - gmin) < minimumGap)
	return -1;

// set up everything
sum = csum = 0.0;
n = 0;

for (k = 0; k <= 255; k++) {
  sum += (double) k * (double) ihist[k];   /* x*f(x) 质量矩*/
  n   += ihist[k];                           /* f(x)   质量   */
}

if (!n) {
  // if n has no value, there is problems...
  fprintf (stderr, "NOT NORMAL thresholdValue = 160\n");
  return (160);
}

// do the otsu global thresholding method
fmax = -1.0;
n1 = 0;
for (k = 0; k < 255; k++) {
  n1 += ihist[k];
  if (!n1) { continue; }
  n2 = n - n1;
  if (n2 == 0) { break; }
  csum += (double) k *ihist[k];
  m1 = csum / n1;
  m2 = (sum - csum) / n2;
  sb = (double) n1 *(double) n2 *(m1 - m2) * (m1 - m2);
  /* bbg: note: can be optimized. */
  if (sb > fmax) {
    fmax = sb;
    thresholdValue = k;
  }
}

// at this point we have our thresholding value

// debug code to display thresholding values
if ( vvv & 1 )
fprintf(stderr,"# OTSU: thresholdValue = %d gmin=%d gmax=%d\n",
  thresholdValue, gmin, gmax);

return(thresholdValue);
}
