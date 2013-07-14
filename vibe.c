#include <stdio.h>
#include <stdlib.h>

#include <glib.h>

#include "cv.h"
#include "highgui.h"


#define CACHE_RAND
//#define RGB_IMAGE  // SURPRISINGLY, it works way worse with full color information; perhaps metric issue (see getEuclideanDist)
#define DO_INITIAL_BLUR_HACK
#define DO_POSTPROCESSING

static const gint  nbSamples = 20;                   // number of samples per pixel
static const gint  reqMatches = 2;                   // #_min
static const guint radius = 20;                      // R
static const gint  subsamplingFactor = 16;           // amount of random subsampling

static guint8* samples = NULL;  // background model
static guint8* segmentationMap = NULL;  // foreground detection map

#ifdef CACHE_RAND  /* gives ~2x speedup */
static gfloat* _rand_cache = 0;
static const int _rand_cache_size = 65535 * 1024;
static int _rand_cache_ptr = 0;
#endif /* CACHE_RAND */

static gint _width, _height;
static const guint _channels = 
#ifdef RGB_IMAGE
	3;
#else  /* RGB_IMAGE */
	1; /* grayscale image */
#endif  /* RGB_IMAGE */

#ifndef NDEBUG
# define BEGIN_TIMING { gint64 _begin = g_get_monotonic_time();
# define END_TIMING(a) fprintf(stderr, "%s took %.4fs\n", a, (g_get_monotonic_time()-_begin)/1000000.); }
# define STOP_TIMING(a) (void)_begin;}
#else
# define BEGIN_TIMING {
# define END_TIMING(a) }
# define STOP_TIMING(a) }
#endif  /* NDEBUG */

#define SQUARE(x) ((x)*(x))

inline guint fast_abs(const gint8 x)
{
	const gint8 mask = x >> (sizeof(gint8) * CHAR_BIT - 1);
	return (x + mask) ^ mask;
}

inline guint getEuclideanDist(const guint8* im1, const guint8* im2)
{
#ifdef RGB_IMAGE
	const gint32 p1 = *(gint32*)im1;
	const gint32 p2 = *(gint32*)im2;
	const gint8 rdiff = (p1 & 0xff)         - (p2 & 0xff);
	const gint8 gdiff = ((p1 >> 8)  & 0xff) - ((p2 >> 8)  & 0xff);
	const gint8 bdiff = ((p1 >> 16) & 0xff) - ((p2 >> 16) & 0xff);

	// surprisingly, Manhattan distance performs better
	return fast_abs(rdiff) + fast_abs(gdiff) + fast_abs(bdiff);
	//return SQUARE(rdiff) + SQUARE(gdiff) + SQUARE(bdiff);
#else  /* RGB_IMAGE */
	return fast_abs((gint8)*im1 - *im2);
#endif /* RGB_IMAGE */
}

inline static void setPixelBackground(gint x, gint y)
{
	if(!segmentationMap)
	{
		segmentationMap = g_malloc(_width*_height);
		memset(segmentationMap, 0, _width*_height);
	}
	segmentationMap[y*_width+x] = 0;
}

inline static void setPixelForeground(gint x, gint y)
{
	if(!segmentationMap)
	{
		segmentationMap = g_malloc(_width*_height);
		memset(segmentationMap, 0, _width*_height);
	}
	segmentationMap[y*_width+x] = 1;
}

inline static int getRandomNumber(gint start, gint end)
{
	gfloat r =
#ifdef CACHE_RAND
		_rand_cache[_rand_cache_ptr++];
	if(_rand_cache_ptr == _rand_cache_size)
	{
		fprintf(stderr, "Random cache empty!!!\n");
		_rand_cache_ptr = 0;
	}
#else  /* CACHE_RAND */
		(gfloat)rand() / RAND_MAX;
#endif  /* CACHE_RAND */
	return round(start + (end-start) * r);
}

void chooseRandomNeighbor(gint x, gint y, gint* neighborX, gint* neighborY)
{
	gint deltax = 0, deltay = 0;
	while(deltax == 0 && deltay == 0)
	{
		deltax = getRandomNumber(x == 0 ? 0 : -1, x == _width-1  ? 0 : 1);
		deltay = getRandomNumber(y == 0 ? 0 : -1, y == _height-1 ? 0 : 1);
	}
	*neighborX = x + deltax;
	*neighborY = y + deltay;
	if(*neighborX < 0)
		*neighborX = 0;
	if(*neighborY < 0)
		*neighborY = 0;
}

void plugin_stream_process(guint streamnum, guint64 timestamp,
	guint64 duration, const guint8* image, guint size)
{
	(void)streamnum; (void)timestamp; (void)duration; (void)size;

	gint x, y, index;

	if(!samples)
	{
		fprintf(stderr, "allocating!\n");
		samples = g_malloc(_width*_height*nbSamples*_channels);
		for(index = 0; index < nbSamples; index++)
			memcpy(
				&samples[index*_width*_height*_channels],
				image,
				_width*_height*_channels);
	}

	for(x = 0; x < _width; x++)
	{
		for(y = 0; y < _height; y++)
		{
			// comparison with the model
			int count = 0;
			index = 0;
			const guint8* imgptr = &image[(y*_width+x)*_channels];
			guint8* sptr   = &samples[index*_width*_height*_channels+(y*_width+x)*_channels];
			while((count < reqMatches) && (index < nbSamples))
			{
				guint distance = getEuclideanDist(imgptr, sptr);
				if(distance < radius)
					count++;
				index++;
				sptr += _width*_height*_channels;
			}

			// pixel classification according to reqMatches
			if(count >= reqMatches)
			{
				// the pixel belongs to the background
				// stores the result in the segmentation map
				setPixelBackground(x, y);
				// gets a random number between 0 and subsamplingFactor-1
				int randomNumber = getRandomNumber(0, subsamplingFactor-1);
				// update of the current pixel model
				if(randomNumber == 0)
				{
					// random subsampling
					// other random values are ignored
					randomNumber = getRandomNumber(0, nbSamples-1);
					memcpy(
						&samples[randomNumber*_width*_height*_channels+(y*_width+x)*_channels],
						&image[(y*_width+x)*_channels],
						sizeof(guint8)*_channels
					);
				}
				// update of a neighboring pixel model
				randomNumber = getRandomNumber(0, subsamplingFactor-1);
				if(randomNumber == 0)
				{
					// random subsampling
					// chooses a neighboring pixel randomly
					int neighborX; int neighborY;
					chooseRandomNeighbor(x, y, &neighborX, &neighborY);
					// chooses the value to be replaced randomly
					randomNumber = getRandomNumber(0, nbSamples-1);
					memcpy(
						&samples[randomNumber*_width*_height*_channels+(neighborY*_width+neighborX)*_channels],
						&image[(y*_width+x)*_channels],
						sizeof(guint8)*_channels
					);
				}
			}
			else // the pixel belongs to the foreground
				// stores the result in the segmentation map
				setPixelForeground(x, y);
		}
	}
}

int main(int argc, char** argv)
{
	if(argc != 2)
	{
		fprintf(stderr, "USAGE: %s <video capture source>\n", argv[0]);
		return EXIT_FAILURE;
	}

	CvCapture* capture = cvCreateFileCapture(argv[1]);
	if(!capture)
	{
		fprintf( stderr, "ERROR: capture is NULL \n" );
		return EXIT_FAILURE;
	}
	// Create a window in which the captured images will be presented
	cvNamedWindow("video", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("result", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("background model", CV_WINDOW_AUTOSIZE);

	BEGIN_TIMING;
	srand(g_get_real_time());
#ifdef CACHE_RAND
	int k;
	_rand_cache = g_malloc(_rand_cache_size*sizeof(_rand_cache[0]));
	for(k = 0; k < _rand_cache_size; k++)
		_rand_cache[k] = (gfloat)rand() / RAND_MAX;
#endif  /* CACHE_RAND */
	END_TIMING("initialization");

	guint8* buffer = NULL;
	gint i, j, size = 0;
	IplImage *img = NULL, *res = NULL, *back = NULL;

	// Show the image captured from the camera in the window and repeat
	while(1)
	{
		// Get one frame
		IplImage* frame = cvQueryFrame(capture);
		if(!frame)
		{
			fprintf(stderr, "ERROR: frame is null!\n");
			break;
		}

		_width = frame->width; _height = frame->height;
		if(!img)
			img = cvCreateImage(cvSize(_width, _height), IPL_DEPTH_8U, _channels);
		if(!res)
			res = cvCreateImage(cvSize(_width, _height), IPL_DEPTH_8U, 1);
		if(!back)
			back = cvCreateImage(cvSize(_width, _height), IPL_DEPTH_8U, _channels);

#ifndef RGB_IMAGE
		cvCvtColor(frame, img, CV_RGB2GRAY);
		cvConvertScale(img, img, 1.0, 0);
#else  /* !RGB_IMAGE */
		cvConvertScale(frame, img, 1.0, 0);
#endif  /* !RGB_IMAGE */


#ifdef DO_INITIAL_BLUR_HACK
		static gboolean _initial_frame = 1;
		if(_initial_frame)
		{
			printf("Blurring!\n");
			_initial_frame = 0;
			cvSmooth(img, img, CV_MEDIAN, 3, 3, 0, 0);
		}
#endif  /* DO_INITIAL_BLUR_HACK */

		BEGIN_TIMING;
		if(!buffer)
		{
			size = _width * _height * _channels;
			buffer = g_malloc(size);
		}
		for(i = 0; i < _height; i++)
			for(j = 0; j < _width; j++)
			{
#ifdef RGB_IMAGE
				buffer[3*(i*_width+j)+0] =
					((uchar*)(img->imageData+i*img->widthStep))[j*img->nChannels+2];
				buffer[3*(i*_width+j)+1] =
					((uchar*)(img->imageData+i*img->widthStep))[j*img->nChannels+1];
				buffer[3*(i*_width+j)+2] =
					((uchar*)(img->imageData+i*img->widthStep))[j*img->nChannels+0];
#else  /* RGB_IMAGE */
				buffer[i*_width+j] = ((uchar*)(img->imageData+i*img->widthStep))[j*img->nChannels];
#endif  /* RGB_IMAGE */
			}
		STOP_TIMING("buffer copying");

		BEGIN_TIMING;
		plugin_stream_process(0, 0, 0, buffer, size);
		END_TIMING("processing");

		BEGIN_TIMING;
		for(i = 0; i < _height; i++)
			for(j = 0; j < _width; j++)
			{
				guint8* ptr = &samples[_width*_height*_channels+(i*_width+j)*_channels];
#ifdef RGB_IMAGE
				((uchar*)(back->imageData+i*back->widthStep))[j*back->nChannels+2] =
					*(ptr++);
				((uchar*)(back->imageData+i*back->widthStep))[j*back->nChannels+1] =
					*(ptr++);
				((uchar*)(back->imageData+i*back->widthStep))[j*back->nChannels+0] =
					*(ptr++);
#else  /* RGB_IMAGE */
				((uchar*)(back->imageData+i*back->widthStep))[j*back->nChannels] = *ptr;
#endif  /* RGB_IMAGE */

				((uchar*)(res->imageData+i*res->widthStep))[j*res->nChannels] =
					segmentationMap[i*_width+j] * 255;
			}
		STOP_TIMING("result copying");

#ifdef DO_POSTPROCESSING
		BEGIN_TIMING;
		static IplConvKernel* morphology_kernel = NULL;
		if(!morphology_kernel)
		{
			morphology_kernel =
				cvCreateStructuringElementEx(5, 5, 2, 2, 0/*MORPH_RECT*/, NULL);
		}
		/* sorta morph open */
		cvMorphologyEx(res, res, NULL, morphology_kernel, CV_MOP_ERODE, 2);
		cvMorphologyEx(res, res, NULL, morphology_kernel, CV_MOP_DILATE, 8);


		static CvMemStorage* st = NULL;
		if(!st)
			st = cvCreateMemStorage(0);
		else
			cvClearMemStorage(st);
		CvSeq* contours = NULL;
		IplImage* im = cvCloneImage(res);
		cvFindContours(im, st, &contours, sizeof(CvContour), CV_RETR_LIST,
			CV_CHAIN_APPROX_SIMPLE/*CV_CHAIN_APPROX_NONE*/, cvPoint(0, 0));

		CvSeq* contour;
		for(contour = contours; contour; contour = contour->h_next)
		{
			CvRect r = cvBoundingRect(contour, 0);
			cvRectangle(frame, cvPoint(r.x, r.y), cvPoint(r.x+r.width, r.y+r.height),
				CV_RGB(0,255,0), 1, 8, 0);
		}

		END_TIMING("postprocessing");
#endif  /* DO_POSTPROCESSING */

		cvShowImage("video", frame);
		cvShowImage("result", res);
		cvShowImage("background model", back);


		if((cvWaitKey(10*50) & 0xff) == 27) break;
	}
	g_free(samples);
	g_free(segmentationMap);
	g_free(buffer);
#ifdef CACHE_RAND
	g_free(_rand_cache);
#endif  /* CACHE_RAND */

	// Release the capture device housekeeping
	cvReleaseCapture(&capture);
	cvDestroyAllWindows();

	return EXIT_SUCCESS;
}

