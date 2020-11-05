#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>              /* low-level i/o */
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/mman.h> // mmap

#include <linux/videodev2.h>

#include <opencv2/opencv.hpp>

#include <jpeglib.h>

static int xioctl(int fh, int request, void *arg)
{
    int r;
    do {
        r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno);
    return r;
}

int allocCamera(const char* file) 
{
    struct v4l2_capability cap;
    struct v4l2_crop crop;
    struct v4l2_format fmt;

    int camera_fd = open(file, O_RDWR);

    if (-1 == xioctl (camera_fd, VIDIOC_QUERYCAP, &cap)) {
        if (EINVAL == errno) {
            fprintf (stderr, "%s is no V4L2 device\n", file);
            exit (EXIT_FAILURE);
        } else {
            printf("\nError in ioctl VIDIOC_QUERYCAP\n\n");
            exit(0);
        }
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf (stderr, "%s is no video capture device\n", file);
        exit (EXIT_FAILURE);
    }

    /*if (!(cap.capabilities & V4L2_CAP_READWRITE)) {
        fprintf (stderr, "%s does not support read i/o\n", file);
        exit (EXIT_FAILURE);
    }*/

    memset(&fmt, 0, sizeof(fmt));
    fmt.type    = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    //fmt.fmt.pix.width       = 320; 
    //fmt.fmt.pix.height      = 240;
    //fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    //fmt.fmt.pix.field       = V4L2_FIELD_INTERLACED;

    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.width = 640;
    fmt.fmt.pix.height = 480;

    if (-1 == xioctl(camera_fd, VIDIOC_S_FMT, &fmt)) {
        printf("VIDIOC_S_FMT");
    }
    return camera_fd;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s device file_name.jpg\n", argv[0]);
        return 0;
    }

    int cam = allocCamera(argv[1]);

    struct v4l2_requestbuffers bufRequest;
    bufRequest.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bufRequest.memory = V4L2_MEMORY_MMAP;
    bufRequest.count = 1;

    if (ioctl(cam, VIDIOC_REQBUFS, &bufRequest) < 0) {
        perror("VIDIOC_REQBUFS");
        exit(EXIT_FAILURE);
    }

    struct v4l2_buffer bufferInfo;
    memset(&bufferInfo, 0, sizeof(bufferInfo));
    
    bufferInfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bufferInfo.memory = V4L2_MEMORY_MMAP;
    bufferInfo.index = 0;
    
    if(ioctl(cam, VIDIOC_QUERYBUF, &bufferInfo) < 0){
        perror("VIDIOC_QUERYBUF");
        exit(1);
    }

    void* bufferStart = mmap(nullptr, bufferInfo.length,
            PROT_READ | PROT_WRITE, MAP_SHARED, cam, bufferInfo.m.offset);

    if (bufferStart == MAP_FAILED) {
        perror("mmap");
        exit(EXIT_FAILURE);
    }

    memset(bufferStart, 0, bufferInfo.length);

    bufferInfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bufferInfo.memory = V4L2_MEMORY_MMAP;
    bufferInfo.index = 0;
    
    int type = bufferInfo.type;
    if (ioctl(cam, VIDIOC_STREAMON, &type) < 0) {
        perror("VIDIOC_STREAMON");
        exit(EXIT_FAILURE);
    }

    /* Here is where you typically start two loops:
    * - One which runs for as long as you want to
    *   capture frames (shoot the video).
    * - One which iterates over your buffers everytime. */

   char str[32];
    
    for (int i = 0; i < 1; ++i) {
        bufferInfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        bufferInfo.memory = V4L2_MEMORY_MMAP;

        // Put the buffer in the incoming queue.
        if(ioctl(cam, VIDIOC_QBUF, &bufferInfo) < 0){
            perror("VIDIOC_QBUF");
            exit(1);
        }

        // The buffer's waiting in the outgoing queue.
        if(ioctl(cam, VIDIOC_DQBUF, &bufferInfo) < 0){
            perror("VIDIOC_QBUF");
            exit(1);
        }

        //sprintf(str, "./my-image-%d.jpg", i);
        strcpy(str, argv[2]);

        if (int jpgFile = open(str, O_WRONLY | O_CREAT, 0660);
                jpgFile < 0) {
            perror("open jpgfile");
            exit(EXIT_FAILURE);
        }
        else {
            write(jpgFile, bufferStart, bufferInfo.length);
            close(jpgFile);
        }

        sleep(1);
    }    
    /* Your loops end here. */
    
    // Deactivate streaming
    if(ioctl(cam, VIDIOC_STREAMOFF, &type) < 0){
        perror("VIDIOC_STREAMOFF");
        exit(1);
    }

    close(cam);

    return 0;
}