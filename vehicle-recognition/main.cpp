#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <time.h>
#include <algorithm>
#include <map>
#include <iostream>




using namespace std;
using namespace cv;


void detectCars(cv::Mat &frame)
{
   // cv::dnn::Net netVehicle = cv::dnn::readNetFromTensorflow("vehicle_detect.pb","vehicle_detect.pbtxt");

    cv::dnn::Net netVehicle = cv::dnn::readNetFromCaffe("MobileNetSSD_deploy.prototxt","MobileNetSSD_deploy.caffemodel");
    cv::dnn::Net classifierVehicle = cv::dnn::readNetFromCaffe("ve_ce.prototxt","ve_ce_shrink.caffemodel");
    std::vector<cv::String> names = netVehicle.getLayerNames();
    for(auto name : names){
        std::cout<<name<<std::endl;
    }

    const char* classNames[] = {"background",
                                "aeroplane", "bicycle", "bird", "boat",
                                "bottle", "bus", "car", "cat", "chair",
                                "cow", "diningtable", "dog", "horse",
                                "motorbike", "person", "pottedplant",
                                "sheep", "sofa", "train", "tvmonitor"};
    Mat inputBlob = cv::dnn::blobFromImage(frame, 1/127.5, Size(300, 300), Scalar(104, 117, 123), false, false); //Convert Mat to batch of images
    netVehicle.setInput(inputBlob, "data"); //set the network input
    Mat detection = netVehicle.forward("detection_out"); //compute output
    cv::Mat detectionMat;
    detection.copyTo(detectionMat);
    std::cout<<detectionMat.size<<std::endl;

     for(int i = 0; i < detection.size.p[1]; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);
        std::cout<<confidence<<std::endl;
        if(confidence > 0.5)
        {
            size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
            Rect object(xLeftBottom, yLeftBottom,
                        xRightTop - xLeftBottom,
                        yRightTop - yLeftBottom);
            cv::Mat cropped ;
            frame(object).copyTo(cropped);
            float scale = cropped.rows/224.0;
            cv::resize(cropped,cropped,cv::Size(),scale,scale);
            cv::imshow("cropped",cropped);
            cv::waitKey(0);
            cv::Mat input_blob = cv::dnn::blobFromImage(cropped,1.0,cv::Size(cropped.cols,cropped.rows),cv::Scalar(0,0,0),true,false);
           // float* data=  (float*)input_blob;

            classifierVehicle.setInput(input_blob);
            cv::Mat output = classifierVehicle.forward();
            std::cout<<output.size<<std::endl;
            float* idx = (float*) output.data;

            int id = std::max_element(idx,idx+1776) - idx;
            float value = idx[id];
            std::cout<<"id:"<<id<<std::endl;
            std::cout<<"value:"<<value<<std::endl;
            cv::rectangle(frame,object,cv::Scalar(255,0,0),1);

        }
    }
    cv::imshow("frame",frame);
    cv::waitKey(0);



}
int main(){

    cv::Mat image = cv::imread("/Users/yujinke/Downloads/timg-59.jpeg");
    detectCars(image);

//    cv::Mat image1 = cv::imread("/Users/yujinke/Downloads/mulCar/蓝色_皖A5M621_3_1_22_1.jpg");
//    detectCars(image);

}