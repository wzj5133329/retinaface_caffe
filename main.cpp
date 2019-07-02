#include"detect.h"

using namespace std;

int main(int argc, char** argv) {

    const string proto="mnet.prototxt";
    const string model="mnet.caffemodel";
    const float confidence=0.5;
    const float nms_threshold = 0.4;
    const string cpu_mode="gpu";
    Detector detector(proto,model,confidence,nms_threshold,cpu_mode);
    
    cv::VideoCapture video_frame;
    //video_frame.open(0);
    cv::Mat img;
    //while(video_frame.read(img))
    while(true)
    {
       
        //cv::Mat input = img.clone();
        img=cv::imread("demo.jpg");
        cout<<"channels"<<img.channels()<<endl;
        //cv::resize(img,img,cv::Size(1920,1080));
        cout<<"img10"<<img.rows<<endl;
        uint64_t time1=current_timestamp();
        std::vector<Anchor> result = detector.Detect(img);       
        for(int i = 0; i < result.size(); i ++) {
            cv::rectangle(img, cv::Point((int)result[i].finalbox.x, (int)result[i].finalbox.y), cv::Point((int)result[i].finalbox.width, (int)result[i].finalbox.height), cv::Scalar(0, 255, 255), 2, 8, 0);
            
            cv::circle(img, result[i].pts[0], 10, cv::Scalar(0, 255, 0),-1);
            cv::circle(img, result[i].pts[1], 10, cv::Scalar(0, 255, 0),-1);
            cv::circle(img, result[i].pts[2], 10, cv::Scalar(0, 255, 0),-1);
            cv::circle(img, result[i].pts[3], 10, cv::Scalar(0, 255, 0),-1);
            cv::circle(img, result[i].pts[4], 10, cv::Scalar(0, 255, 0),-1);
        }

        uint64_t time2=current_timestamp();
        cout<<"time:"<<time2-time1<<endl;
        cv::namedWindow("show",CV_WINDOW_NORMAL);
        cv::resizeWindow("show",960,540);
        cv::imshow("show",img);
        cv::waitKey(1);
        cv::imwrite("result.jpg", img);
    }
    return 0;
}
