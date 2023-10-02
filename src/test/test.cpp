#include <opencv2/opencv.hpp>
#include "smod.h"
#include "timer.hxx"

void TestSMOD(int argc, char* argv[])
{
    using namespace SMOD;

    std::vector<int> T(2, 0);
    ShapeModelObjectDetector smod(-180, 180, 1, 0.9, 1.1, 0.1, true,
        8192, T, 30, 50, 360, 16);
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    smod.SetDebug();
    smod.SetDebugImagePath("smod_result.png");
    smod.Register(img, cv::Mat(), true);
    smod.Save("model.yaml");
    
    /*
    ShapeModelObjectDetector smod;
    smod.Load("model.yaml");
    smod.SetDebug();
    smod.SetDebugImagePath("smod_debug.png");
    */

    std::vector<Result> results;
    cv::Mat imgSrc = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    if (imgSrc.empty()) return;

    smod.Detect(imgSrc, results, 75, 5);

    if (!results.empty()) {
        cv::Mat imgCrop;
        smod.CropImage(imgSrc, imgCrop, results[0]);
        cv::imwrite("A.png", imgCrop);
    }
}

int main(int argc, char* argv[])
{
    using namespace SMOD;
    
    
    TestSMOD(argc, argv);

    /*
    using namespace SMOD;

    ShapeModelObjectDetector smod(-45, 45, 0.5, 0.8, 1.2, 0.05, true,
        256, { 4, 8, 16, 32 }, 50, 100);
    cv::Mat img = cv::imread(argv[1]);
    Result result(cv::Point2f(3209, 1457), cv::Size2f(2000, 1500),
        atof(argv[2]), 1.0, 90.0);
    cv::Mat imgCrop;
    smod.CropImage(img, imgCrop, result);
    cv::imwrite("smod_crop.png", imgCrop);
    */
    
    /*
    std::vector<int> T(2, 0);
    ShapeModelObjectDetector smod(0, 0, 1.0, 1.0, 1.1, 10, true,
        256, T, 20, 50);
    cv::Mat img = cv::imread(argv[1]);
    
    cv::Mat img2;
    smod.BilateralFilter(img, img2, 29, 100, 100);
    cv::imwrite("img2.png", img2);
    */

    return 0;
}
