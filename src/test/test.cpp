#include <opencv2/opencv.hpp>
#include "smod.h"
#include "timer.hxx"

void TestSMOD(int argc, char* argv[])
{
    using namespace SMOD;

    std::vector<int> T(2, 0);
    ShapeModelObjectDetectorEx smod(
        -180.0f, 180.0f, 0.1f, 
        0.9f, 1.1f, 0.1f, 
        true,
        512, 
        T, 
        25, 50, 360, 
        16 /* 8 */);

    // smod.SetDebug();
    smod.SetDebugImagePath("smod_result.png");
    smod.SetRefineAnisoScaling();
    smod.SetGVCompare();
    smod.SetGVCompareThreshold(50.0f);
    smod.SetMaxDistDiff(13);

    std::vector<std::string> classIds;
    std::vector<cv::Mat> imgs;
    for (int i = 1; i < argc - 1; ++i) {
        cv::Mat img = cv::imread(argv[i], cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            return;
        }
        else {
            classIds.push_back(argv[i]);
            imgs.push_back(img.clone());
        }
    }

    {
        Timer timer;
        smod.RegisterEx(classIds, imgs, true);
        timer.out("Register");
    }

    // smod.Save("model.yaml");
    // smod.Load("model.yaml");

    std::vector<Result> results;
    cv::Mat imgSrc = cv::imread(argv[argc - 1], cv::IMREAD_GRAYSCALE);
    if (imgSrc.empty()) return;

    {
        Timer timer;
        smod.Detect(imgSrc, results, 50, 100);
        timer.out("Detect");
    }

    char buf[256];
    for (int i = 0; i != results.size(); ++i) {
        std::cout << i << " " << results[i] << "\n";
        
        cv::Mat img1;
        smod.CropImage(imgSrc, img1, results[i], true, false);
        // smod.CropImage(imgSrc, img1, results[i], false, true);
        sprintf(buf, "roi_%d.png", i + 1);
        cv::imwrite(buf, img1);
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        return 1;
    }
    
    TestSMOD(argc, argv);

    return 0;
}
