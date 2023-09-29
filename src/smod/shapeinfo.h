#ifndef SHAPEINFO_H
#define SHAPEINFO_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

namespace SMOD {

class ShapeInfo {
public:
    class Info {
    public:
        float angle;
        float scale;

        Info(float angle_, float scale_)
        {
            angle = angle_;
            scale = scale_;
        }

        bool operator < (const Info& rhs) const 
        {
            return std::abs(angle) < std::abs(rhs.angle);
        }
    };
    
public:
    ShapeInfo(cv::Mat src = cv::Mat(), cv::Mat mask = cv::Mat());

    static cv::Mat transform(cv::Mat src, float angle, float scale);
    
    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;
    
    void produce();

    cv::Mat srcOf(const Info& info);
    cv::Mat maskOf(const Info& info);
    
public:
    cv::Mat src;
    cv::Mat mask;

    std::vector<float> angle_range;
    std::vector<float> scale_range;

    float angle_step;
    float scale_step;
    float eps;
    
    std::vector<Info> infos;
    std::vector<Info> valid_infos;
};

} // SMOD

#endif        //  #ifndef SHAPEINFO_H

