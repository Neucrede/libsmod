#include "shapeinfo.h"
#include <iostream>

using namespace SMOD;

ShapeInfo::ShapeInfo(cv::Mat src, cv::Mat mask)
: angle_step(1), scale_step(0.1), eps(0.00001f)
{
    this->src = src;
    
    if(mask.empty()){
        // make sure we have masks
        this->mask = cv::Mat(src.size(), CV_8UC1, {255});
    }
    else{
        this->mask = mask;
    }
}

cv::Mat ShapeInfo::transform(cv::Mat src, float angle, float scale)
{
    cv::Mat dst;

    cv::Point2f center(src.cols/2.0f, src.rows/2.0f);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
    cv::warpAffine(src, dst, rot_mat, src.size());

    return dst;
}

void ShapeInfo::read(const cv::FileNode &fn)
{
    infos.clear();
    cv::FileNode fnInfos = fn["infos"];
    cv::FileNodeIterator it = fnInfos.begin(), it_end = fnInfos.end();
    for (; it != it_end; ++it)
    {
        infos.emplace_back(float((*it)["angle"]), float((*it)["scale"]));
    }
    
    valid_infos.clear();
    cv::FileNode fnValidInfos = fn["valid_infos"];
    cv::FileNodeIterator it2 = fnValidInfos.begin(), it2_end = fnValidInfos.end();
    for (; it2 != it2_end; ++it2)
    {
        valid_infos.emplace_back(float((*it2)["angle"]), float((*it2)["scale"]));
    }
}

void ShapeInfo::write(cv::FileStorage &fs) const
{
    fs << "infos"
       << "[";
       
    for (int i = 0; i < infos.size(); i++)
    {
        fs << "{";
        fs << "angle" << infos[i].angle;
        fs << "scale" << infos[i].scale;
        fs << "}";
    }
    
    fs << "]";
    
    fs << "valid_infos"
       << "[";
       
    for (int i = 0; i < infos.size(); i++)
    {
        if (i < valid_infos.size()) {
            fs << "{";
            fs << "angle" << valid_infos[i].angle;
            fs << "scale" << valid_infos[i].scale;
            fs << "}";
        }
    }
    
    fs << "]";
}

void ShapeInfo::produce()
{
    infos.clear();

    assert(angle_range.size() <= 2);
    assert(scale_range.size() <= 2);
    assert(angle_step > eps*10);
    assert(scale_step > eps*10);

    // make sure range not empty
    if(angle_range.size() == 0){
        angle_range.push_back(0);
    }
    if(scale_range.size() == 0){
        scale_range.push_back(1);
    }

    if (angle_range.size() == 1 && scale_range.size() == 1) {
        float angle = angle_range[0];
        float scale = scale_range[0];
        infos.emplace_back(angle, scale);
    }
    else if (angle_range.size() == 1 && scale_range.size() == 2) {
        assert(scale_range[1] > scale_range[0]);
        float angle = angle_range[0];
        for(float scale = scale_range[0]; scale <= scale_range[1]+eps; scale += scale_step) {
            infos.emplace_back(angle, scale);
        }
    }
    else if(angle_range.size() == 2 && scale_range.size() == 1) {
        assert(angle_range[1] > angle_range[0]);
        float scale = scale_range[0];
        for(float angle = angle_range[0]; angle <= angle_range[1]+eps; angle += angle_step) {
            infos.emplace_back(angle, scale);
        }
        std::sort(infos.begin(), infos.end());
    }
    else if(angle_range.size() == 2 && scale_range.size() == 2) {
        /*
        assert(scale_range[1] > scale_range[0]);
        assert(angle_range[1] > angle_range[0]);
        */
        
        if (scale_range[1] <= scale_range[0]) {
            scale_range[0] = scale_range[1];
            scale_range[1] = scale_range[0] + eps;
        }
        
        if (angle_range[1] <= angle_range[0]) {
            angle_range[0] = angle_range[1];
            angle_range[1] = angle_range[0] + eps;
        }
        
        for(float scale = scale_range[0]; scale <= scale_range[1]+eps; scale += scale_step) {
            std::vector<Info> tmp;
            for(float angle = angle_range[0]; angle <= angle_range[1]+eps; angle += angle_step) {
                tmp.emplace_back(angle, scale);
            }

            std::sort(tmp.begin(), tmp.end());
            infos.insert(infos.end(), tmp.begin(), tmp.end());
        }
    }
}

cv::Mat ShapeInfo::srcOf(const ShapeInfo::Info& info)
{
    return transform(src, info.angle, info.scale);
}

cv::Mat ShapeInfo::maskOf(const ShapeInfo::Info& info)
{
    return (transform(mask, info.angle, info.scale) > 0);
}
