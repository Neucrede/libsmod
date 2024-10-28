#include "smod.h"

using namespace SMOD;

Result::Result()
: scale(0), scaleX(0), scaleY(0), score(0), scorePearson(-1)
{
}

Result::Result(const cv::Point2f& centre, const cv::Size2f& size, float angle,
    float _scale, float _score, const std::string& _classId)
: region(centre, size, angle), scale(_scale), score(_score), 
    scaleX(_scale), scaleY(_scale), scorePearson(-1), classId(_classId)
{
}

Result::Result(const cv::Point2f& centre, const cv::Size2f& size, float angle,
    float _scaleX, float _scaleY, float _score, const std::string& _classId)
: region(centre, size, angle), scale(0.5f * (_scaleX + _scaleY)), score(_score), 
    scaleX(_scaleX), scaleY(_scaleY), scorePearson(-1), classId(_classId)
{
}

std::ostream& SMOD::operator << (std::ostream& os, const Result& result)
{
    const cv::RotatedRect& region = result.region;
    
    return 
        os  << "Class: " << result.classId
            << ", Region: { Center: " << region.center
            << ", Angle = " << region.angle << " }"
            << ", Average scale = " << result.scale
            << ", X scale = " << result.scaleX
            << ", Y scale = " << result.scaleY
            << ", Score = " << result.score 
            << ", Pearson coeff = " << result.scorePearson;
}

