#ifndef NMSBOXES_H
#define NMSBOXES_H

#include <vector>
#include <opencv2/core/core.hpp>


void NMSBoxes(const std::vector<cv::Rect>& bboxes, const std::vector<float>& scores,
    const float score_threshold, const float nms_threshold,
    std::vector<int>& indices, const float eta=1, const int top_k=0);


#endif        //  #ifndef NMSBOXES_H

