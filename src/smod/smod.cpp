#include <stdexcept>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <tuple>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/Core>
#include <Eigen/QR>
#include <stdlib.h>
#include <omp.h>
#include "smod.h"
#include "nmsboxes.h"
#include "icp.h"
#include "timer.hxx"

namespace SMOD
{
    const char* defaultClassId = "CLASS1";
}

using namespace SMOD;
using namespace Line2Dup;

static int GetOMPNumThreads();

ShapeModelObjectDetectorBase::ShapeModelObjectDetectorBase( 
    float angleMin, float angleMax, float angleStep,
    float scaleMin, float scaleMax, float scaleStep, bool refine,
    int numFeatures, const std::vector<int>& T, float weakThreshold, 
    float strongThreshold, float maxGradient, int numOri)
: m_detector(numFeatures, T, weakThreshold, strongThreshold, maxGradient, numOri),
    m_debug(false), m_refine(refine), m_gvCompareThreshold(75.0f),
    m_enableGVCompare(false), m_refineAnisoScaling(false), m_maxDistDiff(5.0f),
    m_maxOverlap(0.5f), m_nmsDampFactor(0.9f)
{
    m_shapeInfo0.angle_range = { angleMin, angleMax };
    m_shapeInfo0.scale_range = { scaleMin, scaleMax };
    m_shapeInfo0.angle_step = angleStep;
    m_shapeInfo0.scale_step = scaleStep;
}

ShapeModelObjectDetectorBase::~ShapeModelObjectDetectorBase()
{
}

int ShapeModelObjectDetectorBase::RegisterEx(const std::vector<std::string>& classIds,
    const std::vector<cv::Mat>& imgs, bool fast, const std::vector<cv::Mat>& masks)
{
    int N = classIds.size();
    if (N != imgs.size()) {
        throw std::invalid_argument("classIds and imgs should contain same number of elements.");
    }

    bool hasMasks = !masks.empty();
    if (hasMasks && (N != masks.size())) {
        throw std::invalid_argument("masks and imgs should contain same number of elements.");
    }

    for (const std::string& s : classIds) {
        m_detector.pruneClassTemplate(s);
        auto it = m_tmplInfos.find(s);
        if (it != m_tmplInfos.end()) {
            m_tmplInfos.erase(it);
        }
    }

    if (m_debug) {
        std::cout << "Debug print is enabled. Limited to single worker thread." << std::endl;
    } 

    int numThreads = (!fast || m_debug) ? 1 : GetOMPNumThreads();

    typedef std::tuple<std::vector<Line2Dup::TemplatePyramid>, TemplateInfo> TpInfoPair;
    std::vector<TpInfoPair> tpInfos;

    #pragma omp declare reduction ( \
        omp_insert  \
        : std::vector<TpInfoPair>   \
        : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

    #pragma omp parallel for reduction(omp_insert: tpInfos) num_threads(numThreads) 
    for (int i = 0; i < N; ++i) {
        std::vector<Line2Dup::TemplatePyramid> tps;
        TemplateInfo tmplInfo;
        if (RegisterInternal(imgs[i], hasMasks ? masks[i] : cv::Mat(), fast, classIds[i],
            tps, tmplInfo))
        {
            tpInfos.emplace_back(tps, tmplInfo);
        }
    }

    for (const auto& tpInfo : tpInfos) {
        const auto& tps = std::get<0>(tpInfo);
        const auto& tmplInfo = std::get<1>(tpInfo);
        m_tmplInfos[tmplInfo.classId] = tmplInfo;

        for (const auto& tp : tps) {
            m_detector.addTemplate(tmplInfo.classId, tp);
        }
    }

    return tpInfos.size();
}

bool ShapeModelObjectDetectorBase::RegisterInternal(const cv::Mat& img, const cv::Mat& mask, bool fast, 
    const std::string& classId, std::vector<Line2Dup::TemplatePyramid>& tps, TemplateInfo& tmplInfo)
{
    if (img.empty()) {
        throw std::invalid_argument("Image is empty.");
    }
    
    cv::Mat imgGray;
    if (img.channels() == 1) {
        imgGray = img;
    }
    else if (img.channels() == 3) {
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    }
    else {
        throw std::runtime_error("img: image channels must be 1 or 3.");
    }
    
    std::string clazzId = classId;
    if (clazzId.empty()) {
        clazzId = defaultClassId;
    }
    
    Timer timer;

    tmplInfo.classId = clazzId;
    tmplInfo.size = imgGray.size();
    tmplInfo.shapeInfo = m_shapeInfo0;
    
    int dw, dh;
    ComputePaddings(img, dw, dh);
    tmplInfo.dw = dw;
    tmplInfo.dh = dh;
    
    tmplInfo.imgTmpl = imgGray.clone();
    ComputeMeanSqsum(tmplInfo.imgTmpl, tmplInfo.mean, tmplInfo.sqsum);

    cv::Mat imgPadded;
    cv::copyMakeBorder(imgGray, imgPadded, dh, dh, dw, dw, 
        cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat mask1, maskPadded;
    if (!mask.empty()) {
        mask1 = mask;
    }
    else {
        mask1 = cv::Mat(img.size(), CV_8UC1, cv::Scalar::all(255));
    }
    cv::copyMakeBorder(mask1, maskPadded, dh, dh, dw, dw, 
        cv::BORDER_CONSTANT, cv::Scalar::all(0));

    ShapeInfo& shapeInfo = tmplInfo.shapeInfo;
    shapeInfo.src = imgPadded;
    shapeInfo.mask = maskPadded;
    shapeInfo.produce();
    shapeInfo.valid_infos.reserve(shapeInfo.infos.size() / 2);

    tps.clear();

    if (!fast) {        
        typedef std::tuple<Line2Dup::TemplatePyramid, ShapeInfo::Info> TpInfoPair;
        std::vector<TpInfoPair> tpInfos;
        tpInfos.reserve(shapeInfo.infos.size());

        int numThreads = (m_debug || fast) ? 1 : GetOMPNumThreads();
        
        #pragma omp declare reduction ( \
            omp_insert  \
            : std::vector<TpInfoPair>   \
            : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

        #pragma omp parallel for reduction(omp_insert: tpInfos) num_threads(numThreads) 
        for (int i = 0; i < shapeInfo.infos.size(); ++i) {
            TemplatePyramid tp;
            const ShapeInfo::Info &info = shapeInfo.infos[i];
            
            if (m_debug) {
                std::cout   << "Angle = " << info.angle 
                            << ", Scale = " << info.scale 
                            << std::endl;
            }
            
            bool ok = m_detector.makeTemplate(shapeInfo.srcOf(info), 
                shapeInfo.maskOf(info), tp);
            if (ok) {
                tpInfos.emplace_back(tp, info);
            }
        }

        for (auto tpInfo : tpInfos) {
            shapeInfo.valid_infos.push_back(std::get<1>(tpInfo));
            tps.push_back(std::get<0>(tpInfo));
        }
    }
    else {
        const float maxFastRegAngleDiff = 5.0f;
        const cv::Point2f centre(shapeInfo.src.cols / 2.0f, shapeInfo.src.rows / 2.0f);
        float scale0 = -1, angle0;
        TemplatePyramid tp0, tp;

        for (int i = 0; i < shapeInfo.infos.size(); ++i) {
            const ShapeInfo::Info &info = shapeInfo.infos[i];
        
            if (info.scale != scale0) {
                if (m_debug) {
                    std::cout << "[Precise] ";
                }

                bool ok = m_detector.makeTemplate(shapeInfo.srcOf(info),  
                    shapeInfo.maskOf(info), tp0);
                if (ok) {
                    tps.push_back(tp0);
                    shapeInfo.valid_infos.push_back(info);
                    scale0 = info.scale;
                    angle0 = info.angle;
                }
                else {
                    scale0 = -1;
                }
            }
            else {
                // More accurate.
                if (std::abs(info.angle - angle0) >= maxFastRegAngleDiff) {
                    // Force using precise registration method.
                    scale0 = -1;
                    --i;
                    continue;
                }

                if (m_debug) {
                    std::cout << "[Fast] ";
                }

                bool ok = m_detector.makeTemplateRotate(tp0, info.angle - angle0,
                    centre, tp);
                
                if (m_debug) {
                    std::cout << "Computed from reference features." << std::endl;
                }
                
                if (ok) {
                    tps.push_back(tp);
                    shapeInfo.valid_infos.push_back(info);
                }
            }

            if (m_debug) {
                std::cout   << "Angle = " << info.angle 
                    << ", Scale = " << info.scale 
                    << std::endl;
            }
        }
    }
    
    if (m_debug) {
        timer.out("REGISTER: Time cost");
    }

    return !shapeInfo.valid_infos.empty();
}

void ShapeModelObjectDetectorBase::InitDetect(const cv::Mat& src, int maxNumMatches,
    cv::Mat& srcGray, cv::Mat& srcPadded,  cv::Mat& imgDbg, int& maxNumMatches1, 
    cv::Point& offset) const
{
    if (src.empty()) {
        throw std::invalid_argument("Source image is empty.");
    }
    
    if (src.channels() == 1) {
        srcGray = src;
    }
    else if (src.channels() == 3) {
        cv::cvtColor(src, srcGray, cv::COLOR_BGR2GRAY);
    }
    else {
        throw std::runtime_error("src: image channels must be 1 or 3.");
    }

    int max_dw = 0, max_dh = 0;
    for (const auto& pair : m_tmplInfos) {
        if (pair.second.dw > max_dw) {
            max_dw = pair.second.dw;
        }

        if (pair.second.dh > max_dh) {
            max_dh = pair.second.dh;
        }
    }
    offset = cv::Point(max_dw, max_dh);

    const int stride = std::max(
        (int)(*std::max_element(m_detector.T_at_level.begin(), 
            m_detector.T_at_level.end()) * std::pow(2, m_detector.T_at_level.size() - 1)),
        16);
    const int srcPaddedWidth = std::ceil((float)(src.cols + 2 * max_dw) / (float)stride) * stride;
    const int srcPaddedHeight = std::ceil((float)(src.rows + 2 * max_dh) / (float)stride) * stride;
    srcPadded = cv::Mat(srcPaddedHeight, srcPaddedWidth, srcGray.type(), cv::Scalar::all(0));
    srcGray.copyTo(srcPadded(cv::Rect(max_dw, max_dh, srcGray.cols, srcGray.rows)));

    if (m_debug) {
        if (src.channels() == 1) {
            cv::cvtColor(src, imgDbg, cv::COLOR_GRAY2BGR);
        }
        else {
            imgDbg = src.clone();
        }
    }

    if (maxNumMatches <= 0) {
        maxNumMatches = 1;
    }
    
    maxNumMatches1 = maxNumMatches;
    if ((maxNumMatches > 65535) || (m_enableGVCompare)) {
        maxNumMatches1 = 65535;
    }
}

bool ShapeModelObjectDetectorBase::Detect(const cv::Mat& src, 
    std::vector<Result>& results, float threshold, int maxNumMatches,
    const std::vector<std::string>& classIds, const cv::Mat& mask) const
{
    results.clear();
    results.reserve(maxNumMatches);

    Timer timer, timer0;

    int maxNumMatches1;
    cv::Mat srcGray, srcPadded, imgDbg;
    cv::Point offset;
    InitDetect(src, maxNumMatches, srcGray, srcPadded, imgDbg, maxNumMatches1,
        offset);

    if (m_debug) {
        timer.out("DETECT: Initialization");
    }

    cv::Ptr<Line2Dup::ColorGradientPyramid> quantizer;
    Detector::MatchesMap mm
        = m_detector.match(srcPadded, threshold, quantizer, classIds, mask, maxNumMatches1,
            m_maxOverlap);

    if (m_debug) {
        timer.out("DETECT: Matching");
    }
    
    if (mm.empty()) {
        return false;
    }

    std::vector<Line2Dup::Match> goodMatches;
    goodMatches.reserve(maxNumMatches1);

    for (auto& pair : mm) {
        std::vector<Line2Dup::Match>& matches = pair.second;

        std::vector<cv::Rect> boundBoxes;
        std::vector<float> _scores;
        for (const Line2Dup::Match& match : matches) {
            const std::vector<Line2Dup::Template>& tmpls 
                = m_detector.getTemplates(match.class_id, match.template_id);
            if (tmpls.empty()) {
                continue;
            }

            const auto& it = m_tmplInfos.find(match.class_id);
            if (it == m_tmplInfos.cend()) {
                continue;
            }

            const TemplateInfo& tmplInfo = it->second;
        
            // boundBoxes.emplace_back(match.x, match.y, tmpls[0].width, tmpls[0].height);
            boundBoxes.emplace_back(match.x, match.y, tmplInfo.size.width, tmplInfo.size.height);
            _scores.push_back(match.similarity);
        }

        std::vector<int> indices;    
        ::NMSBoxes(boundBoxes, _scores, threshold, m_maxOverlap, m_maxOverlap, indices,
            m_nmsDampFactor);

        for (int idx : indices) {
            goodMatches.push_back(matches[idx]);
        }
    }

    if (m_debug) {
        timer.out("DETECT: Non-maximum supression");
    }
    
    Scene_edge scene;
    std::vector<::Vec2f> pcdBuf, normalBuf;
    if (m_refine) {
        cv::Mat Dx, Dy;
        quantizer->sobel_dx0.convertTo(Dx, CV_16S, 128);
        quantizer->sobel_dy0.convertTo(Dy, CV_16S, 128);
        
        scene.init_Scene_edge_cpu(
            srcPadded, pcdBuf, normalBuf, 
            m_maxDistDiff,
            0.75 * m_detector.modality->weak_threshold, 
            0.75 * m_detector.modality->strong_threshold,
            Dx, Dy
        );
        
        if (m_debug) {
            timer.out("DETECT: Initialize scene");
        }
    }
    
    const int N = goodMatches.size();
    int numThreads = m_debug ? 1 : GetOMPNumThreads();
    int resultResvSize = (((maxNumMatches / numThreads) >> 1) << 1);
    if (resultResvSize < 1) {
        resultResvSize = 2;
    }

    if (m_debug) {
        std::cout << "Debug print is enabled. Limited to single worker thread." << std::endl;
    }

    #pragma omp declare reduction ( \
            omp_insert  \
            : std::vector<Result>   \
            : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

    #pragma omp parallel num_threads(numThreads)
    {
        #pragma omp for reduction(omp_insert: results)
        for (int k = 0; k < N; ++k) {
            const Line2Dup::Match& match = goodMatches[k];
            const std::vector<Line2Dup::Template>& tmpls
                = m_detector.getTemplates(match.class_id, match.template_id);
            const Line2Dup::Template& tmpl0 = tmpls[0];
            
            const auto it = m_tmplInfos.find(match.class_id);
            if (it == m_tmplInfos.end()) {
                continue;
            }
            const TemplateInfo& tmplInfo = it->second; 
            const ShapeInfo::Info& info = tmplInfo.shapeInfo.valid_infos[match.template_id];

            results.reserve(resultResvSize);            

            cv::Point2f centre(
                match.x - tmpl0.tl_x + (float)(tmplInfo.size.width) / 2.0f + tmplInfo.dw - offset.x,
                match.y - tmpl0.tl_y + (float)(tmplInfo.size.height) / 2.0f + tmplInfo.dh - offset.y
            );
            cv::Size2f rSize(
                (float)(tmplInfo.size.width),
                (float)(tmplInfo.size.height)
            );
            
            Result result0(centre, rSize, -info.angle, info.scale, match.similarity,
                match.class_id);

            if (m_enableGVCompare) {
                cv::Mat imgCandidate;
                CropImage(srcGray, imgCandidate, result0, true, false);
                if (imgCandidate.size() != tmplInfo.size) {
                    cv::resize(imgCandidate, imgCandidate, tmplInfo.size, 0, 0, cv::INTER_NEAREST);
                }
                            
                float pearson = ComputePearson(imgCandidate, tmplInfo);
                if (m_debug) {
                    std::cout << k + 1 << " GVCompare: Pearson correlation coefficient = " << pearson << std::endl;
                }
                
                if (pearson < m_gvCompareThreshold) {
                    if (m_debug) {
                        std::cout << k + 1 << " GVCompare: SKIP" << std::endl;
                    }
                    continue;
                }
                else if (m_debug) {
                    std::cout << k + 1 << " GVCompare: ACCEPT" << std::endl;
                }
            
                result0.scorePearson = pearson;
            }
            
            results.push_back(result0);
            Result& result = results.back();
            
            if (m_debug) {
                std::cout << k + 1 << " [Original] " << result0 << std::endl;
                for (const Line2Dup::Feature& feature : tmpl0.features) {
                    cv::circle(imgDbg, 
                        { feature.x + match.x - offset.x, feature.y + match.y - offset.y},
                        2, cv::Scalar(255, 0, 0), cv::FILLED);
                }
            }
            
            if (m_refine) {
                std::vector<::Vec2f> modelPCD;
                modelPCD.reserve(tmpl0.features.size());
                for (const Line2Dup::Feature& feature : tmpl0.features) {
                    modelPCD.emplace_back(feature.x + match.x, feature.y + match.y);
                }
                
                icp::RegistrationResult regResult;
                if (m_refineAnisoScaling) {
                    regResult = icp::ICP2D_Point2Plane_5DoF(modelPCD, scene, 
                            icp::ConvergenceCriteria(), m_debug);
                }
                else {
                    regResult = icp::ICP2D_Point2Plane_4DoF(modelPCD, scene, 
                            icp::ConvergenceCriteria(), m_debug);
                }

                Mat3x3f &A = regResult.transformation_;
        
                for (const Line2Dup::Feature& feature : tmpl0.features) {
                    float x = feature.x + match.x, y = feature.y + match.y;
                    float x1 = A[0][0] * x + A[0][1] * y + A[0][2];
                    float y1 = A[1][0] * x + A[1][1] * y + A[1][2];
                    
                    if (m_debug) {
                        cv::circle(imgDbg, { (int)x1 - offset.x, (int)y1 - offset.y }, 2,
                            cv::Scalar(0, 0, 255), cv::FILLED);
                    }
                }

                Eigen::MatrixXd A22(2, 2);
                A22 << A[0][0], A[0][1], A[1][0], A[1][1];
                Eigen::HouseholderQR<Eigen::MatrixXd> qr;
                qr.compute(A22);
                Eigen::Matrix2d R = qr.matrixQR().triangularView<Eigen::Upper>();
                Eigen::Matrix2d Q = qr.householderQ();
                Eigen::DiagonalMatrix<double, 2> S( 
                        R(0, 0) > 0 ? 1.0 : -1.0,
                        R(1, 1) > 0 ? 1.0 : -1.0 );
                Q = Q * S;
                R = S * R;
                
                float   dt = std::atan2(Q(1, 0), Q(0, 0)) / (float)(M_PI) * 180.0f,
                        kScaleX = R(0, 0),
                        kScaleY = R(1, 1),
                        kScale = 0.5f * (kScaleX + kScaleY);

                cv::Point2f centreP = centre + cv::Point2f(offset);
                cv::Point2f centrePRefined(
                    A[0][0] * centreP.x + A[0][1] * centreP.y + A[0][2],
                    A[1][0] * centreP.x + A[1][1] * centreP.y + A[1][2]);

                float   dx = centrePRefined.x - centreP.x, 
                        dy = centrePRefined.y - centreP.y;
                
                cv::RotatedRect &rr = result.region;
                rr.center += cv::Point2f(dx, dy);
                rr.angle += dt;
                if (m_refineAnisoScaling) {
                    result.scaleX = kScaleX * result.scale;
                    result.scaleY = kScaleY * result.scale;
                }
                else {
                    result.scaleX = kScale * result.scale;
                    result.scaleY = kScale * result.scale;
                }
                result.scale *= kScale;
                
                if (m_debug) {
                    std::cout << k + 1 << " [Refined] " << result << std::endl;
                }
            }
            
            if (m_debug) {
                cv::RotatedRect &rr = result.region;
                cv::Point2f vertices[4];
                rr.points(vertices);
                for(int j = 0; j < 4; j++){
                    int next = (j+1==4) ? 0 : (j+1);
                    cv::line(imgDbg, vertices[j], vertices[next], cv::Scalar(0, 0, 255), 2);
                }
                
                std::stringstream ss;

                ss << "[" << k + 1 << "]";
                
                cv::putText(imgDbg, ss.str(), cv::Point(match.x, match.y),
                    cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 255), 2);
            }
        }
    }

    std::sort(results.begin(), results.end(), 
        [&results] (const Result& lhs, const Result& rhs) -> bool
        {
            return (lhs.score > rhs.score);
        }
    );

    if (results.size() > maxNumMatches) {
        results.erase(results.begin() + maxNumMatches, results.end());
    }
    
    if (m_debug) {
        timer0.out("DETECT: Total time cost");
    }

    if (m_debug && !m_debugImagePath.empty()) {
        cv::imwrite(m_debugImagePath, imgDbg);
    }
    
    return (!results.empty());
}

void ShapeModelObjectDetectorBase::CropImage(const cv::Mat& src, cv::Mat& dst,
    const cv::RotatedRect& region, float scaleX, float scaleY, bool scaleContentToRegionSize,
    bool enableImageSizeScaling, cv::Size imageSize)
{
    if (src.empty()) {
        throw std::invalid_argument("Source image is empty.");
    }
    
    if ((scaleX < 0.001f) || (scaleY < 0.001f)) {
        throw std::invalid_argument("Scale value is too small.");
    }
    
    cv::Size2f size = region.size;
    float angle = region.angle;    
    cv::Point2f centre = region.center;
    
    cv::Mat H1 = cv::getRotationMatrix2D(centre, angle, 1.0);
    double Rdata[3] = { 0.0, 0.0, 1.0f };
    H1.push_back(cv::Mat(1, 3, CV_64F, Rdata));
    
    double H2data[3][3] = {
        { 1.0,  0.0,    -centre.x },
        { 0.0,  1.0,    -centre.y },
        { 0.0,  0.0,    1.0 }
    };
    cv::Mat H2(3, 3, CV_64F, H2data);

    double H3data_with_cs[3][3] = {
        { 1.0 / scaleX, 0.0, 0.5 * (double)(size.width - 1) },
        { 0.0, 1.0 / scaleY,  0.5 * (double)(size.height - 1) },
        { 0.0, 0.0, 1.0 }
    };
    double H3data_without_cs[3][3] = {
        { 1.0, 0.0, 0.5 * (size.width - 1) * scaleX },
        { 0.0, 1.0,  0.5 * (size.height - 1) * scaleY},
        { 0.0, 0.0, 1.0 }
    };
    cv::Mat H3(3, 3, CV_64F, scaleContentToRegionSize ? H3data_with_cs : H3data_without_cs);

    cv::Mat H = H3 * H2 * H1;
    H.resize(2);

    if (enableImageSizeScaling) {
        imageSize = cv::Size(std::ceil(scaleX * size.width), std::ceil(scaleY * size.height));
    }
    else if (imageSize.area() == 0) {
        imageSize = region.size;
    }

    cv::warpAffine(src, dst, H, imageSize,
        std::hypot(scaleX, scaleY) > 1.0f ? cv::INTER_CUBIC : cv::INTER_AREA);
}

void ShapeModelObjectDetectorBase::CropImage(const cv::Mat& src, cv::Mat& dst,
    const Result& result, bool scaleContentToRegionSize, bool enableImageSizeScaling, 
    cv::Size imageSize)
{
    CropImage(src, dst, result.region, result.scaleX, result.scaleY,
            scaleContentToRegionSize, enableImageSizeScaling, imageSize);
}

void ShapeModelObjectDetectorBase::ComputePaddings(const cv::Mat& src, int& dw, int& dh)
{
    int lenDiag = std::ceil(std::hypot(src.cols, src.rows));
    dw = std::ceil((lenDiag - src.cols) / 2.0f);
    dh = std::ceil((lenDiag - src.rows) / 2.0f);
}

void ShapeModelObjectDetectorBase::SetRefine(bool refine)
{
    m_refine = refine;
}

bool ShapeModelObjectDetectorBase::GetRefine() const
{
    return m_refine;
}

void ShapeModelObjectDetectorBase::SetGVCompareThreshold(float thresh)
{
    if ((thresh >= 0.0f) && (thresh <= 100.0f)) {
        m_gvCompareThreshold = thresh;
    }
}

float ShapeModelObjectDetectorBase::GetGVCompareThreshold() const
{
    return m_gvCompareThreshold;
}

void ShapeModelObjectDetectorBase::SetGVCompare(bool enable)
{
    m_enableGVCompare = enable;
}

bool ShapeModelObjectDetectorBase::GetGVCompare() const
{
    return m_enableGVCompare;
}

void ShapeModelObjectDetectorBase::SetDebug(bool debug)
{
    m_detector.debug = debug;
    m_detector.modality->debug = debug;
    m_debug = debug;
}

bool ShapeModelObjectDetectorBase::GetDebug() const
{
    return m_debug;
}

void ShapeModelObjectDetectorBase::SetDebugImagePath(const std::string& path)
{
    m_debugImagePath = path;
}

std::string ShapeModelObjectDetectorBase::GetDebugImagePath() const
{
    return m_debugImagePath;
}

const Line2Dup::Detector& ShapeModelObjectDetectorBase::GetDetector() const
{
    return m_detector;
}

void ShapeModelObjectDetectorBase::SetRefineAnisoScaling(bool enabled)
{
    m_refineAnisoScaling = enabled;
}

bool ShapeModelObjectDetectorBase::GetRefineAnisoScaling() const
{
    return m_refineAnisoScaling;
}

void ShapeModelObjectDetectorBase::SetMaxDistDiff(float maxDistDiff)
{
    if ((maxDistDiff >= 1.0f) && (maxDistDiff <= 25.0f)) {
        m_maxDistDiff = maxDistDiff;
    }
}

float ShapeModelObjectDetectorBase::GetMaxDistDiff() const
{
    return m_maxDistDiff;
}

void ShapeModelObjectDetectorBase::SetMaxOverlap(float maxOverlap)
{
    if ((maxOverlap > 0.0f) && (maxOverlap <= 1.0f)) {
        m_maxOverlap = maxOverlap;
    }
}

float ShapeModelObjectDetectorBase::GetMaxOverlap() const
{
    return m_maxOverlap;
}

void ShapeModelObjectDetectorBase::SetNMSDampFactor(float damp)
{
    if ((damp > 0.0f) && (damp <= 1.0f)) {
        m_nmsDampFactor = damp;
    }
}

float ShapeModelObjectDetectorBase::GetNMSDampFactor() const
{
    return m_nmsDampFactor;
}

float ShapeModelObjectDetectorBase::ComputePearson(const cv::Mat& src, 
    const TemplateInfo& tmplInfo) const
{
    float srcMean, srcSqsum;
    ComputeMeanSqsum(src, srcMean, srcSqsum);

    float area = (float)(tmplInfo.size.area());
    float numer = src.dot(tmplInfo.imgTmpl) - area * tmplInfo.mean * srcMean;
    float denom = std::sqrt(srcSqsum - area * srcMean * srcMean) 
                * std::sqrt(tmplInfo.sqsum - area * tmplInfo.mean * tmplInfo.mean);

    if (denom == 0.f) {
        denom = 1.0e-9f;
    }

    return (100.0f * numer / denom);
}

void ShapeModelObjectDetectorBase::ComputeMeanSqsum(const cv::Mat& img, float &mean, 
    float &sqsum) const
{
    cv::Scalar mean0 = cv::mean(img);
    mean = mean0[0];
    sqsum = cv::norm(img, cv::NORM_L2SQR);
}


//////////////////////////////////////////////////////////////////////////////

ShapeModelObjectDetector::ShapeModelObjectDetector( 
    float angleMin, float angleMax, float angleStep,
    float scaleMin, float scaleMax, float scaleStep, bool refine,
    int numFeatures, const std::vector<int>& T, float weakThreshold, 
    float strongThreshold, float maxGradient, int numOri)
: ShapeModelObjectDetectorBase(angleMin, angleMax, angleStep,
    scaleMin, scaleMax, scaleStep, refine, numFeatures, T,
    weakThreshold, strongThreshold, maxGradient, numOri)
{
}

ShapeModelObjectDetector::~ShapeModelObjectDetector()
{
}

bool ShapeModelObjectDetector::Load(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }

    TemplateInfo& tmplInfo = m_tmplInfos[defaultClassId];
    
    fs["TemplateImageSize"] >> tmplInfo.size;
    
    cv::FileNode fnPaddings = fs["Paddings"];
    tmplInfo.dw = fnPaddings["dw"];
    tmplInfo.dh = fnPaddings["dh"];
    
    tmplInfo.shapeInfo.read(fs["ShapeInfo"]);

    m_detector.read(fs["Detector"]);
    m_detector.readClasses(fs["Classes"]);
    
    fs["TemplateImage"] >> tmplInfo.imgTmpl;
    
    if (!tmplInfo.imgTmpl.empty()) {
        ComputeMeanSqsum(tmplInfo.imgTmpl, tmplInfo.mean, tmplInfo.sqsum);
    }
    
    return true;
}

bool ShapeModelObjectDetector::Save(const std::string& filename) const
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        return false;
    }

    decltype(m_tmplInfos)::const_iterator it = m_tmplInfos.find(defaultClassId);
    if (it == m_tmplInfos.cend()) {
        return false;
    }

    const TemplateInfo& tmplInfo = it->second;
    
    fs << "TemplateImageSize" << tmplInfo.size;
    
    fs << "Paddings" << "{" ;
    fs << "dw" << tmplInfo.dw ;
    fs << "dh" << tmplInfo.dh ;
    fs << "}";
        
    fs << "ShapeInfo" << "{" ;
    tmplInfo.shapeInfo.write(fs);
    fs << "}";
    
    fs << "Detector" << "{";
    m_detector.write(fs);
    fs << "}";
    
    m_detector.writeClasses(fs);
    
    fs << "TemplateImage" << tmplInfo.imgTmpl;
    
    return true;
}

bool ShapeModelObjectDetector::Register(const cv::Mat& img, const cv::Mat& mask,
    bool fast, const std::string& classId)
{
    return (ShapeModelObjectDetectorBase::RegisterEx(
            { defaultClassId }, { img }, fast, { mask }) > 0);
}

bool ShapeModelObjectDetector::Detect(const cv::Mat& src, 
    std::vector<Result>& results, float threshold, int maxNumMatches) const
{
    return ShapeModelObjectDetectorBase::Detect(src, results, threshold, 
        maxNumMatches, { defaultClassId } );
}

//////////////////////////////////////////////////////////////////////////////

static const char* s_smodex_magic = "ShapeModelObjectDetectorEx";

ShapeModelObjectDetectorEx::ShapeModelObjectDetectorEx( 
    float angleMin, float angleMax, float angleStep,
    float scaleMin, float scaleMax, float scaleStep, bool refine,
    int numFeatures, const std::vector<int>& T, float weakThreshold, 
    float strongThreshold, float maxGradient, int numOri)
: ShapeModelObjectDetectorBase(angleMin, angleMax, angleStep,
    scaleMin, scaleMax, scaleStep, refine, numFeatures, T,
    weakThreshold, strongThreshold, maxGradient, numOri)
{
}

ShapeModelObjectDetectorEx::~ShapeModelObjectDetectorEx()
{
}

bool ShapeModelObjectDetectorEx::Load(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }
    
    cv::FileNode fnMagic = fs["Type"];
    if (!fnMagic.isString()) {
        return false;
    }
    
    std::string magic;
    fnMagic >> magic;
    if (magic != s_smodex_magic) {
        return false;
    }
 
    m_tmplInfos.clear();
    
    cv::FileNode fnTemplates = fs["Templates"];
    for (cv::FileNodeIterator it = fnTemplates.begin(); it != fnTemplates.end(); it++) {
        cv::FileNode fn = *it;
        
        std::string classId;
        fn["ClassID"] >> classId;
        
        TemplateInfo& tmplInfo = m_tmplInfos[classId];
        
        fn["ImageSize"] >> tmplInfo.size;
        
        cv::FileNode fnPaddings = fn["Paddings"];
        tmplInfo.dw = fnPaddings["dw"];
        tmplInfo.dh = fnPaddings["dh"];
        
        tmplInfo.shapeInfo.read(fn["ShapeInfo"]);
    
        fn["TemplateImage"] >> tmplInfo.imgTmpl;
        
        if (!tmplInfo.imgTmpl.empty()) {
            ComputeMeanSqsum(tmplInfo.imgTmpl, tmplInfo.mean, tmplInfo.sqsum);
        }
    }
    
    m_detector.read(fs["Detector"]);
    m_detector.readClasses(fs["Classes"]);
    
    return true;
}

bool ShapeModelObjectDetectorEx::Save(const std::string& filename) const
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        return false;
    }
    
    fs << "Type" << s_smodex_magic;

    fs << "Templates" << "[";
    for (const auto& pair : m_tmplInfos) {
        fs << "{";
        
        const TemplateInfo& tmplInfo = pair.second;
        
        fs << "ClassID" << pair.first;
        
        fs << "ImageSize" << tmplInfo.size;
        
        fs << "Paddings" << "{" ;
        fs << "dw" << tmplInfo.dw ;
        fs << "dh" << tmplInfo.dh ;
        fs << "}";
            
        fs << "ShapeInfo" << "{" ;
        tmplInfo.shapeInfo.write(fs);
        fs << "}";
        
        fs << "TemplateImage" << tmplInfo.imgTmpl;
        fs << "}";
    }
    fs << "]";
    
    fs << "Detector" << "{";
    m_detector.write(fs);
    fs << "}";
    
    m_detector.writeClasses(fs);
    
    return true;
}

bool ShapeModelObjectDetectorEx::Register(const cv::Mat& img, const cv::Mat& mask,
    bool fast, const std::string& classId)
{
    return (ShapeModelObjectDetectorBase::RegisterEx({ classId }, { img }, fast, { mask }) > 0);
}

//////////////////////////////////////////////////////////////////////////////

static int GetOMPNumThreads()
{
    int numProcs = omp_get_num_procs();

    int numThreadsEnv = 0;
    char *ompNumThreadsEnv = getenv("OMP_NUM_THREADS");
    if (ompNumThreadsEnv && (strlen(ompNumThreadsEnv) > 0)) {
        numThreadsEnv = strtol(ompNumThreadsEnv, NULL, 10);
    }

    if ((numThreadsEnv > 0) && (numThreadsEnv <= numProcs)) {
        return numThreadsEnv;
    }
    else {
        return numProcs;
    }
}

