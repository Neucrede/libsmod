#include <stdexcept>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "smod.h"
#include "nmsboxes.h"
#include "icp.h"
#include "timer.hxx"

using namespace SMOD;

ShapeModelObjectDetector::ShapeModelObjectDetector( 
    float angleMin, float angleMax, float angleStep,
    float scaleMin, float scaleMax, float scaleStep, bool refine,
    int numFeatures, const std::vector<int>& T, float weakThreshold, 
    float strongThreshold, float maxGradient, int numOri)
: m_detector(numFeatures, T, weakThreshold, strongThreshold, maxGradient, numOri),
    m_classId("CLASS1"), m_debug(false), m_refine(refine)
{
    m_shapeInfo.angle_range = { angleMin, angleMax };
    m_shapeInfo.scale_range = { scaleMin, scaleMax };
    m_shapeInfo.angle_step = angleStep;
    m_shapeInfo.scale_step = scaleStep;
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
    
    fs["TemplateImageSize"] >> m_tmplSize;
    
    cv::FileNode fnPaddings = fs["Paddings"];
    m_dw = fnPaddings["dw"];
    m_dh = fnPaddings["dh"];
    
    m_shapeInfo.read(fs["ShapeInfo"]);
    m_detector.read(fs["Detector"]);
    m_detector.readClasses(fs["Classes"]);
    
    return true;
}

bool ShapeModelObjectDetector::Save(const std::string& filename) const
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        return false;
    }
    
    fs << "TemplateImageSize" << m_tmplSize;
    
    fs << "Paddings" << "{" ;
    fs << "dw" << m_dw ;
    fs << "dh" << m_dh ;
    fs << "}";
        
    fs << "ShapeInfo" << "{" ;
    m_shapeInfo.write(fs);
    fs << "}";
    
    fs << "Detector" << "{";
    m_detector.write(fs);
    fs << "}";
    
    m_detector.writeClasses(fs);
    
    return true;
}

bool ShapeModelObjectDetector::Register(const cv::Mat& img, const cv::Mat& mask,
    bool fast)
{
    if (img.empty()) {
        throw std::invalid_argument("Image is empty.");
    }
    
    Timer timer;
    
    m_tmplSize = img.size();
    
    ComputePaddings(img);

    /*
    cv::Mat imgBlurred;
    cv::GaussianBlur(img, imgBlurred, cv::Size(3, 3), 1.0, 1.0);
    */

    cv::Mat imgPadded;
    cv::copyMakeBorder(img /* imgBlurred */, imgPadded, m_dh, m_dh, m_dw, m_dw, 
        cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat mask1, maskPadded;
    if (!mask.empty()) {
        mask1 = mask;
    }
    else {
        mask1 = cv::Mat(img.size(), CV_8UC1, cv::Scalar::all(255));
    }
    cv::copyMakeBorder(mask1, maskPadded, m_dh, m_dh, m_dw, m_dw, 
        cv::BORDER_CONSTANT, cv::Scalar::all(0));
    m_shapeInfo.src = imgPadded;
    m_shapeInfo.mask = maskPadded;
    m_shapeInfo.produce();
    
    if (m_debug) {
        std::cout << "Registering template images..." << std::endl;
    }
    
    m_shapeInfo.valid_infos.clear();
    if (!fast) {
        for (const ShapeInfo::Info &info : m_shapeInfo.infos) {
            if (m_debug) {
                std::cout   << "Angle = " << info.angle 
                            << ", Scale = " << info.scale 
                            << std::endl;
            }
            
            int id = m_detector.addTemplate(m_shapeInfo.srcOf(info), m_classId, 
                m_shapeInfo.maskOf(info));
            if (id >= 0) {
                m_shapeInfo.valid_infos.push_back(info);
            }
        }
    }
    else {
        float scale0 = -1, angle0;
        int id0;
        cv::Point2f centre(m_shapeInfo.src.cols / 2.0f, m_shapeInfo.src.rows / 2.0f);
        for (size_t i = 0; i != m_shapeInfo.infos.size(); ++i) {
            const ShapeInfo::Info &info = m_shapeInfo.infos[i];
            
            if (m_debug) {
                std::cout   << "Angle = " << info.angle 
                            << ", Scale = " << info.scale 
                            << std::endl;
            }
            
            if (info.scale != scale0) {
                int id = m_detector.addTemplate(m_shapeInfo.srcOf(info), m_classId, 
                    m_shapeInfo.maskOf(info));
                if (id >= 0) {
                    m_shapeInfo.valid_infos.push_back(info);
                    scale0 = info.scale;
                    angle0 = info.angle;
                    id0 = id;
                }
                else {
                    scale0 = -1;
                }
            }
            else {
                int id = m_detector.addTemplateRotate(m_classId, id0, info.angle - angle0,
                    centre);
                
                if (m_debug) {
                    std::cout << "Computed from reference features." << std::endl;
                }
                
                if (id >= 0) {
                    m_shapeInfo.valid_infos.push_back(info);
                }
            }
        }
    }
    
    if (m_debug) {
        timer.out("REGISTER: Time cost");
    }
    
    return !(m_shapeInfo.valid_infos.empty());
}

bool ShapeModelObjectDetector::Detect(cv::Mat& src, 
    std::vector<Result>& results, float threshold, int maxNumMatches)
{   
    if (src.empty()) {
        throw std::invalid_argument("Source image is empty.");
    }
    
    cv::Mat imgDbg;
    Timer timer, timer0;

    if (maxNumMatches <= 0) {
        maxNumMatches = 65535;
    }
    
    results.clear();
    
    if (m_srcPadded.empty() || m_srcSize.width != src.cols || m_srcSize.height != src.rows) {
        const int stride = std::max(
            (int)(*std::max_element(m_detector.T_at_level.begin(), 
                m_detector.T_at_level.end()) * std::pow(2, m_detector.T_at_level.size() - 1)),
            16);
        int w = std::ceil((float)(src.cols + 2 * m_dw) / (float)stride) * stride;
        int h = std::ceil((float)(src.rows + 2 * m_dh) / (float)stride) * stride;
        
        m_srcPadded = cv::Mat(h, w, src.type(), cv::Scalar::all(0));
        m_srcSize = cv::Size(src.cols, src.rows);
    }
    
    src.copyTo(m_srcPadded(cv::Rect(m_dw, m_dh, src.cols, src.rows)));
    
    if (m_debug) {
        if (src.channels() == 1) {
            cv::cvtColor(src, imgDbg, cv::COLOR_GRAY2BGR);
        }
        else {
            imgDbg = src.clone();
        }
        
        timer.out("DETECT: Prepare padded image");
    }
    
    cv::Ptr<Line2Dup::ColorGradientPyramid> quantizer;
    std::vector<std::string> ids = { m_classId };
    std::vector<Line2Dup::Match> matches 
        = m_detector.match(m_srcPadded, threshold, quantizer, ids, cv::Mat(), maxNumMatches);
    if (m_debug) {
        timer.out("DETECT: Match");
    }
    
    if (matches.empty()) {
        return false;
    }
    
    std::vector<cv::Rect> boundBoxes;
    std::vector<float> _scores;
    for (const Line2Dup::Match& match : matches) {
        const std::vector<Line2Dup::Template>& tmpls 
            = m_detector.getTemplates(m_classId, match.template_id);
        if (tmpls.empty()) {
            continue;
        }
        
        boundBoxes.emplace_back(match.x, match.y, tmpls[0].width, tmpls[0].height);
        _scores.push_back(match.similarity);
    }

    std::vector<int> indices;    
    ::NMSBoxes(boundBoxes, _scores, 0.9f * threshold, 0.5f, indices);
    
    if (m_debug) {
        timer.out("DETECT: NMS");
    }
    
    Scene_edge scene;
    std::vector<::Vec2f> pcdBuf, normalBuf;
    if (m_refine) {
        cv::Mat Dx, Dy;
        quantizer->sobel_dx0.convertTo(Dx, CV_16S, 128);
        quantizer->sobel_dy0.convertTo(Dy, CV_16S, 128);
        
        scene.init_Scene_edge_cpu(m_srcPadded, pcdBuf, normalBuf, 2.0f,
            0.8 * m_detector.modality->weak_threshold, 
            0.8 * m_detector.modality->strong_threshold,
            Dx, Dy);
        
        if (m_debug) {
            timer.out("DETECT: Init scene");
        }
    }
    
    for (size_t k = 0; k != std::min((size_t)maxNumMatches, indices.size()); ++k) {
        int i = indices[k];
        const Line2Dup::Match& match = matches[i];
        const std::vector<Line2Dup::Template>& tmpls
            = m_detector.getTemplates(m_classId, match.template_id);
        const Line2Dup::Template& tmpl0 = tmpls[0];
        const ShapeInfo::Info& info = m_shapeInfo.infos[match.template_id];
        
        cv::Point2f centre(
            match.x - tmpl0.tl_x + (float)(m_tmplSize.width) / 2.0f,
            match.y - tmpl0.tl_y + (float)(m_tmplSize.height) / 2.0f
        );
        cv::Size2f rSize(
            (float)m_tmplSize.width,
            (float)m_tmplSize.height
        );
                
        results.emplace_back(centre, rSize, -info.angle, info.scale, match.similarity);
        Result& result = results.back();
        
        if (m_debug) {
            std::cout << k + 1 << " [Original] " << result << std::endl;
            for (const Line2Dup::Feature& feature : tmpl0.features) {
                cv::circle(imgDbg, 
                    { feature.x + match.x - m_dw, feature.y + match.y - m_dh},
                    2, cv::Scalar(255, 0, 0), cv::FILLED);
            }
        }
        
        if (m_refine) {
            std::vector<::Vec2f> modelPCD;
            modelPCD.reserve(tmpl0.features.size());
            for (const Line2Dup::Feature& feature : tmpl0.features) {
                modelPCD.emplace_back(feature.x + match.x, feature.y + match.y);
            }
            
            icp::RegistrationResult regResult
                = icp::ICP2D_Point2Plane_cpu(modelPCD, scene);
    
            Mat3x3f A = regResult.transformation_;
    
            for (const Line2Dup::Feature& feature : tmpl0.features) {
                float x = feature.x + match.x, y = feature.y + match.y;
                float x1 = A[0][0] * x + A[0][1] * y + A[0][2];
                float y1 = A[1][0] * x + A[1][1] * y + A[1][2];
                
                if (m_debug) {
                    cv::circle(imgDbg, { (int)x1 - m_dw, (int)y1 - m_dh }, 2,
                        cv::Scalar(0, 0, 255), cv::FILLED);
                }
            }
            
            cv::Point2f centreP = centre + cv::Point2f(m_dw, m_dh);
            cv::Point2f centrePRefined(
                A[0][0] * centreP.x + A[0][1] * centreP.y + A[0][2],
                A[1][0] * centreP.x + A[1][1] * centreP.y + A[1][2] );
            float   dx = centrePRefined.x - centreP.x, 
                    dy = centrePRefined.y - centreP.y, 
                    dt = std::atan2(A[1][0], A[0][0]) / (float)(M_PI) * 180.0f,
                    kScale = std::hypot(A[0][0], A[1][0]);
            
            cv::RotatedRect &rr = result.region;
            rr.center += cv::Point2f(dx, dy);
            rr.angle += dt;
            result.scale *= kScale;
            
            if (m_debug) {
                std::cout << k + 1 << " [Refined] " << result << std::endl;
                timer.out("DETECT: Refine(k)");
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
            ss  << std::fixed << std::setprecision(2)
                << "Angle = " << rr.angle
                << ", Scale = " << result.scale
                << ", Score = " << result.score;
            
            cv::putText(imgDbg, ss.str(), cv::Point(match.x, match.y),
                cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 255), 2);
        }
    }
    
    if (m_debug) {
        timer0.out("DETECT: Time cost");
    }

    if (m_debug && !m_debugImagePath.empty()) {
        cv::imwrite(m_debugImagePath, imgDbg);
    }
    
    return true;
}

bool ShapeModelObjectDetector::CropImage(const cv::Mat& src, cv::Mat& dst,
    const cv::RotatedRect& region, float scale, bool enableScaling)
{
    if (src.empty()) {
        throw std::invalid_argument("Source image is empty.");
    }
    
    if (std::abs(scale - 1.0f) < 0.001) {
        scale = 1.0f;
    }
    
    if (!enableScaling) {
        scale = 1.0f;
    }
    
    cv::Size2f size = region.size;
    float angle = region.angle;    
    cv::Point2f centre = region.center;
    
    cv::Mat H1 = cv::getRotationMatrix2D(centre, angle, 1.0);
    double Rdata[3] = { 0.0, 0.0, 1.0f };
    H1.push_back(cv::Mat(1, 3, CV_64F, Rdata));
    
    float ex = -centre.x + 0.5f * size.width * scale;
    float ey = -centre.y + 0.5f * size.height * scale;
    double H2data[3][3] = {
        { 1.0,  0.0,    ex },
        { 0.0,  1.0,    ey },
        { 0.0,  0.0,    1.0 }
    };
    cv::Mat H2(3, 3, CV_64F, H2data);
    
    /*
    double H3data[3][3] = {
        { 1.0 / scale,  0.0,            0.0 },
        { 0.0,          1.0 / scale,    0.0 },
        { 0.0,          0.0,            1.0 }
    };
    cv::Mat H3(3, 3, CV_64F, H3data);
    
    cv::Mat H = H3 * H2 * H1;
    */
    
    cv::Mat H = H2 * H1;
    H.resize(2);

    cv::warpAffine(src, dst, H, 
        cv::Size(std::ceil(scale * size.width), std::ceil(scale * size.height)),
        scale > 1.0 ? cv::INTER_CUBIC : cv::INTER_AREA);
    
    return true;
}

bool ShapeModelObjectDetector::CropImage(const cv::Mat& src, cv::Mat& dst,
    const Result& result, bool enableScaling)
{
    return CropImage(src, dst, result.region, result.scale, enableScaling);
}

void ShapeModelObjectDetector::BilateralFilter(const cv::Mat& src, cv::Mat& dst,
    int kernelSize, float sigmaDomain, float sigmaRange)
{
    if (src.empty()) {
        throw std::invalid_argument("Source image is empty.");
    }
    
    if (kernelSize >= 2) {
        kernelSize = ((kernelSize >> 1) << 1) + 1;
    }
    else if (kernelSize >= 0) {
        dst = src.clone();
        return;
    }
    
    if (sigmaDomain <= 0.001f) sigmaDomain = 1.0f;
    if (sigmaRange <= 0.001f) sigmaRange = 1.0f;

    cv::bilateralFilter(src, dst, kernelSize, sigmaRange, sigmaDomain);
} 

void ShapeModelObjectDetector::CLAHE(const cv::Mat& src, cv::Mat& dst, float clipLimit,
    cv::Size gridSize)
{
    if (src.empty()) {
        throw std::invalid_argument("Source image is empty.");
    }
    
    if (src.channels() != 1) {
        throw std::runtime_error("Source image must be monochannel.");
    }

    cv::Ptr<cv::CLAHE> pCLAHE = cv::createCLAHE(clipLimit, gridSize);
    try {
        pCLAHE->apply(src, dst);
    }
    catch (std::exception &e) {
        std::cerr << "CLAHE: " << e.what() << "\n";
        dst = src.clone();
    }
}

void ShapeModelObjectDetector::CreateMask(const cv::Mat& src, cv::Mat& dst, int lowThresh,
    int highThresh, int dilateSize)
{
    if (src.empty()) {
        throw std::invalid_argument("Source image is empty.");
    }

    cv::Mat srcGray;
    if (src.channels() == 1) {
        srcGray = src;
    }
    else {
        cv::cvtColor(src, srcGray, cv::COLOR_BGR2GRAY);
    }

    if (lowThresh < highThresh) {
        dst = (srcGray >= lowThresh) & (srcGray <= highThresh);
    }
    else {
        dst = (srcGray >= lowThresh) | (srcGray <= highThresh);
    }

    if (dilateSize > 0) {
        cv::dilate(dst, dst, 
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilateSize, dilateSize)));
    }

    cv::rectangle(dst, cv::Rect(0, 0, dst.cols, dst.rows), cv::Scalar::all(0),
        std::max(7, dilateSize));
}

void ShapeModelObjectDetector::SetRefine(bool refine)
{
    m_refine = refine;
}

void ShapeModelObjectDetector::SetDebug(bool debug)
{
    m_detector.debug = debug;
    m_detector.modality->debug = debug;
    m_debug = debug;
}

void ShapeModelObjectDetector::SetDebugImagePath(const std::string& path)
{
    m_debugImagePath = path;
}

Line2Dup::Detector& ShapeModelObjectDetector::GetDetector()
{
    return m_detector;
}

void ShapeModelObjectDetector::GetPaddings(int &dw, int &dh)
{
    dw = m_dw;
    dh = m_dh;
}

void ShapeModelObjectDetector::ComputePaddings(const cv::Mat& src)
{
    int lenDiag = std::ceil(std::hypot(src.cols, src.rows));

    m_dw = std::ceil((lenDiag - src.cols) / 2.0f);
    m_dh = std::ceil((lenDiag - src.rows) / 2.0f);
}
