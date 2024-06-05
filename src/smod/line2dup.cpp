#include <iostream>
#include <algorithm>
#include <opencv2/imgcodecs.hpp>
#include "line2dup.h"
#include "mipp.h" 
#include "nmsboxes.h"
#include "timer.hxx"

using namespace std;
using namespace cv;

namespace Line2Dup
{

static const int max_features = 8190;

// ----------------------------------------------------------------------------

void Feature::read(const FileNode &fn)
{
    FileNodeIterator fni = fn.begin();
    fni >> x >> y >> label;
}

void Feature::write(FileStorage &fs) const
{
    fs << "[:" << x << y << label << "]";
}

// ----------------------------------------------------------------------------

void Template::read(const FileNode &fn)
{
    width = fn["width"];
    height = fn["height"];
    tl_x = fn["tl_x"];
    tl_y = fn["tl_y"];
    pyramid_level = fn["pyramid_level"];

    FileNode features_fn = fn["features"];
    features.resize(features_fn.size());
    FileNodeIterator it = features_fn.begin(), it_end = features_fn.end();
    for (int i = 0; it != it_end; ++it, ++i) {
        features[i].read(*it);
    }
}

void Template::write(FileStorage &fs) const
{
    fs << "width" << width;
    fs << "height" << height;
    fs << "tl_x" << tl_x;
    fs << "tl_y" << tl_y;
    fs << "pyramid_level" << pyramid_level;

    fs << "features"
       << "[";
    for (int i = 0; i < (int)features.size(); ++i) {
        features[i].write(fs);
    }
    fs << "]"; // features
}

// ----------------------------------------------------------------------------

ColorGradientPyramid::ColorGradientPyramid(const Mat &_src, const Mat &_mask,
        float _weak_threshold, size_t _num_features,float _strong_threshold, 
        bool _debug, int _num_ori, float _max_gradient)
: src(_src), mask(_mask), pyramid_level(0), weak_threshold(_weak_threshold),
    num_features(_num_features), strong_threshold(_strong_threshold),
    debug(_debug), num_ori(_num_ori), max_gradient(_max_gradient)
{
    if ((num_ori != 8) && (num_ori != 16)) {
        CV_Error(Error::StsBadArg, "Invalid value of number of orientations.");
    }

    if ((max_gradient > 360.0f) || (max_gradient <= 0)) {
        max_gradient = 361.0f;
    }

    update(false);
}

void ColorGradientPyramid::update(bool fast)
{
    quantizedOrientations(src, magnitude, angle, angle_ori, weak_threshold, fast);
}

void ColorGradientPyramid::pyrDown(bool fast)
{
    // Some parameters need to be adjusted
    num_features /= 2;
    ++pyramid_level;

    Size size(src.cols / 2, src.rows / 2);
    
    // Downsample the current inputs
    if ((src.channels() != 1) || (!fast)) {
        Mat next_src;
        cv::pyrDown(src, next_src, size);
        src = next_src;
    }

    if (!mask.empty()) {
        Mat next_mask;
        resize(mask, next_mask, size, 0.0, 0.0, INTER_NEAREST);
        mask = next_mask;
    }

    update(fast);
}

void ColorGradientPyramid::quantize(Mat &dst) const
{
    dst = Mat::zeros(angle.size(), num_ori == 8 ? CV_8U : CV_16U);
    angle.copyTo(dst, mask);
}

bool ColorGradientPyramid::extractTemplate(Template &templ) const
{
    if (debug) {
        std::cout << "Extracting template. L = " << pyramid_level << std::endl;
    }
    
    // Want features on the border to distinguish from background
    Mat local_mask;
    if (!mask.empty()) {
        erode(mask, local_mask, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
    }
    
    std::vector<Candidate> candidates;
    bool no_mask = local_mask.empty();
    float threshold_sq = std::pow(0.5f * (weak_threshold + strong_threshold), 2);

    float max_mag = max_gradient * max_gradient;
    if (max_mag < threshold_sq) {
        max_mag = threshold_sq + 1;
    }

    int nms_kernel_size = 3;
    cv::Mat magnitude_valid = cv::Mat(magnitude.size(), CV_8UC1, cv::Scalar(255));

    for (int r = 0+nms_kernel_size/2; r < magnitude.rows-nms_kernel_size/2; ++r) {
        const uchar *mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);

        for (int c = 0+nms_kernel_size/2; c < magnitude.cols-nms_kernel_size/2; ++c) {
            if (no_mask || mask_r[c]) {
                float score = 0;
                if(magnitude_valid.at<uchar>(r, c)>0) {
                    score = magnitude.at<float>(r, c);
                    bool is_max = true;
                    for(int r_offset = -nms_kernel_size/2; r_offset <= nms_kernel_size/2; r_offset++) {
                        for(int c_offset = -nms_kernel_size/2; c_offset <= nms_kernel_size/2; c_offset++) {
                            if(r_offset == 0 && c_offset == 0) {
                                continue;
                            }

                            if(score < magnitude.at<float>(r+r_offset, c+c_offset)) {
                                score = 0;
                                is_max = false;
                                break;
                            }
                        }

                        if(!is_max) {
                            break;
                        }
                    }

                    if(is_max) {
                        for(int r_offset = -nms_kernel_size/2; r_offset <= nms_kernel_size/2; r_offset++) {
                            for(int c_offset = -nms_kernel_size/2; c_offset <= nms_kernel_size/2; c_offset++) {
                                if(r_offset == 0 && c_offset == 0) {
                                    continue;
                                }

                                magnitude_valid.at<uchar>(r+r_offset, c+c_offset) = 0;
                            }
                        }
                    }
                }

                if ((score < threshold_sq) || (score > max_mag)) {
                    continue;
                }
                else if ((num_ori == 8) && (angle.at<uchar>(r, c) > 0)) {
                    candidates.push_back(Candidate(c, r, getLabel(angle.at<uchar>(r, c)), score));
                    candidates.back().f.theta = angle_ori.at<float>(r, c);
                }
                else if ((num_ori == 16) && (angle.at<uint16_t>(r, c) > 0)) {
                    candidates.push_back(Candidate(c, r, getLabel(angle.at<uint16_t>(r, c)), score));
                    candidates.back().f.theta = angle_ori.at<float>(r, c);
                }
            }
        }
    }

    // We require a certain number of features
    if (candidates.size() < num_features) {
        if(candidates.size() <= 4) {
            std::cerr << "Too few features, abort" << std::endl;
            return false;
        }
        
        std::cerr << "Have no enough features, exaustive mode" << std::endl;
    }

    // NOTE: Stable sort to agree with old code, which used std::list::sort()
    std::stable_sort(candidates.begin(), candidates.end());

    // Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
    float distance = static_cast<float>(candidates.size() / num_features + 1);

    // selectScatteredFeatures always return true
    if (!selectScatteredFeatures(candidates, templ.features, num_features, distance)) {
        return false;
    }

    if (templ.features.size() > max_features) {
        templ.features.resize(max_features);
    }
    
    // Size determined externally, needs to match templates for other modalities
    templ.width = -1;
    templ.height = -1;
    templ.pyramid_level = pyramid_level;
    
    if (debug) {
        std::cout << "Number of features: " << templ.features.size() << std::endl;
    }

    return true;
}

inline int ColorGradientPyramid::getLabel(int quantized) const
{
    switch (quantized)
    {
    case 1 << 0 : return 0;
    case 1 << 1 : return 1;
    case 1 << 2 : return 2;
    case 1 << 3 : return 3;
    case 1 << 4 : return 4;
    case 1 << 5 : return 5;
    case 1 << 6 : return 6;
    case 1 << 7 : return 7;
    case 1 << 8 : return 8;
    case 1 << 9 : return 9;
    case 1 << 10 : return 10;
    case 1 << 11 : return 11;
    case 1 << 12 : return 12;
    case 1 << 13 : return 13;
    case 1 << 14 : return 14;
    case 1 << 15 : return 15;
    default:
        CV_Error(Error::StsBadArg, "Invalid value of quantized parameter");
        return -1; //avoid warning
    }
}

bool ColorGradientPyramid::selectScatteredFeatures(const std::vector<Candidate> &candidates,
        std::vector<Feature> &features, size_t num_features, float distance) const
{
    features.clear();
    float distance_sq = distance * distance;
    int i = 0;

    bool first_select = true;

    while(true) {
        Candidate c = candidates[i];

        // Add if sufficient distance away from any previously chosen feature
        bool keep = true;
        
        for (int j = 0; (j < (int)features.size()) && keep; ++j) {
            Feature f = features[j];
            keep = (c.f.x - f.x) * (c.f.x - f.x) + (c.f.y - f.y) * (c.f.y - f.y) >= distance_sq;
        }

        if (keep) {
            features.push_back(c.f);
        }

        if (++i == (int)candidates.size()) {
            bool num_ok = features.size() >= num_features;

            if(first_select) {
                if(num_ok) {
                    features.clear(); // we don't want too many first time
                    i = 0;
                    distance += 1.0f;
                    distance_sq = distance * distance;
                    continue;
                }
                else {
                    first_select = false;
                }
            }

            // Start back at beginning, and relax required distance
            i = 0;
            distance -= 1.0f;
            distance_sq = distance * distance;

            if (num_ok || distance < 3) {
                break;
            }
        }
    }

    return true;
}

void ColorGradientPyramid::hysteresisGradient(Mat &magnitude, Mat &quantized_angle,
        Mat &angle, float threshold)
{
    // Quantize 360 degree range of orientations into (2 * num_ori) buckets
    Mat_<unsigned char> quantized_unfiltered;
    angle.convertTo(quantized_unfiltered, CV_8U, (double)(2 * num_ori) / 360.0);

    // Zero out top and bottom rows
    memset(quantized_unfiltered.ptr(), 0, quantized_unfiltered.cols);
    memset(quantized_unfiltered.ptr(quantized_unfiltered.rows - 1), 0, quantized_unfiltered.cols);
    // Zero out first and last columns
    for (int r = 0; r < quantized_unfiltered.rows; ++r) {
        quantized_unfiltered(r, 0) = 0;
        quantized_unfiltered(r, quantized_unfiltered.cols - 1) = 0;
    }

    // Mask (2 * num_ori) buckets into num_ori quantized orientations.
    const uint8_t uMask = (num_ori == 8) ? 0x07 : 0x0F;
    mipp::Reg<uint8_t> rMask = uMask;
    int N = angle.rows * angle.cols;
    const auto S = mipp::N<uint8_t>();
    int n0 = (N / S * S);
    for (int i = 0; i < n0; i += S) {
        uint8_t* p = &quantized_unfiltered.at<uint8_t>(i);
        mipp::Reg<uint8_t> rAngle(p);
        rAngle &= rMask;
        rAngle.store(p);
    }

    for (int i = n0; i < N; ++i) {
        uint8_t* p = &quantized_unfiltered.at<uint8_t>(i);
        *p &= uMask;
    }

    // Filter the raw quantized image. Only accept pixels where the magnitude is above some
    // threshold, and there is local agreement on the quantization.
    quantized_angle = Mat::zeros(angle.size(), num_ori == 8 ? CV_8U : CV_16U);

    float max_mag = max_gradient * max_gradient;
    if (max_mag < threshold) {
        max_mag = threshold + 1;
    }

    for (int r = 1; r < angle.rows - 1; ++r) {
        float *mag_r = magnitude.ptr<float>(r);

        for (int c = 1; c < angle.cols - 1; ++c) {
            if ((mag_r[c] > threshold) && (mag_r[c] <= max_mag)) {
                // Compute histogram of quantized bins in 3x3 patch around pixel
                int histogram[num_ori];
                ::memset(histogram, 0, sizeof(histogram));

                uchar *patch3x3_row = &quantized_unfiltered(r - 1, c - 1);
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                // Find bin with the most votes from the patch
                int max_votes = 0;
                int index = -1;
                for (int i = 0; i < num_ori; ++i)
                {
                    if (max_votes < histogram[i])
                    {
                        index = i;
                        max_votes = histogram[i];
                    }
                }

                // Only accept the quantization if majority of pixels in the patch agree
                static const int NEIGHBOR_THRESHOLD = num_ori / 4;
                if (max_votes >= NEIGHBOR_THRESHOLD) {
                    if (num_ori == 8) {
                        quantized_angle.at<uchar>(r, c) = uchar(1 << index);
                    }
                    else {
                        quantized_angle.at<uint16_t>(r, c) = uint16_t(1 << index);
                    }
                }
            }
        }
    }
}

void ColorGradientPyramid::quantizedOrientations(const Mat &src, Mat &magnitude,
    Mat &angle, Mat& angle_ori, float threshold, bool fast)
{
    if(src.channels() == 1) {
        Mat sobel_dx, sobel_dy, sobel_ag;
        if ((pyramid_level == 0) || (!fast)) {
            Sobel(src, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
            Sobel(src, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);

            if (pyramid_level == 0) {
                sobel_dx0 = sobel_dx.clone();
                sobel_dy0 = sobel_dy.clone();
            }
        }
        else {
            cv::pyrDown(sobel_dx_prev, sobel_dx);
            cv::pyrDown(sobel_dy_prev, sobel_dy);
        }
        
        sobel_dx_prev = sobel_dx;
        sobel_dy_prev = sobel_dy;

        // magnitude = (sobel_dx .DOT. (sobel_dx)' + sobel_dy .DOT. (sobel_dy)') / 16.0
        magnitude.create(sobel_dx.size(), CV_32F);
        int N = magnitude.rows * magnitude.cols;
        const auto S = mipp::N<float>();
        const mipp::Reg<float> rNormalizeFactor = 1.0f / 16.0f;
        int n0 = (N / S * S);
        for (int i = 0; i < n0; i += S) {
            float *px = &sobel_dx.at<float>(i);
            float *py = &sobel_dy.at<float>(i);
            mipp::Reg<float> rX(px);                 
            mipp::Reg<float> rY(py);
            mipp::Reg<float> rXX(rX * rX);
            mipp::Reg<float> rResult = mipp::mul(mipp::fmadd<float>(rY, rY, rXX), rNormalizeFactor);
            rResult.store(&magnitude.at<float>(i));
        }

        for (int i = n0; i < N; ++i) {
            float px = sobel_dx.at<float>(i);
            float py = sobel_dy.at<float>(i);
            magnitude.at<float>(i) = (px * px + py * py) / 16.0f;
        }

        phase(sobel_dx, sobel_dy, sobel_ag, true);

        hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
        angle_ori = sobel_ag;
    }
    else {
        CV_Error(Error::StsBadArg, "Number of image channels must be 1.");
    }
}

// ----------------------------------------------------------------------------

ColorGradient::ColorGradient(bool _debug, int _num_ori)
: weak_threshold(30.0f), num_features(63), strong_threshold(60.0f), debug(_debug),
    num_ori(_num_ori), max_gradient(255.0f)
{
}

ColorGradient::ColorGradient(float _weak_threshold, size_t _num_features, 
    float _strong_threshold, bool _debug, int _num_ori, float _max_gradient)
: weak_threshold(_weak_threshold), num_features(_num_features),
    strong_threshold(_strong_threshold), debug(_debug), num_ori(_num_ori),
    max_gradient(_max_gradient)
{
}

static const char CG_NAME[] = "ColorGradient";

std::string ColorGradient::name() const
{
    return CG_NAME;
}

void ColorGradient::read(const FileNode &fn)
{
    String type = fn["type"];
    CV_Assert(type == CG_NAME);

    weak_threshold = fn["weak_threshold"];
    num_features = int(fn["num_features"]);
    strong_threshold = fn["strong_threshold"];

    // Initialise with fail-safe values.
    num_ori = 8;
    max_gradient = 361.0f;

    try {
        num_ori = fn["num_ori"];
        max_gradient = fn["max_gradient"];
    }
    catch (std::exception &e) {
        // ignore;
    }
}

void ColorGradient::write(FileStorage &fs) const
{
    fs << "type" << CG_NAME;
    fs << "weak_threshold" << weak_threshold;
    fs << "num_features" << int(num_features);
    fs << "strong_threshold" << strong_threshold;
    fs << "num_ori" << num_ori;
    fs << "max_gradient" << max_gradient;
}

cv::Ptr<ColorGradientPyramid> ColorGradient::process(const cv::Mat src, const cv::Mat &mask) const
{
    return cv::makePtr<ColorGradientPyramid>(src, mask, weak_threshold, 
        num_features, strong_threshold, debug, num_ori, max_gradient);
}

// ----------------------------------------------------------------------------

static void spread(const Mat &src, Mat &dst, int T)
{
    int n = std::floor(std::log2((float)T));
    if (n <= 1) {
        dst = src.clone();
        return;
    }
    
    int N = (int)(std::pow(2, n));
    int cols = src.cols, rows = src.rows;
    cv::Mat M[n];
    
    for (int p = 0; p < n; ++p) {
        int s = (int)(std::pow(2, p));
        M[p] = cv::Mat::zeros(src.size(), src.type());
        const cv::Mat &S = (p == 0) ? src : M[p - 1];
        cv::bitwise_or(S.colRange(s, cols), S.colRange(0, cols - s),
            M[p].colRange(0, cols - s));
    }
    
    cv::Mat &M0 = M[0];
    for (int i = 1; i < n; ++i) {
        cv::bitwise_or(M0, M[i], M0);
    }
    
    for (int s = N; s < T; ++s) {
        cv::bitwise_or(M0.colRange(s, src.cols), src.colRange(0, src.cols - s),
            M0.colRange(0, src.cols - s));
    }
    
    cv::Mat A = M0.clone();
    
    for (int p = 0; p < n; ++p) {
        int s = (int)(std::pow(2, p));
        const cv::Mat &S = (p == 0) ? A : M[p - 1];
        cv::bitwise_or(S.rowRange(s, rows), S.rowRange(0, rows - s),
            M[p].rowRange(0, rows - s));
    }
    
    for (int i = 1; i < n; ++i) {
        cv::bitwise_or(M0, M[i], M0);
    }
    
    for (int s = N; s < T; ++s) {
        cv::bitwise_or(M0.rowRange(s, M0.rows), M0.rowRange(0, M0.rows - s),
            M0.rowRange(0, M0.rows - s));
    }
    
    dst = M0;
}

CV_DECL_ALIGNED(16) static const unsigned char SIMILARITY_LUT[256] = {
    /* Generated by lut_gen_8 */

    /*
     * 0..3
     * 4..7
     */

    0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4, 
    0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    
    0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4, 
    0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    
    0, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 
    0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    
    0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 
    0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3,
    
    0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4,
    
    0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4,
    
    0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    0, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4,
    
    0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3,
    0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4
};

static void computeResponseMaps(const Mat &src, std::vector<Mat> &response_maps)
{
    CV_Assert((src.rows * src.cols) % 16 == 0);

    // Allocate response maps
    response_maps.resize(8);
    for (int i = 0; i < 8; ++i) {
        response_maps[i].create(src.size(), CV_8U);
    }

    Mat lsb4(src.size(), CV_8U);
    Mat msb4(src.size(), CV_8U);

    const int N = mipp::N<int8_t>();
    CV_DbgAssert(N == 16);
    const int N0 = src.cols / N * N;
    const mipp::Reg<int8_t> rMaskLSB = 0x0F;
    const mipp::Reg<int8_t> rMaskMSB = 0xF0;
    for (int r = 0; r < src.rows; ++r) {
        const int8_t *src_r = (int8_t*)src.ptr(r);
        int8_t *prLSB = (int8_t*)lsb4.ptr(r);
        int8_t *prMSB = (int8_t*)msb4.ptr(r);

        for (int c = 0; c < N0; c += N) {
            mipp::Reg<int8_t> rSrc0(src_r + c);

            mipp::andb(rSrc0, rMaskLSB).store(prLSB);
            mipp::rshift(mipp::andb(rSrc0, rMaskMSB), 4).store(prMSB);

            prLSB += N;
            prMSB += N;
        }

        for (int c = N0, k = 0; c < N; ++c, ++k) {
            int8_t s = src.at<int8_t>(r, c);
            prLSB[k] = (int8_t)(s & 0x0F);
            prMSB[k] = (int8_t)((s & 0xF0) >> 4);
        }
    }

    uchar *lsb4_data = lsb4.ptr<uchar>();
    uchar *msb4_data = msb4.ptr<uchar>();

#pragma omp parallel for
    // For each of the 8 quantized orientations...
    for (int ori = 0; ori < 8; ++ori) {
        uchar *map_data = response_maps[ori].ptr<uchar>();
        const uchar *lut_low = SIMILARITY_LUT + 32 * ori;

        if(mipp::N<uint8_t>() == 16) { // 128 SIMD
            const uchar *lut_low = SIMILARITY_LUT + 32 * ori;
            mipp::Reg<uint8_t> lut_low_v((uint8_t*)lut_low);
            mipp::Reg<uint8_t> lut_high_v((uint8_t*)lut_low + 16);

            for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>()) {
                mipp::Reg<uint8_t> low_mask((uint8_t*)lsb4_data + i);
                mipp::Reg<uint8_t> high_mask((uint8_t*)msb4_data + i);

                mipp::Reg<uint8_t> low_res = mipp::shuff(lut_low_v, low_mask);
                mipp::Reg<uint8_t> high_res = mipp::shuff(lut_high_v, high_mask);

                mipp::Reg<uint8_t> result = mipp::max(low_res, high_res);
                result.store((uint8_t*)map_data + i);
            }
        }
        else {
            const uchar *lut_high = lut_low + 16;
            for (int i = 0; i < src.rows * src.cols; ++i) {
                map_data[i] = std::max(lut_low[lsb4_data[i]], lut_high[msb4_data[i]]);
            }
        }
    }
}

CV_DECL_ALIGNED(16) static const unsigned char SIMILARITY_LUT_16[] = {
    /* Generated by lut_gen_16 */

    /*
     * 0..3
     * 4..7
     * 8..11
     * 12..15
     */

    0, 8, 7, 8, 6, 8, 7, 8, 5, 8, 7, 8, 6, 8, 7, 8,
    0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4, 
    0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 
    0, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,

    0, 7, 8, 8, 7, 7, 8, 8, 6, 7, 8, 8, 7, 7, 8, 8, 
    0, 5, 4, 5, 3, 5, 4, 5, 2, 5, 4, 5, 3, 5, 4, 5, 
    0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    0, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6,
    
    0, 6, 7, 7, 8, 8, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8, 
    0, 6, 5, 6, 4, 6, 5, 6, 3, 6, 5, 6, 4, 6, 5, 6, 
    0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    0, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
    
    0, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8,
    0, 7, 6, 7, 5, 7, 6, 7, 4, 7, 6, 7, 5, 7, 6, 7, 
    0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3,
    0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
    
    0, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
    0, 8, 7, 8, 6, 8, 7, 8, 5, 8, 7, 8, 6, 8, 7, 8, 
    0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4,
    0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    
    0, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 
    0, 7, 8, 8, 7, 7, 8, 8, 6, 7, 8, 8, 7, 7, 8, 8, 
    0, 5, 4, 5, 3, 5, 4, 5, 2, 5, 4, 5, 3, 5, 4, 5,
    0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    
    0, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 
    0, 6, 7, 7, 8, 8, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8, 
    0, 6, 5, 6, 4, 6, 5, 6, 3, 6, 5, 6, 4, 6, 5, 6, 
    0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    
    0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 
    0, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 
    0, 7, 6, 7, 5, 7, 6, 7, 4, 7, 6, 7, 5, 7, 6, 7, 
    0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3,
    
    0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    0, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 
    0, 8, 7, 8, 6, 8, 7, 8, 5, 8, 7, 8, 6, 8, 7, 8, 
    0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4,
    
    0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 
    0, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 
    0, 7, 8, 8, 7, 7, 8, 8, 6, 7, 8, 8, 7, 7, 8, 8,
    0, 5, 4, 5, 3, 5, 4, 5, 2, 5, 4, 5, 3, 5, 4, 5,
    
    0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 
    0, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 
    0, 6, 7, 7, 8, 8, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8, 
    0, 6, 5, 6, 4, 6, 5, 6, 3, 6, 5, 6, 4, 6, 5, 6,
    
    0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3,
    0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 
    0, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8,
    0, 7, 6, 7, 5, 7, 6, 7, 4, 7, 6, 7, 5, 7, 6, 7,
    
    0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4,
    0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 
    0, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 
    0, 8, 7, 8, 6, 8, 7, 8, 5, 8, 7, 8, 6, 8, 7, 8,
    
    0, 5, 4, 5, 3, 5, 4, 5, 2, 5, 4, 5, 3, 5, 4, 5,
    0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 
    0, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6,
    0, 7, 8, 8, 7, 7, 8, 8, 6, 7, 8, 8, 7, 7, 8, 8,
    
    0, 6, 5, 6, 4, 6, 5, 6, 3, 6, 5, 6, 4, 6, 5, 6,
    0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 
    0, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
    0, 6, 7, 7, 8, 8, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8,
    
    0, 7, 6, 7, 5, 7, 6, 7, 4, 7, 6, 7, 5, 7, 6, 7,
    0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3, 
    0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 
    0, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8 
};

static void computeResponseMaps16(const Mat &src, std::vector<Mat> &response_maps)
{
    CV_Assert((src.rows * src.cols) % 16 == 0);

    // Allocate response maps
    response_maps.resize(16);
    for (int i = 0; i < 16; ++i) {
        response_maps[i].create(src.size(), CV_8U);
    }

    cv::Mat r0_3(src.size(), CV_8U);
    cv::Mat r4_7(src.size(), CV_8U);
    cv::Mat r8_11(src.size(), CV_8U);
    cv::Mat r12_15(src.size(), CV_8U);

    const int N = mipp::N<int16_t>();
    CV_DbgAssert(N == 8);
    const int N0 = src.cols / (N * 2) * (N * 2);
    const mipp::Reg<int16_t> rMask0_3   = 0x000F;
    const mipp::Reg<int16_t> rMask4_7   = 0x00F0;
    const mipp::Reg<int16_t> rMask8_11  = 0x0F00;
    const mipp::Reg<int16_t> rMask12_15 = 0xF000;
    for (int r = 0; r < src.rows; ++r) {
        const int16_t *src_r = (int16_t*)src.ptr(r);
        int8_t *pr0_3 = (int8_t*)r0_3.ptr(r);
        int8_t *pr4_7 = (int8_t*)r4_7.ptr(r);
        int8_t *pr8_11 = (int8_t*)r8_11.ptr(r);
        int8_t *pr12_15 = (int8_t*)r12_15.ptr(r);

        for (int c = 0; c < N0; c += 2 * N) {
            mipp::Reg<int16_t> rSrc0(src_r + c);
            mipp::Reg<int16_t> rSrc1(src_r + c + N);

            mipp::pack<int16_t, int8_t>(
                mipp::andb(rSrc0, rMask0_3),
                mipp::andb(rSrc1, rMask0_3)
            ).store(pr0_3);

            mipp::pack<int16_t, int8_t>(
                mipp::rshift(mipp::andb(rSrc0, rMask4_7), 4),
                mipp::rshift(mipp::andb(rSrc1, rMask4_7), 4)
            ).store(pr4_7);

            mipp::pack<int16_t, int8_t>(
                mipp::rshift(mipp::andb(rSrc0, rMask8_11), 8),
                mipp::rshift(mipp::andb(rSrc1, rMask8_11), 8)
            ).store(pr8_11);

            mipp::pack<int16_t, int8_t>(
                mipp::rshift(mipp::andb(rSrc0, rMask12_15), 12),
                mipp::rshift(mipp::andb(rSrc1, rMask12_15), 12)
            ).store(pr12_15);

            pr0_3 += N * 2;
            pr4_7 += N * 2;
            pr8_11 += N * 2;
            pr12_15 += N * 2;
        }

        for (int c = N0, k = 0; c < N; ++c, ++k) {
            int16_t s = src.at<int16_t>(r, c);
            pr0_3[k] = (int8_t)(s & 0x000F);
            pr4_7[k] = (int8_t)((s & 0x00F0) >> 4);
            pr8_11[k] = (int8_t)((s & 0x0F00) >> 8);
            pr12_15[k] = (int8_t)((s & 0xF000) >> 12);
        }
    }

    int8_t* pr0_3 = r0_3.ptr<int8_t>();
    int8_t* pr4_7 = r4_7.ptr<int8_t>();
    int8_t* pr8_11 = r8_11.ptr<int8_t>();
    int8_t* pr12_15 = r12_15.ptr<int8_t>();

#pragma omp parallel for
    // For each of the 16 quantized orientations...
    for (int ori = 0; ori < 16; ++ori) {
        int8_t *map_data = response_maps[ori].ptr<int8_t>();
        const int8_t *lut0_3 = (int8_t*)SIMILARITY_LUT_16 + 64 * ori;
        const int8_t *lut4_7 = lut0_3 + 16;
        const int8_t *lut8_11 = lut4_7 + 16;
        const int8_t *lut12_15 = lut8_11 + 16;

        if(mipp::N<int8_t>() == 16) { // 128 SIMD
            const int M = mipp::N<int8_t>();
            for (int i = 0; i < src.rows * src.cols; i += M) {
                mipp::Reg<int8_t> rA(pr0_3 + i);
                mipp::Reg<int8_t> rB(pr4_7 + i);
                mipp::Reg<int8_t> rC(pr8_11 + i);
                mipp::Reg<int8_t> rD(pr12_15 + i);
                mipp::Reg<int8_t> rLUTA(lut0_3);
                mipp::Reg<int8_t> rLUTB(lut4_7);
                mipp::Reg<int8_t> rLUTC(lut8_11);
                mipp::Reg<int8_t> rLUTD(lut12_15);

                mipp::max(
                    mipp::max(
                        mipp::shuff(rLUTA, rA), 
                        mipp::shuff(rLUTB, rB)),
                    mipp::max(
                        mipp::shuff(rLUTC, rC), 
                        mipp::shuff(rLUTD, rD))
                ).store(map_data);

                map_data += M;
            }
        }
        else {
            for (int i = 0; i < src.rows * src.cols; ++i) {
                map_data[i] = std::max(
                    std::max(lut0_3[pr0_3[i]], lut4_7[pr4_7[i]]),
                    std::max(lut8_11[pr8_11[i]], lut12_15[pr12_15[i]])
                );
            }
        }
    }
}

static void linearize2(const Mat &response_map, Mat &linearized)
{
    const auto S = mipp::N<int8_t>();
    const int M = response_map.rows;
    const int N = response_map.cols;
    const int N0 = N / 32 * 32;
    
    assert(S == 16); /* SSE */
    
    int mem_width = response_map.cols / 2;
    int mem_height = response_map.rows / 2;
    linearized.create(4, mem_width * mem_height, CV_8U);
    
    int8_t *rowMemory[4];
    for (int i = 0; i < 4; ++i) {
        rowMemory[i] = (int8_t*)linearized.ptr(i);
    }
    
    const int8_t shuffMask[16] = { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15 }; 
    mipp::Reg<int8_t> vShuffMask(shuffMask);    
    
    for (int r = 0; r < M; ++r) {
        int rWrapped = r % 2;
        
        const int8_t *p0 = (const int8_t*)response_map.ptr(r);
        
        for (int c = 0; c < N0; c += 32) {
            const int8_t *p = p0 + c;
            
            mipp::Reg<int8_t> v0(p), v1(p + S);
            mipp::Reg<int8_t> w0 = mipp::shuff(v0, vShuffMask);
            mipp::Reg<int8_t> w1 = mipp::shuff(v1, vShuffMask);
            
            mipp::Regx2<int8_t> s = mipp::interleave(w0, w1);
            
            mipp::Reg<int8_t> u0 = mipp::shuff(s[0], vShuffMask);
            mipp::Reg<int8_t> u1 = mipp::shuff(s[1], vShuffMask);
            
            u0.store(rowMemory[2 * rWrapped]);
            u1.store(rowMemory[2 * rWrapped + 1]);
            
            rowMemory[2 * rWrapped] += 16;
            rowMemory[2 * rWrapped + 1] += 16;
        }
        
        for (int c = N0; c < N; c += 2) {
            const int8_t *p = &response_map.at<int8_t>(r * N + c);
            for (int i = 0; i < 2; ++i) {
                *rowMemory[2 * rWrapped + i] = p[i];
                rowMemory[2 * rWrapped + i] += 1;
            }
        }
    }
}

static void linearize4(const Mat &response_map, Mat &linearized)
{
    const auto S = mipp::N<int8_t>();
    const int M = response_map.rows;
    const int N = response_map.cols;
    const int N0 = N / 64 * 64;
    
    assert(S == 16); /* SSE */
    
    int mem_width = response_map.cols / 4;
    int mem_height = response_map.rows / 4;
    linearized.create(16, mem_width * mem_height, CV_8U);
    
    int8_t *rowMemory[16];
    for (int i = 0; i < 16; ++i) {
        rowMemory[i] = (int8_t*)linearized.ptr(i);
    }
    
    const int8_t shuffMask[16] = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 }; 
    mipp::Reg<int8_t> vShuffMask(shuffMask);    
    
    for (int r = 0; r < M; ++r) {
        int rWrapped = r % 4;
        
        const int8_t *p0 = (const int8_t*)response_map.ptr(r);
        
        for (int c = 0; c < N0; c += 64) {
            const int8_t *p = p0 + c;
            
            mipp::Reg<int8_t> v0(p), v1(p + S), v2(p + 2 * S), v3(p + 3 * S);
            mipp::Reg<int8_t> w0 = mipp::shuff(v0, vShuffMask);
            mipp::Reg<int8_t> w1 = mipp::shuff(v1, vShuffMask);
            mipp::Reg<int8_t> w2 = mipp::shuff(v2, vShuffMask);
            mipp::Reg<int8_t> w3 = mipp::shuff(v3, vShuffMask);
            
            mipp::Regx2<int8_t> s0 = mipp::interleave(w0, w2);
            mipp::Regx2<int8_t> s1 = mipp::interleave(w1, w3);
            
            mipp::Regx2<int8_t> t0 = mipp::interleave(s0[0], s1[0]);
            mipp::Regx2<int8_t> t1 = mipp::interleave(s0[1], s1[1]);
            
            mipp::Reg<int8_t> u0 = mipp::shuff(t0[0], vShuffMask);
            mipp::Reg<int8_t> u1 = mipp::shuff(t0[1], vShuffMask);
            mipp::Reg<int8_t> u2 = mipp::shuff(t1[0], vShuffMask);
            mipp::Reg<int8_t> u3 = mipp::shuff(t1[1], vShuffMask);
            
            u0.store(rowMemory[4 * rWrapped]);
            u1.store(rowMemory[4 * rWrapped + 1]);
            u2.store(rowMemory[4 * rWrapped + 2]);
            u3.store(rowMemory[4 * rWrapped + 3]);
            
            rowMemory[4 * rWrapped] += 16;
            rowMemory[4 * rWrapped + 1] += 16;
            rowMemory[4 * rWrapped + 2] += 16;
            rowMemory[4 * rWrapped + 3] += 16;
        }
        
        for (int c = N0; c < N; c += 4) {
            const int8_t *p = &response_map.at<int8_t>(r * N + c);
            for (int i = 0; i < 4; ++i) {
                *rowMemory[4 * rWrapped + i] = p[i];
                rowMemory[4 * rWrapped + i] += 1;
            }
        }
    }
}

static void linearize8(const Mat &response_map, Mat &linearized)
{
    const auto S = mipp::N<int8_t>();
    const int M = response_map.rows;
    const int N = response_map.cols;
    const int N0 = N / 128 * 128;
    
    assert(S == 16); /* SSE */
    
    int mem_width = response_map.cols / 8;
    int mem_height = response_map.rows / 8;
    linearized.create(64, mem_width * mem_height, CV_8U);
    
    int8_t *rowMemory[64];
    for (int i = 0; i < 64; ++i) {
        rowMemory[i] = (int8_t*)linearized.ptr(i);
    }
    
    /* XMM0 - XMM15 on x64. */
    
    const int8_t shuffMask1[16] = { 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15 }; 
    mipp::Reg<int8_t> vShuffMask1(shuffMask1);
    
    const int8_t shuffMask2[16] = { 0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15 };
    mipp::Reg<int8_t> vShuffMask2(shuffMask2);
    
    for (int r = 0; r < M; ++r) {
        int rWrapped = r % 8;
        
        const int8_t *p0 = (const int8_t*)response_map.ptr(r);
        
        for (int c = 0; c < N0; c += 128) {
            const int8_t *p = p0 + c;
            
            mipp::Reg<int8_t> v0(p), v1(p + S), v2(p + 2 * S), v3(p + 3 * S),
                v4(p + 4 * S), v5(p + 5 * S), v6(p + 6 * S), v7(p + 7 * S);
            
            mipp::Reg<int8_t> w0 = mipp::shuff(v0, vShuffMask1);
            mipp::Reg<int8_t> w1 = mipp::shuff(v1, vShuffMask1);
            mipp::Reg<int8_t> w2 = mipp::shuff(v2, vShuffMask1);
            mipp::Reg<int8_t> w3 = mipp::shuff(v3, vShuffMask1);
            mipp::Reg<int8_t> w4 = mipp::shuff(v4, vShuffMask1);
            mipp::Reg<int8_t> w5 = mipp::shuff(v5, vShuffMask1);
            mipp::Reg<int8_t> w6 = mipp::shuff(v6, vShuffMask1);
            mipp::Reg<int8_t> w7 = mipp::shuff(v7, vShuffMask1);
            
            mipp::Regx2<int8_t> s0 = mipp::interleave(w0, w1);          /* 0 - 1 */
            mipp::Regx2<int8_t> s1 = mipp::interleave(w2, w3);          /* 2 - 3 */
            mipp::Regx2<int8_t> s2 = mipp::interleave(w4, w5);          /* 4 - 5 */
            mipp::Regx2<int8_t> s3 = mipp::interleave(w6, w7);          /* 6 - 7 */
            
            mipp::Regx2<int8_t> t0 = mipp::interleave(s0[0], s1[0]);    /* 0 - 2 */
            mipp::Regx2<int8_t> t1 = mipp::interleave(s0[1], s1[1]);    /* 1 - 3 */
            mipp::Regx2<int8_t> t2 = mipp::interleave(s2[0], s3[0]);    /* 4 - 6 */
            mipp::Regx2<int8_t> t3 = mipp::interleave(s2[1], s3[1]);    /* 5 - 7 */
            
            mipp::Regx2<int8_t> u0 = mipp::interleave(t0[0], t2[0]);    /* 0 - 4 */
            mipp::Regx2<int8_t> u1 = mipp::interleave(t0[1], t2[1]);    /* 1 - 5 */
            mipp::Regx2<int8_t> u2 = mipp::interleave(t1[0], t3[0]);    /* 2 - 6 */
            mipp::Regx2<int8_t> u3 = mipp::interleave(t1[1], t3[1]);    /* 3 - 7 */
            
            mipp::Reg<int8_t> q0 = mipp::shuff(u0[0], vShuffMask2);
            mipp::Reg<int8_t> q1 = mipp::shuff(u0[1], vShuffMask2);
            mipp::Reg<int8_t> q2 = mipp::shuff(u1[0], vShuffMask2);
            mipp::Reg<int8_t> q3 = mipp::shuff(u1[1], vShuffMask2);
            mipp::Reg<int8_t> q4 = mipp::shuff(u2[0], vShuffMask2);
            mipp::Reg<int8_t> q5 = mipp::shuff(u2[1], vShuffMask2);
            mipp::Reg<int8_t> q6 = mipp::shuff(u3[0], vShuffMask2);
            mipp::Reg<int8_t> q7 = mipp::shuff(u3[1], vShuffMask2);
            
            q0.store(rowMemory[8 * rWrapped]);
            q1.store(rowMemory[8 * rWrapped + 1]);
            q2.store(rowMemory[8 * rWrapped + 2]);
            q3.store(rowMemory[8 * rWrapped + 3]);
            q4.store(rowMemory[8 * rWrapped + 4]);
            q5.store(rowMemory[8 * rWrapped + 5]);
            q6.store(rowMemory[8 * rWrapped + 6]);
            q7.store(rowMemory[8 * rWrapped + 7]);
            
            rowMemory[8 * rWrapped] += 16;
            rowMemory[8 * rWrapped + 1] += 16;
            rowMemory[8 * rWrapped + 2] += 16;
            rowMemory[8 * rWrapped + 3] += 16;
            rowMemory[8 * rWrapped + 4] += 16;
            rowMemory[8 * rWrapped + 5] += 16;
            rowMemory[8 * rWrapped + 6] += 16;
            rowMemory[8 * rWrapped + 7] += 16;
        }
        
        for (int c = N0; c < N; c += 8) {
            const int8_t *p = &response_map.at<int8_t>(r * N + c);
            for (int i = 0; i < 8; ++i) {
                *rowMemory[8 * rWrapped + i] = p[i];
                rowMemory[8 * rWrapped + i] += 1;
            }
        }
    }
}

static void linearize(const Mat &response_map, Mat &linearized, int T)
{
    CV_Assert(response_map.rows % T == 0);
    CV_Assert(response_map.cols % T == 0);
    
    // SIMD implementations.
    if (T == 2) {
        linearize2(response_map, linearized);
        return;
    }
    else if (T == 4) {
        linearize4(response_map, linearized);
        return;
    }
    else if (T == 8) {
        linearize8(response_map, linearized);
        return;
    }
    
    // linearized has T^2 rows, where each row is a linear memory
    int mem_width = response_map.cols / T;
    int mem_height = response_map.rows / T;
    linearized.create(T * T, mem_width * mem_height, CV_8U);

    // Outer two for loops iterate over top-left T^2 starting pixels
    int index = 0;
    for (int r_start = 0; r_start < T; ++r_start) {
        for (int c_start = 0; c_start < T; ++c_start) {
            uchar *memory = linearized.ptr(index);
            ++index;

            // Inner two loops copy every T-th pixel into the linear memory
            for (int r = r_start; r < response_map.rows; r += T) {
                const uchar *response_data = response_map.ptr(r);
                for (int c = c_start; c < response_map.cols; c += T) {
                    *memory++ = response_data[c];
                }
            }
        }
    }
}

static const unsigned char *accessLinearMemory(const std::vector<Mat> &linear_memories,
           const Feature &f, int T, int W)
{
    // Retrieve the TxT grid of linear memories associated with the feature label
    const Mat &memory_grid = linear_memories[f.label];
    CV_DbgAssert(memory_grid.rows == T * T);
    CV_DbgAssert(f.x >= 0);
    CV_DbgAssert(f.y >= 0);

    // The LM we want is at (x%T, y%T) in the TxT grid (stored as the rows of memory_grid)
    int grid_x = f.x % T;
    int grid_y = f.y % T;
    int grid_index = grid_y * T + grid_x;
    CV_DbgAssert(grid_index >= 0);
    CV_DbgAssert(grid_index < memory_grid.rows);
    const unsigned char *memory = memory_grid.ptr(grid_index);
    
    // Within the LM, the feature is at (x/T, y/T). W is the "width" of the LM, the
    // input image width decimated by T.
    int lm_x = f.x / T;
    int lm_y = f.y / T;
    int lm_index = lm_y * W + lm_x;
    CV_DbgAssert(lm_index >= 0);
    CV_DbgAssert(lm_index < memory_grid.cols);

    return memory + lm_index;
}

static void similarity(const std::vector<Mat> &linear_memories, const Template &templ,
        Mat &dst, Size size, int T)
{
    // we only have one modality, so 8192*2, due to mipp, back to 8192
    CV_Assert(templ.features.size() < 8192);

    // Decimate input image size by factor of T
    int W = size.width / T;
    int H = size.height / T;

    // Feature dimensions, decimated by factor T and rounded up
    int wf = (templ.width - 1) / T + 1;
    int hf = (templ.height - 1) / T + 1;

    // Span is the range over which we can shift the template around the input image
    int span_x = W - wf;
    int span_y = H - hf;

    int template_positions = span_y * W + span_x + 1; // why add 1?

    dst = Mat::zeros(H, W, CV_16U);
    short *dst_ptr = dst.ptr<short>();

    mipp::Reg<uint8_t> zero_v(uint8_t(0));

    for (int i = 0; i < std::min((int)templ.features.size(), max_features); ++i) {
        Feature f = templ.features[i];

        if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height) {
            continue;
        }

        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);
        int j = 0;
        const int N = mipp::N<int16_t>();

        // Double N() to avoid int8 read out of range
        for(; j <= template_positions - 2 * N; j += N) {
            mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr + j);
            mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);
            mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + j);
            mipp::Reg<int16_t> res_v = src16_v + dst_v;
            res_v.store((int16_t*)dst_ptr + j);
        }

        for(; j<template_positions; j++) {
            dst_ptr[j] += short(lm_ptr[j]);
        }
    }
}

static void similarityLocal(const std::vector<Mat> &linear_memories, const Template &templ,
                            Mat &dst, Size size, int T, Point center)
{
    CV_Assert(templ.features.size() < 8192);

    int W = size.width / T;
    dst = Mat::zeros(16, 16, CV_16U);

    int offset_x = (center.x / T - 8) * T;
    int offset_y = (center.y / T - 8) * T;
    mipp::Reg<uint8_t> zero_v = uint8_t(0);

    for (int i = 0; i < std::min((int)templ.features.size(), max_features); ++i) {
        Feature f = templ.features[i];
        f.x += offset_x;
        f.y += offset_y;

        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height) {
            continue;
        }

        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);
        short *dst_ptr = dst.ptr<short>();
        const int N = mipp::N<int16_t>();

        for (int row = 0; row < 16; ++row) {
            for(int col = 0; col < 16; col += N) {
                mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr + col);

                // uchar to short, once for N bytes
                mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

                mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + col);
                mipp::Reg<int16_t> res_v = src16_v + dst_v;
                res_v.store((int16_t*)dst_ptr + col);
            }

            dst_ptr += 16;
            lm_ptr += W;
        }
    }
}

static Rect cropTemplates(std::vector<Template> &templates)
{
    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int max_y = std::numeric_limits<int>::min();

    // First pass: find min/max feature x,y over all pyramid levels and modalities
    for (int i = 0; i < (int)templates.size(); ++i) {
        Template &templ = templates[i];

        for (int j = 0; j < (int)templ.features.size(); ++j) {
            int x = templ.features[j].x << templ.pyramid_level;
            int y = templ.features[j].y << templ.pyramid_level;
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
        }
    }

    if (min_x % 2 == 1) {
        --min_x;
    }

    if (min_y % 2 == 1) {
        --min_y;
    }

    // Second pass: set width/height and shift all feature positions
    for (int i = 0; i < (int)templates.size(); ++i) {
        Template &templ = templates[i];

        templ.width = (max_x - min_x) >> templ.pyramid_level;
        templ.height = (max_y - min_y) >> templ.pyramid_level;
        templ.tl_x = min_x >> templ.pyramid_level;
        templ.tl_y = min_y >> templ.pyramid_level;

        for (int j = 0; j < (int)templ.features.size(); ++j) {
            templ.features[j].x -= templ.tl_x;
            templ.features[j].y -= templ.tl_y;
        }
    }

    return Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

static cv::Point2f rotate2d(const cv::Point2f inPoint, const double angRad)
{
    cv::Point2f outPoint;
    //CW rotation
    outPoint.x = std::cos(angRad)*inPoint.x - std::sin(angRad)*inPoint.y;
    outPoint.y = std::sin(angRad)*inPoint.x + std::cos(angRad)*inPoint.y;
    return outPoint;
}

static cv::Point2f rotatePoint(const cv::Point2f inPoint, const cv::Point2f center, const double angRad)
{
    return rotate2d(inPoint - center, angRad) + center;
}

// Used to filter out weak matches
struct MatchPredicate
{
    MatchPredicate(float _threshold) : threshold(_threshold) {}
    bool operator()(const Match &m) { return m.similarity < threshold; }
    float threshold;
};

Detector::Detector()
{
    this->num_ori = 16;
    this->max_gradient = 360.0f;
    this->debug = false;
    this->modality = makePtr<ColorGradient>(false, 16);
    pyramid_levels = 2;
    T_at_level.push_back(4);
    T_at_level.push_back(8);
}

Detector::Detector(std::vector<int> T, int num_ori)
{
    this->num_ori = num_ori;
    this->max_gradient = 360.0f;
    this->debug = false;
    this->modality = makePtr<ColorGradient>(false, num_ori);
    pyramid_levels = T.size();
    T_at_level = T;
}

Detector::Detector(int num_features, std::vector<int> T, float weak_thresh,
    float strong_thresh, float max_gradient, int num_ori)
{
    if (num_features <= 63) {
        num_features = 63;
    }
    else if (num_features > 8190) {
        num_features = 8190;
    }

    this->max_gradient = max_gradient;
    this->num_ori = num_ori;
    this->debug = false;
    this->modality = makePtr<ColorGradient>(weak_thresh, num_features, strong_thresh,
            false, num_ori, max_gradient);

    if (!T.empty()) {
        pyramid_levels = T.size();
        if (T[0] > 0) {
            T_at_level = T;
        }
        else {
            int N = pyramid_levels;
            int T0 = (T[0] == 0) ? 4 : -T[0];
            for (int i = 1, k = 1; i <= N; ++i, k *= 2) {
                T_at_level.push_back(k * T0);
            }
        }
    }
    else {
        pyramid_levels = 2;
        T_at_level.push_back(4);
        T_at_level.push_back(8);    
    }
}

std::vector<Match> Detector::match(Mat source, float threshold, 
    Ptr<ColorGradientPyramid> &quantizer,
    const std::vector<std::string> &class_ids, const Mat mask, 
    int num_max_matches) const
{
    Timer timer;
    std::vector<Match> matches;

    // Initialize each ColorGradient with our sources
    CV_Assert(mask.empty() || mask.size() == source.size());
    quantizer = modality->process(source, mask);

    // pyramid level -> ColorGradient -> quantization
    LinearMemoryPyramid lm_pyramid(pyramid_levels,
            std::vector<LinearMemories>(1, LinearMemories(num_ori)));

    // For each pyramid level, precompute linear memories for each ColorGradient
    std::vector<Size> sizes;
    sizes.reserve(pyramid_levels);

    for (int l = 0; l < pyramid_levels; ++l) {
        int T = T_at_level[l];
        std::vector<LinearMemories> &lm_level = lm_pyramid[l];
        
        if (debug) {
            std::cout << "L = " << l << ", T = " << T << std::endl;
        }

        if (l > 0) {
            quantizer->pyrDown();
        }

        Mat quantized, spread_quantized;
        std::vector<Mat> response_maps;
        
        Timer tmr;
        
        quantizer->quantize(quantized);
        if (debug) {
            tmr.out("Quantize");
        }

        spread(quantized, spread_quantized, T);
        if (debug) {
            tmr.out("Spread");
        }

        if (num_ori == 8) {
            computeResponseMaps(spread_quantized, response_maps);
        }
        else {
            computeResponseMaps16(spread_quantized, response_maps);
        }
        
        if (debug) {
            tmr.out("Compute response maps");
        }

        LinearMemories &memories = lm_level[0];
        #pragma omp parallel for
        for (int j = 0; j < num_ori; ++j) {
            linearize(response_maps[j], memories[j], T);
        }

        if (debug) {
            tmr.out("Linearize response maps");
        }

        sizes.push_back(quantized.size());
    }

    if (debug) {
        timer.out("Construct response map");
    }

    if (class_ids.empty())
    {
        // Match all templates
        TemplatesMap::const_iterator it = class_templates.begin(), itend = class_templates.end();
        for (; it != itend; ++it) {
            matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second, num_max_matches);
        }
    }
    else
    {
        // Match only templates for the requested class IDs
        for (int i = 0; i < (int)class_ids.size(); ++i) {
            TemplatesMap::const_iterator it = class_templates.find(class_ids[i]);
            if (it != class_templates.end()) {
                matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second, num_max_matches);
            }
        }
    }

    // Sort matches by similarity, and prune any duplicates introduced by pyramid refinement
    std::sort(matches.begin(), matches.end());
    std::vector<Match>::iterator new_end = std::unique(matches.begin(), matches.end());
    matches.erase(new_end, matches.end());

    if (debug) {
        timer.out("Template matching");
    }

    return matches;
}

int Detector::addTemplate(const Mat source, const std::string &class_id,
        const Mat &object_mask, int num_features)
{
    std::vector<TemplatePyramid> &template_pyramids = class_templates[class_id];
    int template_id = static_cast<int>(template_pyramids.size());

    TemplatePyramid tp;
    tp.resize(pyramid_levels);

    // Extract a template at each pyramid level
    Ptr<ColorGradientPyramid> qp = modality->process(source, object_mask);

    if (num_features > 0) {
        qp->num_features = num_features;
    }

    for (int l = 0; l < pyramid_levels; ++l) {
        if (l > 0) {
            qp->pyrDown(false);
        }

        bool success = qp->extractTemplate(tp[l]);
        if (!success) {
            return -1;
        }
    }

    cropTemplates(tp);

    template_pyramids.push_back(tp);
    return template_id;
}

int Detector::addTemplateRotate(const string &class_id, int zero_id,
        float theta, cv::Point2f center)
{
    std::vector<TemplatePyramid> &template_pyramids = class_templates[class_id];
    int template_id = static_cast<int>(template_pyramids.size());

    const auto& to_rotate_tp = template_pyramids[zero_id];

    TemplatePyramid tp;
    tp.resize(pyramid_levels);

    for (int l = 0; l < pyramid_levels; ++l) {
        if (l>0) {
            center /= 2;
        }

        for (auto& f: to_rotate_tp[l].features) {
            Point2f p;
            p.x = f.x + to_rotate_tp[l].tl_x;
            p.y = f.y + to_rotate_tp[l].tl_y;
            Point2f p_rot = rotatePoint(p, center, -theta/180*CV_PI);

            Feature f_new;
            f_new.x = int(p_rot.x + 0.5f);
            f_new.y = int(p_rot.y + 0.5f);

            f_new.theta = f.theta - theta;
            while(f_new.theta > 360) {
                f_new.theta -= 360;
            }

            while(f_new.theta < 0) {
                f_new.theta += 360;
            }

            if (num_ori == 8) {
                f_new.label = int(f_new.theta * 16 / 360 + 0.5f);
                f_new.label &= 0x07;
            }
            else {
                f_new.label = int(f_new.theta * 32 / 360 + 0.5f);
                f_new.label &= 0x0F;
            }

            tp[l].features.push_back(f_new);
        }
        tp[l].pyramid_level = l;
    }

    cropTemplates(tp);

    template_pyramids.push_back(tp);
    return template_id;
}

const std::vector<Template> &Detector::getTemplates(const std::string &class_id, int template_id) const
{
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    CV_Assert(i != class_templates.end());
    CV_Assert(i->second.size() > size_t(template_id));
    return i->second[template_id];
}

int Detector::numTemplates() const
{
    int ret = 0;
    TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
    for (; i != iend; ++i) {
        ret += static_cast<int>(i->second.size());
    }
    return ret;
}

int Detector::numTemplates(const std::string &class_id) const
{
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    if (i == class_templates.end()) {
        return 0;
    }
    return static_cast<int>(i->second.size());
}

std::vector<std::string> Detector::classIds() const
{
    std::vector<std::string> ids;
    TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
    for (; i != iend; ++i) {
        ids.push_back(i->first);
    }

    return ids;
}

void Detector::read(const FileNode &fn)
{
    class_templates.clear();
    pyramid_levels = fn["pyramid_levels"];
    fn["T"] >> T_at_level;

    modality = makePtr<ColorGradient>();
    const FileNode& fnModality = fn["Modality"];
    modality->read(fnModality);

    this->num_ori = modality->num_ori;
}

void Detector::write(FileStorage &fs) const
{
    fs << "pyramid_levels" << pyramid_levels;
    fs << "T" << T_at_level;

    fs << "Modality" << "{" ;
    modality->write(fs);
    fs << "}";
}

std::string Detector::readClass(const FileNode &fn, const std::string &class_id_override)
{
    // Detector should not already have this class
    String class_id;
    if (class_id_override.empty()) {
        String class_id_tmp = fn["class_id"];
        CV_Assert(class_templates.find(class_id_tmp) == class_templates.end());
        class_id = class_id_tmp;
    }
    else {
        class_id = class_id_override;
    }

    TemplatesMap::value_type v(class_id, std::vector<TemplatePyramid>());
    std::vector<TemplatePyramid> &tps = v.second;
    int expected_id = 0;

    FileNode tps_fn = fn["template_pyramids"];
    tps.resize(tps_fn.size());
    FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
    for (; tps_it != tps_it_end; ++tps_it, ++expected_id) {
        int template_id = (*tps_it)["template_id"];
        CV_Assert(template_id == expected_id);
        FileNode templates_fn = (*tps_it)["templates"];
        tps[template_id].resize(templates_fn.size());

        FileNodeIterator templ_it = templates_fn.begin(), templ_it_end = templates_fn.end();
        int idx = 0;
        for (; templ_it != templ_it_end; ++templ_it) {
            tps[template_id][idx++].read(*templ_it);
        }
    }

    class_templates.insert(v);
    return class_id;
}

void Detector::writeClass(const std::string &class_id, FileStorage &fs) const
{
    TemplatesMap::const_iterator it = class_templates.find(class_id);
    CV_Assert(it != class_templates.end());
    const std::vector<TemplatePyramid> &tps = it->second;

    fs << "{";
    fs << "class_id" << it->first;
    fs << "pyramid_levels" << pyramid_levels;
    fs << "template_pyramids"
       << "[";
    for (size_t i = 0; i < tps.size(); ++i) {
        const TemplatePyramid &tp = tps[i];
        fs << "{";
        fs << "template_id" << int(i);
        fs << "templates"
           << "[";

        for (size_t j = 0; j < tp.size(); ++j) {
            fs << "{";
            tp[j].write(fs);
            fs << "}"; // current template
        }

        fs << "]"; // templates
        fs << "}"; // current pyramid
    }
    fs << "]"; // pyramids
    fs << "}";
}

void Detector::readClasses(const cv::FileNode& fn)
{
    for (FileNodeIterator it = fn.begin(); it != fn.end(); ++it) {
        const FileNode& fnClass = *it;
        readClass(fnClass);
    }
}

void Detector::writeClasses(cv::FileStorage &fs) const
{
    fs << "Classes" << "[" ;
    
    TemplatesMap::const_iterator it = class_templates.begin(), it_end = class_templates.end();
    for (; it != it_end; ++it) {
        const String &class_id = it->first;
        writeClass(class_id, fs);
    }
    
    fs << "]";
}

void Detector::matchClass(const LinearMemoryPyramid &lm_pyramid,
    const std::vector<Size> &sizes,
    float threshold, std::vector<Match> &matches,
    const std::string &class_id,
    const std::vector<TemplatePyramid> &template_pyramids,
    int num_max_matches) const
{
    if ((num_max_matches <= 0) || (num_max_matches > 1024)) {
        num_max_matches = 1024;
    }
    
#pragma omp declare reduction \
    (omp_insert: std::vector<Match>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

#pragma omp parallel for reduction(omp_insert:matches)
    for (size_t template_id = 0; template_id < template_pyramids.size(); ++template_id) {
        const TemplatePyramid &tp = template_pyramids[template_id];
        // First match over the whole image at the lowest pyramid level
        /// @todo Factor this out into separate function
        const std::vector<LinearMemories> &lowest_lm = lm_pyramid.back();
        const int maxResponse = (num_ori == 8) ? 4 : 8;

        std::vector<int> bestIndices;
        std::vector<Match> candidates, bestCandidates;
        {
            // Compute similarity maps for each ColorGradient at lowest pyramid level
            Mat similarities;
            int lowest_start = static_cast<int>(tp.size() - 1);
            int lowest_T = T_at_level.back();
            int num_features = 0;

            const Template &templ = tp[lowest_start];
            num_features += static_cast<int>(templ.features.size());

            if (templ.features.size() < 8192) {
                similarity(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
            }
            else {
                CV_Error(Error::StsBadArg, "feature size too large");
            }

            // Find initial matches
            for (int r = 0; r < similarities.rows; ++r) {
                ushort *row = similarities.ptr<ushort>(r);
                for (int c = 0; c < similarities.cols; ++c) {
                    int raw_score = row[c];
                    float score = (raw_score * 100.f) / (maxResponse * num_features);

                    if (score > threshold) {
                        int offset = lowest_T / 2 + (lowest_T % 2 - 1);
                        int x = c * lowest_T + offset;
                        int y = r * lowest_T + offset;
                        candidates.push_back(Match(x, y, score, class_id, static_cast<int>(template_id)));
                    }
                }
            }
            
            std::vector<cv::Rect> boundBoxes;
            std::vector<float> scores;
            for (const Match& match : candidates) {
                boundBoxes.emplace_back(match.x, match.y, templ.width, templ.height);
                scores.push_back(match.similarity);
            }
            ::NMSBoxes(boundBoxes, scores, 0.9 * threshold, 0.5f, bestIndices);
            
            bestCandidates.clear();
            size_t N = std::min((size_t)(2 * num_max_matches), bestIndices.size());
            bestCandidates.reserve(N);
            for (size_t k = 0; k != N; ++k) {
                bestCandidates.push_back(candidates[bestIndices[k]]);
            }
            
            candidates = std::move(bestCandidates);
        }
        
        // Locally refine each match by marching up the pyramid
        for (int l = pyramid_levels - 2; l >= 0; --l) {
            const std::vector<LinearMemories> &lms = lm_pyramid[l];
            int T = T_at_level[l];
            int start = static_cast<int>(l);
            Size size = sizes[l];
            int border = 8 * T;
            int offset = T / 2 + (T % 2 - 1);
            int max_x = size.width - tp[start].width - border;
            int max_y = size.height - tp[start].height - border;

            Mat similarities2;
            for (int m = 0; m < (int)candidates.size(); ++m) {
                Match &match2 = candidates[m];
                if (match2.similarity < 0) {
                    continue;
                }
                
                int x = match2.x * 2 + 1;
                int y = match2.y * 2 + 1;

                // Require 8 (reduced) row/cols to the up/left
                x = std::max(x, border);
                y = std::max(y, border);

                // Require 8 (reduced) row/cols to the down/left, plus the template size
                x = std::min(x, max_x);
                y = std::min(y, max_y);

                // Compute local similarity maps for each ColorGradient
                int numFeatures = 0;
                const Template &templ = tp[start];
                numFeatures += static_cast<int>(templ.features.size());

                if (templ.features.size() < 8192) {
                    similarityLocal(lms[0], templ, similarities2, size, T, Point(x, y));
                }
                else {
                    CV_Error(Error::StsBadArg, "feature size too large");
                }

                // Find best local adjustment
                float best_score = 0;
                int best_r = -1, best_c = -1;
                for (int r = 0; r < similarities2.rows; ++r) {
                    ushort *row = similarities2.ptr<ushort>(r);
                    for (int c = 0; c < similarities2.cols; ++c) {
                        int score_int = row[c];
                        float score = (score_int * 100.f) / (maxResponse * numFeatures);

                        if (score > best_score) {
                            best_score = score;
                            best_r = r;
                            best_c = c;
                        }
                    }
                }
                
                if (best_score < threshold) {
                    match2.similarity = -1.0f;
                    continue;
                }
                
                // Update current match
                match2.similarity = best_score;
                match2.x = (x / T - 8) * T + best_c * T + offset;
                match2.y = (y / T - 8) * T + best_r * T + offset;
            }
            
            // Filter out any matches that drop below the similarity threshold
            std::vector<Match>::iterator new_end 
                = std::remove_if(candidates.begin(), candidates.end(),
                    MatchPredicate(threshold));
            candidates.erase(new_end, candidates.end());
        }

        matches.insert(matches.end(), candidates.begin(), candidates.end());
    }
}

} // namespace Line2Dup
