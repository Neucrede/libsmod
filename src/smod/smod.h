/**
 * \file smod.h
 *
 * \author Neucrede     liqinpeng2014@sina.com
 * \version 1.0 
 *
 * \brief Shape Model Object Detector
 *
 */

#ifndef SMOD_H
#define SMOD_H

#include <string>
#include <iostream>
#include <iomanip>
#include "line2dup.h"
#include "shapeinfo.h"

namespace SMOD {

/**
 * \brief Object detection result.
 */
struct Result
{
    /** Default constructor. */
    Result() 
    {}
    
    /** Constructor. */
    Result(const cv::Point2f& centre, const cv::Size2f& size, float angle,
        float _scale, float _score)
    : region(centre, size, angle), scale(_scale), score(_score)
    {}
    
    /** Stream output function. */
    friend std::ostream& operator << (std::ostream& os, const Result& result)
    {
        const cv::RotatedRect& region = result.region;
        
        return 
            os  << std::fixed << std::setprecision(2)
                << "Region: { " 
                    << region.center << ", " 
                    << region.size << ", " 
                    << region.angle 
                << "}"
                << ", Scale = " << result.scale
                << ", Score = " << result.score ;
    }
    

    /** Stores the position, orientation and unscaled size of the object found. */
    cv::RotatedRect region;
    /** Scale of the object relative to the template. */
    float scale;
    /** Score. */
    float score;
};
    
/**
 * \brief Object detector class.
 */
class ShapeModelObjectDetector
{
public:
    /**
     * \brief Constructor.
     *
     * \param [in] angleMin             Minimum rotation angle relative to the template,
     *                                  in degrees.
     * \param [in] angleMax             Maximum rotation angle relative to the template,
     *                                  in degrees.
     * \param [in] angleStep            Stepping of rotation angle in degrees.
     * \param [in] scaleMin             Minimum scale relative to the template.
     * \param [in] scaleMax             Maximum scale relative to the template.
     * \param [in] scaleStep            Stepping of scale factor.
     * \param [in] refine               To refine detection results or not.
     * \param [in] numFeatures          Maximum number of features. Must not exceed 8190.
     * \param [in] T                    Size of the orientation spreading neighbourhood
     *                                  for each pyramid level. The number of levels of 
     *                                  the image pyramid is the same as the number of
     *                                  elements of T. Setting T to a vector filled with 
     *                                  n zeros will result an image pyramid of n levels
     *                                  and a T_at_level array of {4, 8, 16, ...}.
     * \param [in] weakThreshold        Lower threshold used in hysteresis thresholing
     *                                  on gradient magnitudes. The valid value range
     *                                  is from 0 to 361.
     * \param [in] strongThreshold      Upper threshold used in hysteresis thresholing
     *                                  on gradient magnitudes. The valid value range
     *                                  is from 0 to 361.
     * \param [in] maxGradient          Maximum gradient magnitude allowed. The valid value range
     *                                  is from 0 to 361.
     * \param [in] numOri               Number of quantization bins. Must be 8 or 16 (default).
     * \param [in] gvCompareThreshold   Threshold value of grayvalue comparison.
     *                                  The valid value range is from 0 to 100.
     * \param [in] enableGVCompare      Whether to enable grayvalue comparison.
     */
    ShapeModelObjectDetector( 
        float angleMin = 0.0f, float angleMax = 360.0f, float angleStep = 1.0f,
        float scaleMin = 0.8f, float scaleMax = 1.2f, float scaleStep = 0.1f,
        bool refine = true,
        int numFeatures = 128, 
        const std::vector<int>& T = { 4, 8 },
        float weakThreshold = 30.0f, float strongThreshold = 60.0f, 
        float maxGradient = 255.0f, int numOri = 16, float gvCompareThreshold = 60.0f,
        bool enableGVCompare = false);
    
    /** Destructor. */
    ~ShapeModelObjectDetector();
    
    /** Load model from file specified by <code>filename</code>. */
    bool Load(const std::string& filename);
    
    /** Save model to file named <code>filename</code>. */
    bool Save(const std::string& filename) const;

    /** 
     * \brief Register template.
     *
     * \param [in] img      Template image.
     * \param [in] mask     Optional mask image.
     * \param [in] fast     To use fast feature extraction method or not.
     */
    bool Register(const cv::Mat& img, const cv::Mat& mask = cv::Mat(),
        bool fast = false);

    /**
     * \brief Search for template matches.
     *
     * \param [in] src              Source image.
     * \param [inout] results       Vector that stores detection results.
     * \param [in] threshold        Score threshold range from 0 to 100.
     * \param [in] maxNumMatches    Maximum number of matches.
     */
    bool Detect(cv::Mat& src, std::vector<Result>& results,
        float threshold = 90, int maxNumMatches = 1);
    
    /**
     * \brief Extract image of matched region from source image.
     *
     * \param [in] src              Source image.
     * \param [inout] dst           Image of extracted region.
     * \param [in] region           Region.
     * \param [in] scale            Scale.
     * \param [in] enableScaling    Enable image scaling or not. If false,
     *                              the resulting image will have the same size
     *                              as <code>region.size</code>.
     */
    static void CropImage(const cv::Mat& src, cv::Mat& dst,
        const cv::RotatedRect& region, float scale, bool enableScaling = true);
    
    /** Provided for convenience. */
    static void CropImage(const cv::Mat& src, cv::Mat& dst,
        const Result& result, bool enableScaling = true);

    /** Enable / disable detection result refinement. */
    void SetRefine(bool refine = true);
        
    /** Set grayvalue comparison threshold. The valid value range is from 0 to 100. */
    void SetGVCompareThreshold(float thresh);
        
    /** Enable / disable grayvalue comparison. */
    void SetGVCompare(bool enable = true);
        
    /** Enable / disable debug info printout. */
    void SetDebug(bool debug = true);
        
    /** Specifies the path of the image on which regions of match were drawn. */
    void SetDebugImagePath(const std::string& path);
        
    /** Get the reference of internal <code>Line2Dup::Detector</code> object. */
    Line2Dup::Detector& GetDetector();
        
    /** Get size of paddings. */
    void GetPaddings(int &dw, int &dh);
    
protected:
    /** Compute size of paddings for image <code>src</code>. */
    void ComputePaddings(const cv::Mat& src);
        
    /** Compute Pearson correlation coefficient between template image and source image. */
    float ComputePearson(const cv::Mat& src);
        
    /** Compute mean and sqsum of the given image. */
    void ComputeMeanSqsum(const cv::Mat& img, float &mean, float &sqsum);
        
protected:
    /** Padding width. */
    int m_dw;
    /** Padding height. */
    int m_dh;
    /** Size of the template image. */
    cv::Size m_tmplSize;
    /** Detector object. */
    Line2Dup::Detector m_detector;
    /** Class ID. */
    const std::string m_classId;
    
    /** Debugging. */
    bool m_debug;
    /** Path of the image on which match regions were drawn. */
    std::string m_debugImagePath;
    
    /** Padded source image. */
    cv::Mat m_srcPadded;
    /** Size of the original source image. */
    cv::Size m_srcSize;
    
    /** Template image. */
    cv::Mat m_imgTmpl;
    /** Mean value of template image. */
    float m_tmplMean;
    /** Sum of squared pixel values of template image. */
    float m_tmplSqsum;
    
public:
    /** Information of scaling and rotation. */
    SMOD::ShapeInfo m_shapeInfo;
    /** Refinement. */
    bool m_refine;
    /** Grayvalue comparison Threshold. */
    float m_gvCompareThreshold;
    /** Enable grayvalue comparison. */
    bool m_enableGVCompare;
};

}

#endif        //  #ifndef SMOD_H
