/**
 * \file smod.h
 *
 * \author Neucrede     liqinpeng2014@sina.com
 * \version 1.1 
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

/** Default class ID. */
extern const char* defaultClassId;
    
/**
 * \brief Object detection result.
 */
struct Result
{
    /** Default constructor. */
    Result();
    
    /** Constructor. */
    Result(const cv::Point2f& centre, const cv::Size2f& size, float angle,
        float _scale, float _score, const std::string& _classId);

    /** Constructor. */
    Result(const cv::Point2f& centre, const cv::Size2f& size, float angle,
        float _scaleX, float _scaleY, float _score, const std::string& _classId);
    

    /** Stores the position, orientation and unscaled size of the object found. */
    cv::RotatedRect region;
    /** Average scale of the object relative to the template. */
    float scale;
    /** Scale along X direction. */
    float scaleX;
    /** Scale along Y direction. */
    float scaleY;
    /** Shape model matching score (similarity %). */
    float score;
    /** Gray-value comparison score (Pearson coefficient %). */
    float scorePearson;
    /** Template class ID. */
    std::string classId;
};

/** Stream output function. */
std::ostream& operator << (std::ostream& os, const Result& result);

/**
 * \brief Base class.
 */
class ShapeModelObjectDetectorBase
{
public:
    struct TemplateInfo
    {
        /** Class ID. */
        std::string classId;
        /** Padding width. */
        int dw;
        /** Padding height. */
        int dh;
        /** Size of the template image. */
        cv::Size size;
        /** Template image. */
        cv::Mat imgTmpl;
        /** Mean value of template image. */
        float mean;
        /** Sum of squared pixel values of template image. */
        float sqsum;
        /** Information of isotropic scalings and rotations. */
        SMOD::ShapeInfo shapeInfo;
    };

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
     * \param [in] refine               Enable / disable localization results refinement.
     * \param [in] numFeatures          Maximum number of features. Must not exceed 8190.
     * \param [in] T                    Size of the orientation spreading neighbourhood
     *                                  for each pyramid level. The number of levels of 
     *                                  the image pyramid is the same as the number of
     *                                  elements of T. Setting T to a vector filled with 
     *                                  n zeros will result an image pyramid of n levels
     *                                  and a T_at_level array of {4, 8, 16, ...}.
     * \param [in] weakThreshold        Lower threshold used in hysteresis thresholding
     *                                  on gradient magnitudes. Value ranged in 0 to 360. 
     * \param [in] strongThreshold      Upper threshold used in hysteresis thresholding
     *                                  on gradient magnitudes. Value ranged in 0 to 360. 
     * \param [in] maxGradient          Maximum gradient magnitude allowed. Value ranged in 0 to 360. 
     * \param [in] numOri               Number of quantization bins. Must be 8 or 16 (default).
     */
    ShapeModelObjectDetectorBase( 
        float angleMin = 0.0f, float angleMax = 360.0f, float angleStep = 1.0f,
        float scaleMin = 0.8f, float scaleMax = 1.2f, float scaleStep = 0.1f,
        bool refine = true,
        int numFeatures = 128, 
        const std::vector<int>& T = { 4, 8 },
        float weakThreshold = 30.0f, float strongThreshold = 60.0f, 
        float maxGradient = 360.0f, int numOri = 16);
    
    /** Destructor. */
    virtual ~ShapeModelObjectDetectorBase();
    
    /** Load model from file specified by <code>filename</code>. */
    virtual bool Load(const std::string& filename) = 0;
    
    /** Save model to file named <code>filename</code>. */
    virtual bool Save(const std::string& filename) const = 0;
    
    /** 
     * \brief Register template.
     *
     * Provided for compatibility with v1.0.
     *
     * \param [in] img      Template image.
     * \param [in] mask     Optional mask image.
     * \param [in] fast     Enable / disable fast feature extraction.
     * \param [in] classId  Template class ID. Only effective in
     *                      \ref ShapeModelObjectDetectorEx.
     */
    virtual bool Register(const cv::Mat& img, const cv::Mat& mask = cv::Mat(),
        bool fast = false, const std::string& classId = defaultClassId) = 0;

    /**
     * \brief Extract image of matched region from source image.
     *
     * \param [in] src              Source image.
     * \param [inout] dst           Image of extracted region.
     * \param [in] region           Region.
     * \param [in] scaleX           Scale along X direction.
     * \param [in] scaleY           Scale along Y direction.
     * \param [in] scaleContentToRegionSize
     *                              If true, the size of the content will be the
     *                              same as the size of <code>region</code>.
     * \param [in] enableImageSizeScaling    
     *                              Enable / disable image scaling. If false,
     *                              the resulting image will have the same size
     *                              as <code>region.size</code>.
     * \param [in] imageSize        Destination image size if 
     *                              <code>enableImageSizeScaling</code> is false.                              
     */
    static void CropImage(const cv::Mat& src, cv::Mat& dst,
        const cv::RotatedRect& region, float scaleX, float scaleY,
        bool scaleContentToRegionSize = false, bool enableImageSizeScaling = true, 
        cv::Size imageSize = cv::Size());
    
    /** Provided for convenience. */
    static void CropImage(const cv::Mat& src, cv::Mat& dst,
        const Result& result, bool scaleContentToRegionSize = false, 
        bool enableImageSizeScaling = true,
        cv::Size imageSize = cv::Size());

    /** Compute size of paddings for image <code>src</code>. */
    static void ComputePaddings(const cv::Mat& src, int& dw, int& dh);
        
    /** Enable / disable detection result refinement. */
    void SetRefine(bool refine = true);
    
    /** Returns <code>true</code> if refinement is enabled. */
    bool GetRefine() const;
        
    /** Set grayvalue comparison threshold. The valid value range is from 0 to 100. */
    void SetGVCompareThreshold(float thresh = 60.0f);

    /** Returns grayvalue comparison threshold. */
    float GetGVCompareThreshold() const;
        
    /** Enable / disable grayvalue comparison. */
    void SetGVCompare(bool enable = true);

    /** Returns <code>true</code> if grayvalue compare is enabled. */
    bool GetGVCompare() const;
        
    /** Enable / disable debug info printout. */
    void SetDebug(bool debug = true);

    /** Returns <code>true</code> if debug enabled. */
    bool GetDebug() const;
        
    /** Specifies the path of the image on which regions of match were drawn. */
    void SetDebugImagePath(const std::string& path);

    /** Returns debug image path. */
    std::string GetDebugImagePath() const;
        
    /** Get the reference of internal <code>Line2Dup::Detector</code> object. */
    const Line2Dup::Detector& GetDetector() const;

    /** Enable / disable anisotropic scaling refinement. */
    void SetRefineAnisoScaling(bool enabled = true);

    /** Returns <code>true</code> if anisotropic scaling refinement is enabled. */
    bool GetRefineAnisoScaling() const;

    /** Set size of closest point searching neibourhood. */
    void SetMaxDistDiff(float maxDistDiff);

    /** Returns size of closest point searching neibourhood. */
    float GetMaxDistDiff() const;

    /** Set the maximum overlap ratio between bounding boxes of two instances. */
    void SetMaxOverlap(float maxOverlap);

    /** Returns the maximum overlap ratio. */
    float GetMaxOverlap() const;

    /** 
     * Set the damping factor used by adaptive non-maximum supression. 
     * A small damping factor value may leads to an aggressive NMS scenario.
     *
     * A value of 0 disables adaptive method. 
     */
    void SetNMSDampFactor(float damp);

    /** Returns adaptive NMS damping factor. */
    float GetNMSDampFactor() const;
        
protected:

    /** 
     * \brief Register multiple templates.
     *
     * \param [in] classIds Template class IDs / names.
     * \param [in] imgs     Template images.
     * \param [in] fast     Enable / disable fast feature extraction.
     * \param [in] masks    Optional mask images.
     */
    virtual int RegisterEx(
        const std::vector<std::string>& classIds,
        const std::vector<cv::Mat>& imgs, 
        bool fast = false,
        const std::vector<cv::Mat>& masks = std::vector<cv::Mat>() );

    /** Internal use. */
    bool RegisterInternal(const cv::Mat& img, const cv::Mat& mask, bool fast, 
        const std::string& classId, std::vector<Line2Dup::TemplatePyramid>& tps,
        TemplateInfo& tmplInfo);

    /**
     * \brief Search for template matches.
     *
     * \param [in] src              Source image.
     * \param [out] results         Vector to which detection results were stored.
     * \param [in] threshold        Score threshold range from 0 to 100.
     * \param [in] maxNumMatches    Maximum number of matches.
     * \param [in] classIds         Vector of class IDs. Use all registered
     *                              template if left empty.
     * \param [in] mask             Optional 8-bit single channel mask. 
     */
    virtual bool Detect(cv::Mat& src, std::vector<Result>& results,
        float threshold = 75.0f, int maxNumMatches = 1,
        const std::vector<std::string>& classIds = {},
        const cv::Mat& mask = cv::Mat()) const;

    /** Initialize data needed by Detect(). */
    void InitDetect(const cv::Mat& src, int maxNumMatches, cv::Mat& srcGray, 
        cv::Mat& srcPadded, cv::Mat& imgDbg, int& maxNumMatches1, cv::Point& offset) const;

    /** Compute Pearson correlation coefficient between template image and source image. */
    float ComputePearson(const cv::Mat& src, const TemplateInfo& tmplInfo) const;
        
    /** Compute mean and sqsum of the given image. */
    void ComputeMeanSqsum(const cv::Mat& img, float &mean, float &sqsum) const;

protected:
    SMOD::ShapeInfo m_shapeInfo0;
        
protected:
    /** Stores information of each template class. */
    std::map<std::string, TemplateInfo> m_tmplInfos;

    /** Detector object. */
    Line2Dup::Detector m_detector;
    
    /** Debugging. */
    bool m_debug;
    /** Path of the image on which match regions were plotted. */
    std::string m_debugImagePath;
    
    /** Refinement. */
    bool m_refine;
    /** Grayvalue comparison Threshold. */
    float m_gvCompareThreshold;
    /** Enable grayvalue comparison. */
    bool m_enableGVCompare;
    /** Refine anisotropic scaling. */
    bool m_refineAnisoScaling;
    /** Half-size of the square neighbourhood used for closest point searching. */
    float m_maxDistDiff;
    /** Maximum overlap ratio. */
    float m_maxOverlap;
    /** Adaptive NMS damping factor. */
    float m_nmsDampFactor;
};

/** 
 * \brief Single template implementation.
 *
 * Recommended using \ref ShapeModelObjectDetectorEx .
 */
class ShapeModelObjectDetector : public ShapeModelObjectDetectorBase
{
public:
    /** 
     * \brief Constructor. 
     *
     * \see \ref ShapeModelObjectDetectorBase::ShapeModelObjectDetectorBase()
     */
    ShapeModelObjectDetector( 
        float angleMin = 0.0f, float angleMax = 360.0f, float angleStep = 1.0f,
        float scaleMin = 0.8f, float scaleMax = 1.2f, float scaleStep = 0.1f,
        bool refine = true,
        int numFeatures = 128, 
        const std::vector<int>& T = { 4, 8 },
        float weakThreshold = 30.0f, float strongThreshold = 60.0f, 
        float maxGradient = 360.0f, int numOri = 16);
    
    /** Destructor. */
    virtual ~ShapeModelObjectDetector();

    bool Load(const std::string& filename) override;
    
    bool Save(const std::string& filename) const override;

    bool Register(const cv::Mat& img, const cv::Mat& mask = cv::Mat(),
        bool fast = false, const std::string& classId = defaultClassId) override;

    /**
     * \brief Search for template matches.
     *
     * \param [in] src              Source image.
     * \param [out] results         Vector to which detection results were stored.
     * \param [in] threshold        Score threshold range from 0 to 100.
     * \param [in] maxNumMatches    Maximum number of matches.
     */
    bool Detect(cv::Mat& src, std::vector<Result>& results,
        float threshold = 75.0f, int maxNumMatches = 1) const;
};

/** 
 * \brief Multiple templates implementation.
 */
class ShapeModelObjectDetectorEx : public ShapeModelObjectDetectorBase
{
public:
    /** 
     * \brief Constructor. 
     *
     * \see \ref ShapeModelObjectDetectorBase::ShapeModelObjectDetectorBase()
     */
    ShapeModelObjectDetectorEx( 
        float angleMin = 0.0f, float angleMax = 360.0f, float angleStep = 1.0f,
        float scaleMin = 0.8f, float scaleMax = 1.2f, float scaleStep = 0.1f,
        bool refine = true,
        int numFeatures = 128, 
        const std::vector<int>& T = { 4, 8 },
        float weakThreshold = 30.0f, float strongThreshold = 60.0f, 
        float maxGradient = 360.0f, int numOri = 16);
    
    /** Destructor. */
    virtual ~ShapeModelObjectDetectorEx();

    bool Load(const std::string& filename) override;
    
    bool Save(const std::string& filename) const override;
    
    bool Register(const cv::Mat& img, const cv::Mat& mask = cv::Mat(),
        bool fast = false, const std::string& classId = defaultClassId) override;
    
public:
    using ShapeModelObjectDetectorBase::RegisterEx;
    using ShapeModelObjectDetectorBase::Detect;
};

} 

#endif        //  #ifndef SMOD_H
