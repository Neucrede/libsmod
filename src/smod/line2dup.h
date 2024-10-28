#ifndef CXXLINEMOD_H
#define CXXLINEMOD_H

#include <map>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

namespace SMOD
{
    class ShapeModelObjectDetectorBase;
}

namespace Line2Dup
{

// ----------------------------------------------------------------------------
    
struct Feature
{
    int x;
    int y;
    int label;
    float theta;

    Feature() 
    : x(0), y(0), label(0) 
    {
    }
    
    Feature(int _x, int _y, int _label)
    : x(_x), y(_y), label(_label)
    {
    }
    
    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;//
};

// ----------------------------------------------------------------------------

struct Template
{
    int width;
    int height;
    int tl_x;
    int tl_y;
    int pyramid_level;
    std::vector<Feature> features;

    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;
};

// ----------------------------------------------------------------------------

class ColorGradientPyramid
{
public:
    /// Candidate feature with a score
    struct Candidate
    {
        Feature f;
        float score;
        
        Candidate(int x, int y, int label, float _score)
        : f(x, y, label), score(_score)
        {
        }

        /// Sort candidates with high score to the front
        bool operator<(const Candidate &rhs) const
        {
            return score > rhs.score;
        }
    };

public:
    ColorGradientPyramid(const cv::Mat &src, const cv::Mat &mask,
        float weak_threshold, size_t num_features, float strong_threshold,
        bool debug = false, int num_ori = 16, float max_gradient = 360.0f);
    
    void update(bool fast = true);
    void pyrDown(bool fast = true);
    void quantize(cv::Mat &dst) const;
    bool extractTemplate(Template &templ) const;

protected:
    int getLabel(int quantized) const;
    bool selectScatteredFeatures(const std::vector<Candidate> &candidates,
        std::vector<Feature> &features, size_t num_features, float distance) const;
    void hysteresisGradient(cv::Mat &magnitude, cv::Mat &quantized_angle,
        cv::Mat &angle, float threshold);
    void quantizedOrientations(const cv::Mat &src, cv::Mat &magnitude,
        cv::Mat &angle, cv::Mat& angle_ori, float threshold, bool fast = true);

public:
    cv::Mat src;
    cv::Mat mask;

    int pyramid_level;
    cv::Mat angle;
    cv::Mat magnitude;
    cv::Mat angle_ori;
    
    cv::Mat sobel_dx0;
    cv::Mat sobel_dy0;
    
    cv::Mat sobel_dx_prev;
    cv::Mat sobel_dy_prev;

    float weak_threshold;
    size_t num_features;
    float strong_threshold;
    float max_gradient;
    
    bool debug;
    int num_ori;
};

// ----------------------------------------------------------------------------

class ColorGradient
{
public:
    ColorGradient(bool debug = false, int num_ori = 16);
    ColorGradient(float weak_threshold, size_t num_features, float strong_threshold,
        bool debug = false, int num_ori = 16, float max_gradient = 360.0f);

    std::string name() const;

    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;

    cv::Ptr<ColorGradientPyramid> process(const cv::Mat src, const cv::Mat &mask = cv::Mat()) const;
    
public:
    float weak_threshold;
    size_t num_features;
    float strong_threshold;

    bool debug;
    int num_ori;
    float max_gradient;
};

// ----------------------------------------------------------------------------

struct Match
{
    int x;
    int y;
    float similarity;
    std::string class_id;
    int template_id;
    
    Match()
    {
    }
    
    Match(int _x, int _y, float _similarity, const std::string &_class_id, 
        int _template_id)
    : x(_x), y(_y), similarity(_similarity), class_id(_class_id), 
        template_id(_template_id)
    {
    }

    /// Sort matches with high similarity to the front
    bool operator<(const Match &rhs) const
    {
        // Secondarily sort on template_id for the sake of duplicate removal
        if (similarity != rhs.similarity) {
            return similarity > rhs.similarity;
        }
        else {
            return template_id < rhs.template_id;
        }
    }

    bool operator==(const Match &rhs) const
    {
        return x == rhs.x && y == rhs.y && similarity == rhs.similarity && class_id == rhs.class_id;
    }
};

// ----------------------------------------------------------------------------

typedef std::vector<Template> TemplatePyramid;

class Detector
{
public:
    typedef std::map<std::string, std::vector<TemplatePyramid>> TemplatesMap;
    typedef std::vector<cv::Mat> LinearMemories;

    // Indexed as [pyramid level][ColorGradient][quantized label]
    typedef std::vector<std::vector<LinearMemories>> LinearMemoryPyramid;

    typedef std::map<std::string, std::vector<Match>> MatchesMap;

public:
    Detector();
    Detector(std::vector<int> T, int num_ori = 16);
    Detector(int num_features, std::vector<int> T, float weak_thresh = 30.0f, 
        float strong_thresh = 60.0f, float max_gradient = 360.0f, int num_ori = 16);

    MatchesMap match(cv::Mat sources, float threshold,
        cv::Ptr<ColorGradientPyramid> &quantizer,
        const std::vector<std::string> &class_ids = std::vector<std::string>(),
        const cv::Mat mask = cv::Mat(), int num_max_matches = -1,
        float nms_thresh = 0.5) const;
    
    bool makeTemplate(const cv::Mat source, const cv::Mat &object_mask, 
        TemplatePyramid& tp, int num_features = 0);
    bool addTemplate(const cv::Mat source, const std::string &class_id,
        const cv::Mat &object_mask, int num_features = 0);
    bool addTemplate(const std::string &class_id, const TemplatePyramid& tp);
    bool makeTemplateRotate(const TemplatePyramid& tp0,
        float theta, cv::Point2f center, TemplatePyramid& tp);
    
    bool pruneClassTemplate(const std::string &class_id);
    const TemplatePyramid& getTemplates(const std::string &class_id, int template_id) const;
    int numTemplates() const;
    int numTemplates(const std::string &class_id) const;
    std::vector<std::string> classIds() const;
    
    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;
    std::string readClass(const cv::FileNode &fn, const std::string &class_id_override = "");
    void writeClass(const std::string &class_id, cv::FileStorage &fs) const;
    void readClasses(const cv::FileNode& fn);
    void writeClasses(cv::FileStorage &fs) const;
    
    void setDebug(bool dbg) 
    { 
        debug = dbg; 
    }
    
    const cv::Ptr<ColorGradient> &getModalities() const 
    { 
        return modality; 
    }
    
    int getT(int pyramid_level) const 
    { 
        return T_at_level[pyramid_level]; 
    }
    
    int pyramidLevels() const 
    { 
        return pyramid_levels; 
    }
    
    int numClasses() const 
    {
        return static_cast<int>(class_templates.size()); 
    }
    
protected:
    void matchClass(const LinearMemoryPyramid &lm_pyramid,
        const std::vector<cv::Size> &sizes,
        float threshold, std::vector<Match> &matches,
        const std::string &class_id,
        const std::vector<TemplatePyramid> &template_pyramids,
        int num_max_matches, float nms_thresh) const;
    
protected:
    friend class SMOD::ShapeModelObjectDetectorBase;

    int num_ori;
    float max_gradient;
    bool debug;
    
    cv::Ptr<ColorGradient> modality;
    int pyramid_levels;
    std::vector<int> T_at_level;

    TemplatesMap class_templates;
};

} // namespace Line2Dup

#endif
