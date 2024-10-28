#pragma once

#include "geometry.h"
#include "edge_scene.h"

namespace icp 
{

struct RegistrationResult
{
    RegistrationResult(const Mat3x3f &transformation = Mat3x3f::identity()) 
    : transformation_(transformation), inlier_rmse_(0.0), fitness_(0.0)
    {
    }

    Mat3x3f transformation_;
    float inlier_rmse_;
    float fitness_;
};

struct ConvergenceCriteria
{
    ConvergenceCriteria(float relative_fitness = 1e-6f,
        float relative_rmse = 1e-6f, int max_iteration = 100) 
    : relative_fitness_(relative_fitness), relative_rmse_(relative_rmse),
        max_iteration_(max_iteration) 
    {
    }

    float relative_fitness_;
    float relative_rmse_;
    int max_iteration_;
};

//////////////////////////////////////////////////////////////////////////////

template <class Scene>
RegistrationResult ICP2D_Point2Plane_4DoF
(
    std::vector<Vec2f>& model_pcd,
    const Scene scene,
    const ConvergenceCriteria criteria = ConvergenceCriteria(),
    bool debug = false
);

extern template
RegistrationResult ICP2D_Point2Plane_4DoF
(
    std::vector<Vec2f> &model_pcd, 
    const Scene_edge scene,
    const ConvergenceCriteria criteria,
    bool debug
);

// 4 DoF tight: A(symetric 4x4 --> (16-4)/2+4) + ATb 4 + mse(b*b 1) + count 1 = 16
typedef vec<16,  float> Vec16f;

template<class Scene>
struct PCD2Ab_4DoF
{
    Scene __scene;

    PCD2Ab_4DoF(Scene scene): __scene(scene)
    {

    }

    Vec16f operator()(const Vec2f &src_pcd) const 
    {
        Vec16f result;
        Vec2f dst_pcd, dst_normal; 
        bool valid;
        __scene.query(src_pcd, dst_pcd, dst_normal, valid);

        if(!valid) {
            return result;
        }
        else{
            result[15] = 1;  // number of valid points.
            
            // dot
            float b_temp = (dst_pcd - src_pcd).x * dst_normal.x +
                          (dst_pcd - src_pcd).y * dst_normal.y;
            result[14] = b_temp*b_temp; // mse

            // cross
            float A_temp[4];
            A_temp[0] = dst_normal.y*src_pcd.x - dst_normal.x*src_pcd.y;
            A_temp[1] = dst_normal.x;
            A_temp[2] = dst_normal.y;
            A_temp[3] = src_pcd.x*dst_normal.x + src_pcd.y*dst_normal.y;

            // result[0..9] stores values of lower triangular part of (A^T)A in
            // column-wise order.
            //
            // 0  x  x  x
            // 1  4  x  x
            // 2  5  7  x
            // 3  6  8  9
            result[ 0] = A_temp[0] * A_temp[0];
            result[ 1] = A_temp[0] * A_temp[1];
            result[ 2] = A_temp[0] * A_temp[2];
            result[ 3] = A_temp[0] * A_temp[3];

            result[ 4] = A_temp[1] * A_temp[1];
            result[ 5] = A_temp[1] * A_temp[2];
            result[ 6] = A_temp[1] * A_temp[3];

            result[ 7] = A_temp[2] * A_temp[2];
            result[ 8] = A_temp[2] * A_temp[3];

            result[ 9] = A_temp[3] * A_temp[3];

            // (A^T)b
            result[10] = A_temp[0] * b_temp;
            result[11] = A_temp[1] * b_temp;
            result[12] = A_temp[2] * b_temp;
            result[13] = A_temp[3] * b_temp;

            return result;
        }
    }
};

//////////////////////////////////////////////////////////////////////////////

template <class Scene>
RegistrationResult ICP2D_Point2Plane_5DoF
(
    std::vector<Vec2f>& model_pcd,
    const Scene scene,
    const ConvergenceCriteria criteria = ConvergenceCriteria(),
    bool debug = false
);

extern template
RegistrationResult ICP2D_Point2Plane_5DoF
(
    std::vector<Vec2f> &model_pcd, 
    const Scene_edge scene,
    const ConvergenceCriteria criteria,
    bool debug
);

// 5 DoF tight: A(symetric 5x5 --> (5*5-5)/2+5) + ATb 5 + mse(b*b 1) + count 1 = 22
typedef vec<22,  float> Vec22f;

template<class Scene>
struct PCD2Ab_5DoF
{
    Scene __scene;

    PCD2Ab_5DoF(Scene scene): __scene(scene)
    {

    }

    Vec22f operator()(const Vec2f &src_pcd) const 
    {
        Vec22f result;

        Vec2f dst_pcd, dst_normal; 
        bool valid;
        __scene.query(src_pcd, dst_pcd, dst_normal, valid);

        if(!valid) {
            return result;
        }
        else{
            result[21] = 1;  // number of valid points.
            
            // dot
            float b_temp = (dst_pcd - src_pcd).x * dst_normal.x +
                          (dst_pcd - src_pcd).y * dst_normal.y;
            result[20] = b_temp*b_temp; // mse

            // cross
            float A_temp[5];
            A_temp[0] = dst_normal.y * src_pcd.x - dst_normal.x * src_pcd.y;
            A_temp[1] = dst_normal.x;
            A_temp[2] = dst_normal.y;
            A_temp[3] = src_pcd.x * dst_normal.x;
            A_temp[4] = src_pcd.y * dst_normal.y;

            // result[0..14] stores values of lower triangular part of (A^T)A in
            // column-wise order.
            //
            // 0  x  x  x  x
            // 1  5  x  x  x
            // 2  6  9  x  x
            // 3  7  10 12 x
            // 4  8  11 13 14
            result[ 0] = A_temp[0] * A_temp[0];
            result[ 1] = A_temp[0] * A_temp[1];
            result[ 2] = A_temp[0] * A_temp[2];
            result[ 3] = A_temp[0] * A_temp[3];
            result[ 4] = A_temp[0] * A_temp[4];

            result[ 5] = A_temp[1] * A_temp[1];
            result[ 6] = A_temp[1] * A_temp[2];
            result[ 7] = A_temp[1] * A_temp[3];
            result[ 8] = A_temp[1] * A_temp[4];

            result[ 9] = A_temp[2] * A_temp[2];
            result[10] = A_temp[2] * A_temp[3];
            result[11] = A_temp[2] * A_temp[4];

            result[12] = A_temp[3] * A_temp[3];
            result[13] = A_temp[3] * A_temp[4];

            result[14] = A_temp[4] * A_temp[4];

            // (A^T)b
            result[15] = A_temp[0] * b_temp;
            result[16] = A_temp[1] * b_temp;
            result[17] = A_temp[2] * b_temp;
            result[18] = A_temp[3] * b_temp;
            result[19] = A_temp[4] * b_temp;

            return result;
        }
    }
};

}


