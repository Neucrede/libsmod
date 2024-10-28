#include "icp.h"
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>

namespace icp
{

static Mat3x3f eigen_to_custom(const Eigen::Matrix3f& extrinsic){
    Mat3x3f result;
    for(uint32_t i=0; i<3; i++){
        for(uint32_t j=0; j<3; j++){
            result[i][j] = extrinsic(i, j);
        }
    }
    return result;
}

static void transform_pcd(std::vector<Vec2f>& model_pcd, Mat3x3f& trans)
{
    #pragma omp parallel for
    for(uint32_t i=0; i < model_pcd.size(); i++){
        Vec2f& pcd = model_pcd[i];
        float new_x = trans[0][0]*pcd.x + trans[0][1]*pcd.y + trans[0][2];
        float new_y = trans[1][0]*pcd.x + trans[1][1]*pcd.y + trans[1][2];
        pcd.x = new_x;
        pcd.y = new_y;
    }
}

//////////////////////////////////////////////////////////////////////////////

static Eigen::Matrix3d TransformVector4dToMatrix3d(const Eigen::Matrix<double, 4, 1> &input) 
{
    // input = [ angle, tx, ty, scale ];

    // rotate
    Eigen::Matrix3d output =
            (Eigen::AngleAxisd(input(0), Eigen::Vector3d::UnitZ())).matrix();

    // scale
    output.block<2, 2>(0, 0) *= (1 + input[3]);

    // translate
    output.block<2, 1>(0, 2) = input.block<2, 1>(1, 0);
    return output;
}

static Mat3x3f eigen_slover_444(float *A, float *b)
{
    Eigen::Matrix<float, 4, 4> A_eigen(A);
    Eigen::Matrix<float, 4, 1> b_eigen(b);
    // ICP point to plane may be unstable, refer to
    // https://www.cs.princeton.edu/~smr/papers/icpstability.pdf
    // add a term ||x|| to make update reasonably small:
    // f = ||(Rp + T - q) * n|| + penalty * ||X||   ==>
    // (ATA + Identity * penalty) * X = B
    Eigen::Matrix4d iden = Eigen::Matrix4d::Identity();
    double penalty = 0.01;
    Eigen::Matrix4d ATA_with_pen = A_eigen.cast<double>() + penalty*iden;
    
    const Eigen::Matrix<double, 4, 1> update = ATA_with_pen.ldlt().solve(b_eigen.cast<double>());
    Eigen::Matrix3d extrinsic = TransformVector4dToMatrix3d(update);

    return eigen_to_custom(extrinsic.cast<float>());
}

template <class Scene>
RegistrationResult ICP2D_Point2Plane_4DoF
(
    std::vector<Vec2f> &model_pcd, 
    const Scene scene,
    const ConvergenceCriteria criteria,
    bool debug
)
{
    RegistrationResult result;
    RegistrationResult prevResult;

    std::vector<float> A(16, 0);
    std::vector<float> b(4, 0);
    PCD2Ab_4DoF<Scene> transformer(scene);

    // use one extra turn
    for(uint32_t iter=0; iter <= criteria.max_iteration_; iter++) {
        Vec16f reducer;

        #pragma omp declare reduction( + : Vec16f : omp_out += omp_in) \
                                initializer (omp_priv = Vec16f::Zero())
        #pragma omp parallel for reduction(+: reducer)
        for(size_t pcd_iter=0; pcd_iter<model_pcd.size(); pcd_iter++) {
            Vec16f result = transformer(model_pcd[pcd_iter]);
            reducer += result;
        }

        Vec16f& Ab_tight = reducer;

        prevResult = result;

        float& count = Ab_tight[15];
        float& total_error = Ab_tight[14];

        if(count == 0) {
            break;  // avoid division by 0
        }

        result.fitness_ = float(count) / model_pcd.size();
        result.inlier_rmse_ = std::sqrt(total_error / count);

        // last extra iter, just compute fitness & mse
        if (iter == criteria.max_iteration_) {
            break;
        }

        if(std::abs(result.fitness_ - prevResult.fitness_) < criteria.relative_fitness_ &&
           std::abs(result.inlier_rmse_ - prevResult.inlier_rmse_) < criteria.relative_rmse_ )
        {
            break;
        }

        for(int i=0; i<4; i++) {
            b[i] = Ab_tight[10 + i];
        }

        int shift = 0;
        for(int y=0; y<4; y++){
            for(int x=y; x<4; x++){
                A[x + y*4] = Ab_tight[shift];
                A[y + x*4] = Ab_tight[shift];
                shift++;
            }
        }

        Mat3x3f extrinsic = eigen_slover_444(A.data(), b.data());

        transform_pcd(model_pcd, extrinsic);
        result.transformation_ = extrinsic * result.transformation_;

        /*
        if (debug) {
            std::cout << "ICP iteration #" << iter+1 << " : fitness = " << result.fitness_
                << ", inlier_rmse = " << result.inlier_rmse_ << std::endl;
        }
        */
    }

    return result;
}

template
RegistrationResult ICP2D_Point2Plane_4DoF
(
    std::vector<Vec2f> &model_pcd, 
    const Scene_edge scene,
    const ConvergenceCriteria criteria,
    bool debug
);

//////////////////////////////////////////////////////////////////////////////

static Eigen::Matrix3d TransformVector5dToMatrix3d(const Eigen::Matrix<double, 5, 1> &input) 
{
    // input = [ angle, tx, ty, scale-x, scale-y ];

    // rotate
    Eigen::Matrix3d output =
            (Eigen::AngleAxisd(input(0), Eigen::Vector3d::UnitZ())).matrix();

    // scale
    output.block<2, 2>(0, 0) *= Eigen::DiagonalMatrix<double, 2>(1.0 + input[3], 1.0 + input[4]);

    // translate
    output.block<2, 1>(0, 2) = input.block<2, 1>(1, 0);

    return output;
}

static Mat3x3f eigen_slover_555(float *A, float *b)
{
    Eigen::Matrix<float, 5, 5> A_eigen(A);
    Eigen::Matrix<float, 5, 1> b_eigen(b);

    // ICP point to plane may be unstable, refer to
    // https://www.cs.princeton.edu/~smr/papers/icpstability.pdf
    // add a term ||x|| to make update reasonably small:
    // f = ||(Rp + T - q) * n|| + penalty * ||X||   ==>
    // (ATA + Identity * penalty) * X = B
    //
    Eigen::Matrix<double, 5, 5> iden = Eigen::Matrix<double, 5, 5>::Identity();
    double penalty = 0.05;
    Eigen::Matrix<double, 5, 5> ATA_with_pen = A_eigen.cast<double>() + penalty*iden;
    
    const Eigen::Matrix<double, 5, 1> update = ATA_with_pen.ldlt().solve(b_eigen.cast<double>());
    Eigen::Matrix3d extrinsic = TransformVector5dToMatrix3d(update);

    return eigen_to_custom(extrinsic.cast<float>());
}

template <class Scene>
RegistrationResult ICP2D_Point2Plane_5DoF
(
    std::vector<Vec2f> &model_pcd, 
    const Scene scene,
    const ConvergenceCriteria criteria,
    bool debug
)
{
    RegistrationResult result;
    RegistrationResult prevResult;

    std::vector<float> A(25, 0);
    std::vector<float> b(5, 0);
    PCD2Ab_5DoF<Scene> transformer(scene);

    // use one extra turn
    for(uint32_t iter=0; iter <= criteria.max_iteration_; iter++) {
        Vec22f reducer;

        #pragma omp declare reduction( + : Vec22f : omp_out += omp_in) \
                                initializer (omp_priv = Vec22f::Zero())
        #pragma omp parallel for reduction(+: reducer)
        for(size_t pcd_iter=0; pcd_iter<model_pcd.size(); pcd_iter++) {
            Vec22f result = transformer(model_pcd[pcd_iter]);
            reducer += result;
        }

        Vec22f& Ab_tight = reducer;

        prevResult = result;

        float& count = Ab_tight[21];
        float& total_error = Ab_tight[20];

        if(count == 0) {
            break;  // avoid division by 0
        }

        result.fitness_ = float(count) / model_pcd.size();
        result.inlier_rmse_ = std::sqrt(total_error / count);

        // last extra iter, just compute fitness & mse
        if (iter == criteria.max_iteration_) {
            break;
        }

        if( std::abs(result.fitness_ - prevResult.fitness_) < criteria.relative_fitness_ &&
           std::abs(result.inlier_rmse_ - prevResult.inlier_rmse_) < criteria.relative_rmse_ )
        {
            break;
        }

        for (int i=0; i<5; i++) {
            b[i] = Ab_tight[15 + i];
        }

        int shift = 0;
        for (int y=0; y<5; y++){
            for(int x=y; x<5; x++){
                A[x + y*5] = Ab_tight[shift];
                A[y + x*5] = Ab_tight[shift];
                shift++;
            }
        }

        Mat3x3f extrinsic = eigen_slover_555(A.data(), b.data());

        transform_pcd(model_pcd, extrinsic);
        result.transformation_ = extrinsic * result.transformation_;
    }

    return result;
}

template
RegistrationResult ICP2D_Point2Plane_5DoF
(
    std::vector<Vec2f> &model_pcd, 
    const Scene_edge scene,
    const ConvergenceCriteria criteria,
    bool debug
);

}





