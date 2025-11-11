#pragma once

#include <sl/Camera.hpp>
#include <sl/CameraOne.hpp>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <cmath>

constexpr int MIN_IMAGE = 20;

struct CameraCalib{
    cv::Mat K;
    cv::Mat D;
    bool disto_model_RadTan = true;

    void print(const std::string& name) const {
      std::cout << name << " K:" << std::endl << K << std::endl;
      std::cout << " D:" << std::endl << D << std::endl;
    }

    void initDefault(bool radtan){
        disto_model_RadTan = radtan;
        K = cv::Mat::eye(3, 3, CV_64FC1);
        if (disto_model_RadTan) {
            // Radial and tangential distortion
            const int nb_coeff = 8; // 6 radial + 2 tangential; could be extended to 12 with prism distortion
            D = cv::Mat::zeros(1, nb_coeff, CV_64FC1);
        } else {
            // Fisheye model has 4 coefficients: k1, k2, k3, k4
            D = cv::Mat::zeros(1, 4, CV_64FC1);
        }
    }

    void setFrom(const sl::CameraParameters & cam) {
        K = cv::Mat::eye(3, 3, CV_64FC1);
        K.at<double>(0, 0) = cam.fx;
        K.at<double>(1, 1) = cam.fy;
        K.at<double>(0, 2) = cam.cx;
        K.at<double>(1, 2) = cam.cy;
        
        // tangential distortion coefficients are not used in the Fisheye model, looking for p1 and p2 equal to 0
        if(cam.disto[2]==0. && cam.disto[3]==0. && cam.disto[4]!=0. && cam.disto[5]!=0.) {
            disto_model_RadTan = false; // -> Fisheye model
            // Fisheye model has 4 coefficients: k1, k2, k3, k4
            D = cv::Mat::zeros(1, 4, CV_64FC1);
            D.at<double>(0) = cam.disto[0];
            D.at<double>(1) = cam.disto[1];
            D.at<double>(2) = cam.disto[4];
            D.at<double>(3) = cam.disto[5];
        } else {
            disto_model_RadTan = true; // Radial and tangential distortion
            const int nb_coeff = 8; // 6 radial + 2 tangential; could be extended to 12 with prism distortion
            D = cv::Mat::zeros(1, nb_coeff, CV_64FC1);
            for(int i = 0; i < nb_coeff; i++)
                D.at<double>(i) = cam.disto[i];
        }
    }

    std::vector<cv::Point2f> undistortPoints(const std::vector<cv::Point2f> &points) const {
        std::vector<cv::Point2f> undistorted_points;
        if (disto_model_RadTan) {
            cv::undistortPoints(points, undistorted_points, K, D);
        } else {
            cv::fisheye::undistortPoints(points, undistorted_points, K, D);
        }
        return undistorted_points;
    }

    double calibrate(const std::vector<std::vector<cv::Point3f>> &object_points, 
                        const std::vector<std::vector<cv::Point2f>> &image_points, 
                        const cv::Size &image_size, 
                        int flags) {
        double rms = 0.0;
        std::vector<cv::Mat> rvec, tvec;
        if (disto_model_RadTan){
            if(D.cols >= 8)
                flags += cv::CALIB_RATIONAL_MODEL;
            rms = cv::calibrateCamera(object_points, image_points, image_size, K, D, rvec, tvec, flags);
        } else
            rms = cv::fisheye::calibrate(object_points, image_points, image_size, K, D, rvec, tvec, flags + cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC + cv::fisheye::CALIB_FIX_SKEW);
        return rms;
    }
};

struct StereoCalib{
    CameraCalib left;
    CameraCalib right;
    cv::Mat R; // Rotation matrix between left and right camera
    cv::Mat Rv; // Rotation vector between left and right camera
    cv::Mat T; // Translation vector between left and right camera
    
    void initDefault(bool radtan){
        left.initDefault(radtan);
        right.initDefault(radtan);
        R = cv::Mat::eye(3, 3, CV_64FC1);
        Rv = cv::Mat::zeros(3, 1, CV_64FC1);
        T = cv::Mat::zeros(3, 1, CV_64FC1);
    }

    void setFrom(const sl::CalibrationParameters & calib_params) {
        left.setFrom(calib_params.left_cam);
        right.setFrom(calib_params.right_cam);
        
        auto translation = calib_params.stereo_transform.getTranslation();
        T.at<double>(0) = translation.x * -1; // the zed configuration file store the absolute value of the Tx part
        T.at<double>(1) = translation.y;
        T.at<double>(2) = translation.z;
        
        auto rot = calib_params.stereo_transform.getRotationVector();
        Rv.at<double>(0) = rot.x;
        Rv.at<double>(1) = rot.y;
        Rv.at<double>(2) = rot.z;
        cv::Rodrigues(Rv, R);
        std::cout<<" Lens disto model "<<(left.disto_model_RadTan && right.disto_model_RadTan)<<std::endl;
    }

    double calibrate(const std::vector<std::vector<cv::Point3f>> &object_points, 
                        const std::vector<std::vector<cv::Point2f>> &image_points_left, 
                        const std::vector<std::vector<cv::Point2f>> &image_points_right, 
                        const cv::Size &image_size, 
                        int flags) {
        double rms = 0.0;
        cv::Mat E, F;        
        if(left.disto_model_RadTan && right.disto_model_RadTan)
            rms = cv::stereoCalibrate(object_points, image_points_left, image_points_right, 
                                  left.K, left.D, right.K, right.D, image_size,
                                  R, T, E, F, flags);
        else
          rms = cv::fisheye::stereoCalibrate(
              object_points, image_points_left, image_points_right, left.K,
              left.D, right.K, right.D, image_size, R, T,
              flags + cv::fisheye::CALIB_CHECK_COND);
        cv::Rodrigues(R, Rv);
        return rms;
    }
};

int calibrate(const std::string &folder, StereoCalib &raw_data, int target_w,
              int target_h, float square_size, int serial,
              bool save_calib_mono = false, bool use_intrinsic_prior = false,
              float max_repr_error = 0.5f);