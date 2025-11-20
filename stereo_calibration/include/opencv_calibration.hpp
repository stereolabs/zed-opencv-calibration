#pragma once

#include <cmath>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>
#include <sl/CameraOne.hpp>

constexpr int MIN_IMAGE = 20;

struct CameraCalib {
  cv::Mat K;
  cv::Mat D;
  bool disto_model_RadTan = true;

  void print(const std::string &name) const {
    std::cout << name << " K:" << std::endl << K << std::endl;
    std::cout << " D:" << std::endl << D << std::endl;
  }

  void initDefault(bool radtan) {
    disto_model_RadTan = radtan;
    K = cv::Mat::eye(3, 3, CV_32FC1);
    if (disto_model_RadTan) {
      // Radial and tangential distortion
      const int nb_coeff = 8;  // 6 radial + 2 tangential; could be extended to
                               // 12 with prism distortion
      D = cv::Mat::zeros(1, nb_coeff, CV_32FC1);
    } else {
      // Fisheye model has 4 coefficients: k1, k2, k3, k4
      D = cv::Mat::zeros(1, 4, CV_32FC1);
    }
  }

  void setFrom(const sl::CameraParameters &cam) {
    K = cv::Mat::eye(3, 3, CV_32FC1);
    K.at<float>(0, 0) = cam.fx;
    K.at<float>(1, 1) = cam.fy;
    K.at<float>(0, 2) = cam.cx;
    K.at<float>(1, 2) = cam.cy;

    // tangential distortion coefficients are not used in the Fisheye model,
    // looking for p1 and p2 equal to 0
    if (cam.disto[2] == 0. && cam.disto[3] == 0. && cam.disto[4] != 0. &&
        cam.disto[5] != 0.) {
      disto_model_RadTan = false;  // -> Fisheye model
      // Fisheye model has 4 coefficients: k1, k2, k3, k4
      D = cv::Mat::zeros(1, 4, CV_32FC1);
      D.at<float>(0) = cam.disto[0];
      D.at<float>(1) = cam.disto[1];
      D.at<float>(2) = cam.disto[4];
      D.at<float>(3) = cam.disto[5];
    } else {
      disto_model_RadTan = true;  // Radial and tangential distortion
      const int nb_coeff = 8;     // 6 radial + 2 tangential; could be extended
                                  // to 12 with prism distortion
      D = cv::Mat::zeros(1, nb_coeff, CV_32FC1);
      for (int i = 0; i < nb_coeff; i++) D.at<float>(i) = cam.disto[i];
    }
  }

  std::vector<cv::Point2f> undistortPoints(
      const std::vector<cv::Point2f> &points) const {
    std::cout << "K:" << std::endl << K << std::endl;
    std::cout << "D:" << std::endl << D << std::endl;
    std::vector<cv::Point2f> undistorted_points;
    if (disto_model_RadTan) {
      cv::undistortPoints(points, undistorted_points, K, D);
    } else {
      cv::fisheye::undistortPoints(points, undistorted_points, K, D);
    }
    return undistorted_points;
  }

  float calibrate(const std::vector<std::vector<cv::Point3f>> &object_points,
                  const std::vector<std::vector<cv::Point2f>> &image_points,
                  const cv::Size &image_size, int flags, bool verbose) {
    float rms = -1.0f;
    std::vector<cv::Mat> rvec, tvec;
    if (disto_model_RadTan) {
      if (D.cols >= 8) flags += cv::CALIB_RATIONAL_MODEL;
      rms = cv::calibrateCamera(object_points, image_points, image_size, K, D,
                                rvec, tvec, flags);
    } else {
      rms = cv::fisheye::calibrate(
          object_points, image_points, image_size, K, D, rvec, tvec,
          flags + cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC +
              cv::fisheye::CALIB_FIX_SKEW);
    }

    if (verbose) {
      std::cout << " * Intrinsic matrix K:" << std::endl << K << std::endl;
      std::cout << " * Distortion coefficients D:" << std::endl
                << D << std::endl;
      std::cout << " * Re-projection error (RMS): " << rms << std::endl;
    }

    return rms;
  }
};

struct StereoCalib {
  CameraCalib left;
  CameraCalib right;

  cv::Mat R;   // Rotation matrix between left and right camera
  cv::Mat Rv;  // Rotation vector between left and right camera
  cv::Mat T;   // Translation vector between left and right camera
  
  cv::Size imageSize;

  void initDefault(bool radtan) {
    left.initDefault(radtan);
    right.initDefault(radtan);
    R = cv::Mat::eye(3, 3, CV_32FC1);
    Rv = cv::Mat::zeros(3, 1, CV_32FC1);
    T = cv::Mat::zeros(3, 1, CV_32FC1);
  }

  void setFrom(const sl::CalibrationParameters &calib_params) {
    left.setFrom(calib_params.left_cam);
    right.setFrom(calib_params.right_cam);

    auto translation = calib_params.stereo_transform.getTranslation();
    T.at<float>(0) = translation.x * -1;  // the zed configuration file store
                                          // the absolute value of the Tx part
    T.at<float>(1) = translation.y;
    T.at<float>(2) = translation.z;

    auto rot = calib_params.stereo_transform.getRotationVector();
    Rv.at<float>(0) = rot.x;
    Rv.at<float>(1) = rot.y;
    Rv.at<float>(2) = rot.z;
    cv::Rodrigues(Rv, R);
    std::cout << " Lens disto model "
              << (left.disto_model_RadTan && right.disto_model_RadTan)
              << std::endl;
  }

  float calibrate(
      const std::vector<std::vector<cv::Point3f>> &object_points,
      const std::vector<std::vector<cv::Point2f>> &image_points_left,
      const std::vector<std::vector<cv::Point2f>> &image_points_right,
      const cv::Size &image_size, int flags, bool verbose) {
    
    imageSize = image_size;
    
    float rms = 0.0;
    cv::Mat E, F;
    
    if (left.disto_model_RadTan && right.disto_model_RadTan) {
      rms = cv::stereoCalibrate(object_points, image_points_left,
                                image_points_right, left.K, left.D, right.K,
                                right.D, image_size, R, T, E, F, flags);
    } else {
      rms = cv::fisheye::stereoCalibrate(object_points, image_points_left,
                                         image_points_right, left.K, left.D,
                                         right.K, right.D, image_size, R, T,
                                         flags + cv::fisheye::CALIB_CHECK_COND);
    }

    cv::Rodrigues(R, Rv);

    if (verbose) {
      std::cout << " * Rotation matrix R:" << std::endl << R << std::endl;
      std::cout << " * Rotation vector Rv:" << std::endl << Rv << std::endl;
      std::cout << " * Translation vector T:" << std::endl << T << std::endl;
      std::cout << " * Re-projection error (RMS): " << rms << std::endl;
    }

    return rms;
  }

  std::string saveCalibOpenCV(int serial);
  std::string saveCalibZED(int serial, bool is_4k = false);
};

int calibrate(int img_count, const std::string &folder, StereoCalib &raw_data,
              int target_w, int target_h, float square_size, int serial, bool is_4k,
              bool save_calib_mono = false, bool use_intrinsic_prior = false,
              float max_repr_error = 0.5f, bool verbose = false);