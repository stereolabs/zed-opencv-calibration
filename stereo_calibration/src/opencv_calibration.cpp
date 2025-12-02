#include "opencv_calibration.hpp"

int calibrate(int img_count, const std::string& folder, StereoCalib& calib_data,
              int target_w, int target_h, float square_size, int serial,
              bool is_dual_mono, bool is_4k, bool save_calib_mono,
              bool use_intrinsic_prior, float max_repr_error, bool verbose) {
  std::vector<cv::Mat> left_images, right_images;

  /// Read images
  cv::Size imageSize = cv::Size(0, 0);

  for (int i = 0; i < img_count; i++) {
    cv::Mat grey_l =
        cv::imread(folder + "image_left_" + std::to_string(i) + ".png",
                   cv::IMREAD_GRAYSCALE);
    cv::Mat grey_r =
        cv::imread(folder + "image_right_" + std::to_string(i) + ".png",
                   cv::IMREAD_GRAYSCALE);

    if (!grey_l.empty() && !grey_r.empty()) {
      if (imageSize.width == 0)
        imageSize = grey_l.size();
      else if (imageSize != left_images.back().size()) {
        std::cout << "Image number " << i
                  << " does not have the same size as the previous ones"
                  << imageSize << " vs " << left_images.back().size()
                  << std::endl;
        break;
      }

      left_images.push_back(grey_l);
      right_images.push_back(grey_r);
    }
  }

  if (verbose) {
    std::cout << std::endl
              << "\t" << left_images.size() << " samples collected"
              << std::endl;
  }

  // Define object points of the target
  std::vector<cv::Point3f> pattern_points;
  for (int i = 0; i < target_h; i++) {
    for (int j = 0; j < target_w; j++) {
      pattern_points.push_back(
          cv::Point3f(square_size * j, square_size * i, 0));
    }
  }

  std::vector<std::vector<cv::Point3f>> object_points;
  std::vector<std::vector<cv::Point2f>> pts_l, pts_r;

  cv::Size t_size(target_w, target_h);

  for (int i = 0; i < left_images.size(); i++) {
    std::vector<cv::Point2f> pts_l_, pts_r_;
    bool found_l =
        cv::findChessboardCorners(left_images.at(i), t_size, pts_l_, 3);
    bool found_r =
        cv::findChessboardCorners(right_images.at(i), t_size, pts_r_, 3);

    if (found_l && found_r) {
      cv::cornerSubPix(
          left_images.at(i), pts_l_, cv::Size(5, 5), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER,
                           30, 0.001));

      cv::cornerSubPix(
          right_images.at(i), pts_r_, cv::Size(5, 5), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER,
                           30, 0.001));

      pts_l.push_back(pts_l_);
      pts_r.push_back(pts_r_);
      object_points.push_back(pattern_points);
    } else {
      std::cout << "No target detected on image " << i << std::endl;
    }
  }

  /// Compute calibration
  std::cout << std::endl << "*** Calibration Report ***" << std::endl;

  if (pts_l.size() < MIN_IMAGE) {
    std::cout << " !!! Not enough images with the target detected !!!"
              << std::endl;
    std::cout << " Please perform a new data acquisition." << std::endl
              << std::endl;
    return EXIT_FAILURE;
  } else {
    std::cout << " * Enough valid samples: " << pts_l.size() << std::endl;

    auto flags = use_intrinsic_prior ? cv::CALIB_USE_INTRINSIC_GUESS : 0;
    if (verbose) {
      std::cout << "Left camera calibration: " << std::endl;
    }
    auto rms_l = calib_data.left.calibrate(object_points, pts_l, imageSize,
                                           flags, verbose);

    if (verbose) {
      std::cout << "Right camera calibration: " << std::endl;
    }
    auto rms_r = calib_data.right.calibrate(object_points, pts_r, imageSize,
                                            flags, verbose);

    if (verbose) {
      std::cout << "Stereo calibration: " << std::endl;
    }
    auto err = calib_data.calibrate(
        object_points, pts_l, pts_r, imageSize,
        cv::CALIB_USE_INTRINSIC_GUESS + cv::CALIB_ZERO_DISPARITY, verbose);

    std::cout << " * Reprojection errors: " << std::endl;
    std::cout << "   * Left " << rms_l
              << (rms_l > max_repr_error ? " !!! TOO HIGH !!!" : "")
              << std::endl;
    std::cout << "   * Right " << rms_r
              << (rms_r > max_repr_error ? " !!! TOO HIGH !!!" : "")
              << std::endl;
    std::cout << "   * Stereo " << err
              << (err > max_repr_error ? " !!! TOO HIGH !!!" : "") << std::endl;

    if (rms_l > 0.5f || rms_r > 0.5f || err > 0.5f) {
      std::cout << std::endl
                << "\t !! Warning !!" << std::endl
                << "The max reprojection error looks too high (>"
                << max_repr_error
                << "), check that the lenses are clean (sharp images)"
                   " and that the pattern is printed/mounted on a RIGID "
                   "and FLAT surface."
                << std::endl;

      return EXIT_FAILURE;
    }

    std::cout << std::endl;

    std::cout << "** Camera parameters **" << std::endl;
    std::cout << "* Intrinsic mat left:" << std::endl
              << calib_data.left.K << std::endl;
    std::cout << "* Distortion mat left:" << std::endl
              << calib_data.left.D << std::endl;
    std::cout << "* Intrinsic mat right:" << std::endl
              << calib_data.right.K << std::endl;
    std::cout << "* Distortion mat right:" << std::endl
              << calib_data.right.D << std::endl;
    std::cout << std::endl;
    std::cout << "** Extrinsic parameters **" << std::endl;
    std::cout << "* Translation:" << std::endl << calib_data.T << std::endl;
    std::cout << "* Rotation:" << std::endl << calib_data.Rv << std::endl;
    std::cout << std::endl;

    std::cout << std::endl << "*** Save Calibration files ***" << std::endl;

    std::string opencv_file = calib_data.saveCalibOpenCV(serial);
    if (!opencv_file.empty()) {
      std::cout << " * OpenCV calibration file saved: " << opencv_file
                << std::endl;
    } else {
      std::cout << " !!! Failed to save OpenCV calibration file " << opencv_file
                << " !!!" << std::endl;
    }

    // SDK format is only supported for dual-mono setups
    if (is_dual_mono) {
      std::string zed_file = calib_data.saveCalibZED(serial, is_4k);
      if (!zed_file.empty()) {
        std::cout << " * ZED SDK calibration file saved: " << zed_file
                  << std::endl;
      } else {
        std::cout << " !!! Failed to save ZED SDK calibration file " << zed_file
                  << " !!!" << std::endl;
      }
    }
  }

  return EXIT_SUCCESS;
}

std::string StereoCalib::saveCalibOpenCV(int serial) {
  std::string calib_filename =
      "zed_calibration_" + std::to_string(serial) + ".yml";

  cv::FileStorage fs(calib_filename, cv::FileStorage::WRITE);
  if (fs.isOpened()) {
    fs << "Size" << imageSize;
    fs << "K_LEFT" << left.K << "K_RIGHT" << right.K;

    if (left.disto_model_RadTan) {
      fs << "D_LEFT" << left.D << "D_RIGHT" << right.D;
    } else {
      fs << "D_LEFT_FE" << left.D << "D_RIGHT_FE" << right.D;
    }

    fs << "R" << Rv << "T" << T;
    fs.release();

    return calib_filename;
  }

  return std::string();
}

void printDisto(const CameraCalib& calib, std::ofstream &outfile) {
  if(calib.disto_model_RadTan) {
    size_t dist_size = calib.D.total();
    outfile << "k1 = " << calib.D.at<float>(0) << "\n";
    outfile << "k2 = " << calib.D.at<float>(1) << "\n";
    outfile << "p1 = " << calib.D.at<float>(2) << "\n";
    outfile << "p2 = " << calib.D.at<float>(3) << "\n";
    outfile << "k3 = " << calib.D.at<float>(4) << "\n";
    outfile << "k4 = " << (dist_size > 5 ? calib.D.at<float>(5) : 0.0) << "\n";
    outfile << "k5 = " << (dist_size > 6 ? calib.D.at<float>(6) : 0.0) << "\n";
    outfile << "k6 = " << (dist_size > 7 ? calib.D.at<float>(7) : 0.0) << "\n";
  }else{
    outfile << "k1 = " << calib.D.at<float>(0) << "\n";
    outfile << "k2 = " << calib.D.at<float>(1) << "\n";
    outfile << "k3 = " << calib.D.at<float>(2) << "\n";
    outfile << "k4 = " << calib.D.at<float>(3) << "\n";
  }
  outfile<<"\n";
}

std::string StereoCalib::saveCalibZED(int serial, bool is_4k) {
  std::string calib_filename = "SN" + std::to_string(serial) + ".conf";

  // Write parameters to a text file
  std::ofstream outfile(calib_filename);
  if (!outfile.is_open()) {
    std::cerr
        << " !!! Cannot save the calibration file: 'Unable to open output file'"
        << std::endl;
    return std::string();
  }

  if (left.K.type() == CV_64F) {
    std::cout << " Data type: double" << std::endl;
  } else if (left.K.type() == CV_32F) {
    std::cout << " Data type: float" << std::endl;
  } else {
    std::cerr << " !!! Cannot save the calibration file: 'Invalid data type'"
              << std::endl;
    return std::string();
  }

  if (!is_4k) {  //  AR0234

    if (imageSize.height != 1200) {
      std::cout << "The resolution for the calibration is not valid\n\nUse "
                   "HD1200 (1920x1200) for ZED X One GS"
                << std::endl;
      return std::string();
    }

    outfile << "[LEFT_CAM_FHD1200]\n";
    outfile << "fx = " << left.K.at<double>(0, 0) << "\n";
    outfile << "fy = " << left.K.at<double>(1, 1) << "\n";
    outfile << "cx = " << left.K.at<double>(0, 2) << "\n";
    outfile << "cy = " << left.K.at<double>(1, 2) << "\n\n";

    outfile << "[RIGHT_CAM_FHD1200]\n";
    outfile << "fx = " << right.K.at<double>(0, 0) << "\n";
    outfile << "fy = " << right.K.at<double>(1, 1) << "\n";
    outfile << "cx = " << right.K.at<double>(0, 2) << "\n";
    outfile << "cy = " << right.K.at<double>(1, 2) << "\n\n";

    outfile << "[LEFT_CAM_FHD]\n";
    outfile << "fx = " << left.K.at<double>(0, 0) << "\n";
    outfile << "fy = " << left.K.at<double>(1, 1) << "\n";
    outfile << "cx = " << left.K.at<double>(0, 2) << "\n";
    outfile << "cy = " << left.K.at<double>(1, 2) - 60 << "\n\n";

    outfile << "[RIGHT_CAM_FHD]\n";
    outfile << "fx = " << right.K.at<double>(0, 0) << "\n";
    outfile << "fy = " << right.K.at<double>(1, 1) << "\n";
    outfile << "cx = " << right.K.at<double>(0, 2) << "\n";
    outfile << "cy = " << right.K.at<double>(1, 2) - 60 << "\n\n";

    outfile << "[LEFT_CAM_SVGA]\n";
    outfile << "fx = " << left.K.at<double>(0, 0) / 2 << "\n";
    outfile << "fy = " << left.K.at<double>(1, 1) / 2 << "\n";
    outfile << "cx = " << left.K.at<double>(0, 2) / 2 << "\n";
    outfile << "cy = " << left.K.at<double>(1, 2) / 2 << "\n\n";

    outfile << "[RIGHT_CAM_SVGA]\n";
    outfile << "fx = " << right.K.at<float>(0, 0) / 2 << "\n";
    outfile << "fy = " << right.K.at<float>(1, 1) / 2 << "\n";
    outfile << "cx = " << right.K.at<float>(0, 2) / 2 << "\n";
    outfile << "cy = " << right.K.at<float>(1, 2) / 2 << "\n\n";

    outfile << "[LEFT_DISTO]\n";
    printDisto(left, outfile);

    outfile << "[RIGHT_DISTO]\n";
    printDisto(right, outfile);

    outfile << "[STEREO]\n";
    outfile << "Baseline = " << -T.at<float>(0) << "\n";
    outfile << "TY = " << T.at<float>(1) << "\n";
    outfile << "TZ = " << T.at<float>(2) << "\n";
    outfile << "CV_FHD = " << Rv.at<float>(1) << "\n";
    outfile << "CV_SVGA = " << Rv.at<float>(1) << "\n";
    outfile << "CV_FHD1200 = " << Rv.at<float>(1) << "\n";
    outfile << "RX_FHD = " << Rv.at<float>(0) << "\n";
    outfile << "RX_SVGA = " << Rv.at<float>(0) << "\n";
    outfile << "RX_FHD1200 = " << Rv.at<float>(0) << "\n";
    outfile << "RZ_FHD = " << Rv.at<float>(2) << "\n";
    outfile << "RZ_SVGA = " << Rv.at<float>(2) << "\n";
    outfile << "RZ_FHD1200 = " << Rv.at<float>(2) << "\n\n";

    outfile.close();
    std::cout << " * Parameter file written successfully: '" << calib_filename
              << "'" << std::endl;
    return calib_filename;
  } else {  //  IMX678

    if (imageSize.height != 2160) {
      std::cout << "The resolution for the calibration is not valid\n\nUse "
                   "4K (3840x2160) for ZED X One 4K"
                << std::endl;
      return std::string();
    }

    outfile << "[LEFT_CAM_4k]\n";
    outfile << "fx = " << left.K.at<double>(0, 0) << "\n";
    outfile << "fy = " << left.K.at<double>(1, 1) << "\n";
    outfile << "cx = " << left.K.at<double>(0, 2) << "\n";
    outfile << "cy = " << left.K.at<double>(1, 2) << "\n\n";

    outfile << "[RIGHT_CAM_4k]\n";
    outfile << "fx = " << right.K.at<double>(0, 0) << "\n";
    outfile << "fy = " << right.K.at<double>(1, 1) << "\n";
    outfile << "cx = " << right.K.at<double>(0, 2) << "\n";
    outfile << "cy = " << right.K.at<double>(1, 2) << "\n\n";

    outfile << "[LEFT_CAM_QHDPLUS]\n";
    outfile << "fx = " << left.K.at<double>(0, 0) << "\n";
    outfile << "fy = " << left.K.at<double>(1, 1) << "\n";
    outfile << "cx = " << left.K.at<double>(0, 2) - (3840 - 3200) / 2 << "\n";
    outfile << "cy = " << left.K.at<double>(1, 2) - (2160 - 1800) / 2 << "\n\n";

    outfile << "[RIGHT_CAM_QHDPLUS]\n";
    outfile << "fx = " << right.K.at<double>(0, 0) << "\n";
    outfile << "fy = " << right.K.at<double>(1, 1) << "\n";
    outfile << "cx = " << right.K.at<double>(0, 2) - (3840 - 3200) / 2 << "\n";
    outfile << "cy = " << right.K.at<double>(1, 2) - (2160 - 1800) / 2
            << "\n\n";

    outfile << "[LEFT_CAM_FHD]\n";
    outfile << "fx = " << left.K.at<double>(0, 0) / 2 << "\n";
    outfile << "fy = " << left.K.at<double>(1, 1) / 2 << "\n";
    outfile << "cx = " << left.K.at<double>(0, 2) / 2 << "\n";
    outfile << "cy = " << left.K.at<double>(1, 2) / 2 << "\n\n";

    outfile << "[RIGHT_CAM_FHD]\n";
    outfile << "fx = " << right.K.at<double>(0, 0) / 2 << "\n";
    outfile << "fy = " << right.K.at<double>(1, 1) / 2 << "\n";
    outfile << "cx = " << right.K.at<double>(0, 2) / 2 << "\n";
    outfile << "cy = " << right.K.at<double>(1, 2) / 2 << "\n\n";

    outfile << "[LEFT_CAM_FHD1200]\n";
    outfile << "fx = " << left.K.at<double>(0, 0) << "\n";
    outfile << "fy = " << left.K.at<double>(1, 1) << "\n";
    outfile << "cx = " << left.K.at<double>(0, 2) - (3840 - 1920) / 2 << "\n";
    outfile << "cy = " << left.K.at<double>(1, 2) - (2160 - 1200) / 2 << "\n\n";

    outfile << "[RIGHT_CAM_FHD1200]\n";
    outfile << "fx = " << right.K.at<double>(0, 0) << "\n";
    outfile << "fy = " << right.K.at<double>(1, 1) << "\n";
    outfile << "cx = " << right.K.at<double>(0, 2) - (3840 - 1920) / 2 << "\n";
    outfile << "cy = " << right.K.at<double>(1, 2) - (2160 - 1200) / 2
            << "\n\n";

    outfile << "[LEFT_DISTO]\n";
    printDisto(left, outfile);

    outfile << "[RIGHT_DISTO]\n";
    printDisto(right, outfile);

    outfile << "[STEREO]\n";
    outfile << "Baseline = " << -T.at<float>(0) << "\n";
    outfile << "TY = " << T.at<float>(1) << "\n";
    outfile << "TZ = " << T.at<float>(2) << "\n";
    outfile << "CV_FHD = " << Rv.at<float>(1) << "\n";
    outfile << "CV_FHD1200 = " << Rv.at<float>(1) << "\n";
    outfile << "CV_4k = " << Rv.at<float>(1) << "\n";
    outfile << "CV_QHDPLUS = " << Rv.at<float>(1) << "\n";
    outfile << "RX_FHD = " << Rv.at<float>(0) << "\n";
    outfile << "RX_FHD1200 = " << Rv.at<float>(0) << "\n";
    outfile << "RX_4k = " << Rv.at<float>(0) << "\n";
    outfile << "RX_QHDPLUS = " << Rv.at<float>(0) << "\n";
    outfile << "RZ_FHD = " << Rv.at<float>(2) << "\n";
    outfile << "RZ_FHD1200 = " << Rv.at<float>(2) << "\n";
    outfile << "RZ_4k = " << Rv.at<float>(2) << "\n\n";
    outfile << "RZ_QHDPLUS = " << Rv.at<float>(2) << "\n\n";

    outfile.close();
    std::cout << " * Parameter file written successfully: '" << calib_filename
              << "'" << std::endl;
    return calib_filename;
  }
}