#include "calibration_checker.hpp"

constexpr float PI = static_cast<float>(M_PI);

constexpr size_t up_left = 0;
constexpr size_t up_right = 1;
constexpr size_t down_right = 2;
constexpr size_t down_left = 3;

CalibrationChecker::CalibrationChecker(cv::Size board_size, float square_size,
                                       size_t min_samples, size_t max_samples,
                                       DetectedBoardParams idealParams,
                                       bool verbose) {
  verbose_ = verbose;

  // Calibration parameters
  min_samples_ = min_samples;
  max_samples_ = max_samples;
  idealParams_ = idealParams;

  // Initialize the board parameters
  board_.board_size = board_size;
  board_.square_size = square_size;
  board_.objp.clear();
  board_.board_size_mm =
      cv::Size(board_size.width * square_size, board_size.height * square_size);

  // Prepare object points based on the known board size and square size
  for (int i = 0; i < board_size.height; i++) {
    for (int j = 0; j < board_size.width; j++) {
      board_.objp.push_back(cv::Point3f(j * square_size, i * square_size, 0));
    }
  }
}

bool CalibrationChecker::testSample(const std::vector<cv::Point2f>& corners,
                                    cv::Size image_size) {
  DetectedBoardParams params = getDetectedBoarParams(corners, image_size);

  if (params.size < 0 || params.skew < 0) {
    return false;  // Invalid parameters
  }

  std::cout << std::setprecision(3) << " * New Sample: Pos(" << params.pos.x
            << ", " << params.pos.y << "), Size: " << params.size
            << ", Skew: " << params.skew << std::endl;

  if (isGoodSample(params)) {
    // Store the valid parameters and associated corners
    paramDb_.push_back(params);
    validCorners_.push_back(corners);
    std::cout << "  Sample stored. Total valid samples: "
              << validCorners_.size() << std::endl;
    return true;
  }

  std::cout << " * Sample rejected." << std::endl;
  return false;
}

float CalibrationChecker::compute_skew(
    const std::vector<cv::Point2f>& outside_corners) {
  /*  Get skew for given checkerboard detection.
    Scaled to [0,1], which 0 = no skew, 1 = high skew
    Skew is proportional to the divergence of three outside corners from 90
    degrees.
   */
  if (outside_corners.size() != 4) {
    return -1.0f;  // Invalid input
  }

  auto angle = [](const cv::Point2f& a, const cv::Point2f& b,
                  const cv::Point2f& c) -> float {
    cv::Point2f ab = a - b;
    cv::Point2f cb = c - b;
    float dot = ab.x * cb.x + ab.y * cb.y;
    float norm_ab = std::sqrt(ab.x * ab.x + ab.y * ab.y);
    float norm_cb = std::sqrt(cb.x * cb.x + cb.y * cb.y);
    float cos_angle = dot / (norm_ab * norm_cb);
    if (cos_angle < -1.0f)
      cos_angle = -1.0f;
    else if (cos_angle > 1.0f)
      cos_angle = 1.0f;
    return std::acos(cos_angle);
  };

  // Original code from here:
  // https://github.com/ros-perception/image_pipeline/blob/rolling/camera_calibration/src/camera_calibration/calibrator.py#L187-L207
  // float skew = std::min(
  //     1.0f, 2.0f * std::abs((PI / 2) - angle(outside_corners[up_left],
  //                                            outside_corners[up_right],
  //                                            outside_corners[down_right])));

  float maxDeviation = 0.0f;
  for (int i = 0; i < 4; i++) {
    float ang = angle(outside_corners[(i + 3) % 4], outside_corners[i],
                      outside_corners[(i + 1) % 4]);
    float deviation = std::abs((PI / 2) - ang);
    if (deviation > maxDeviation) {
      maxDeviation = deviation;
    }
  }

  return maxDeviation / (PI / 2);
}

float CalibrationChecker::compute_area(
    const std::vector<cv::Point2f>& outside_corners) {
  /* Get 2d image area of the detected checkerboard.
    The projected checkerboard is assumed to be a convex quadrilateral, and the
    area computed as |p X q|/2; see
    http://mathworld.wolfram.com/Quadrilateral.html.
   */

  if (outside_corners.size() != 4) {
    return -1.0f;  // Invalid input
  }

  // Using the shoelace formula to compute area of the quadrilateral
  cv::Point2f a = outside_corners[up_right] - outside_corners[up_left];
  cv::Point2f b = outside_corners[down_right] - outside_corners[up_right];
  cv::Point2f c = outside_corners[down_left] - outside_corners[down_right];

  cv::Point2f p = b + c;
  cv::Point2f q = a + b;

  float area = 0.5f * std::abs(p.x * q.y - p.y * q.x);

  return area;
}

std::vector<cv::Point2f> CalibrationChecker::get_outside_corners(
    const std::vector<cv::Point2f>& corners) {
  std::vector<cv::Point2f> outside_corners;

  if (corners.size() != board_.board_size.area()) {
    return outside_corners;
  }

  size_t x_dim = board_.board_size.width;
  size_t y_dim = board_.board_size.height;

  outside_corners.resize(4);

  outside_corners[up_left] = corners[0];           // Top-left
  outside_corners[up_right] = corners[x_dim - 1];  // Top-right
  outside_corners[down_right] =
      corners[(y_dim - 1) * x_dim + (x_dim - 1)];             // Bottom-right
  outside_corners[down_left] = corners[(y_dim - 1) * x_dim];  // Bottom-left

  return outside_corners;
}

DetectedBoardParams CalibrationChecker::getDetectedBoarParams(
    const std::vector<cv::Point2f>& corners, cv::Size image_size) {
  DetectedBoardParams params;

  auto outside_corners = get_outside_corners(corners);
  float area = compute_area(outside_corners);
  float skew = compute_skew(outside_corners);

  if (area < 0 || skew < 0) {
    // Return invalid params
    params.size = -1.0f;
    params.skew = -1.0f;
    return params;
  }

  float border = std::sqrt(area);

  // For X and Y, we "shrink" the image all around by approx.half the board
  // size. Otherwise large boards are penalized because you can't get much X/Y
  // variation.
  float avg_x = 0.0f;
  float avg_y = 0.0f;
  for (const auto& corner : corners) {
    avg_x += corner.x;
    avg_y += corner.y;
  }
  avg_x /= static_cast<float>(corners.size());
  avg_y /= static_cast<float>(corners.size());

  float p_x = std::min(1.0f, std::max(0.0f, (avg_x - border / 2.0f) /
                                                (image_size.width - border)));
  float p_y = std::min(1.0f, std::max(0.0f, (avg_y - border / 2.0f) /
                                                (image_size.height - border)));

  params.pos = cv::Point2f(p_x, p_y);
  params.size = std::sqrt(area / (image_size.width * image_size.height));
  params.skew = skew;

  return params;
}

bool CalibrationChecker::isGoodSample(const DetectedBoardParams& params) {
  if (paramDb_.empty()) {
    return true;  // First sample is always good
  }

  // Original similarity check from:
  // https://github.com/ros-perception/image_pipeline/blob/rolling/camera_calibration/src/camera_calibration/calibrator.py#L485-L507
  // auto param_distance = [](const DetectedBoardParams& p1,
  //                          const DetectedBoardParams& p2) -> float {
  //   return std::abs(p1.size - p2.size) + std::abs(p1.skew - p2.skew) +
  //          std::abs(p1.pos.x - p2.pos.x) + std::abs(p1.pos.y - p2.pos.y);
  // };

  // for (auto& stored_params : paramDb_) {
  //   float dist = param_distance(params, stored_params);
  //   if (dist < 0.2f) {  // TODO tune the threshold
  //     std::cout << "  Rejected: Too similar to existing samples (dist=" <<
  //     dist
  //               << ")" << std::endl;
  //     return false;
  //   }
  // }

  // return true;

  // New similarity check:
  auto is_different = [this](const DetectedBoardParams& p1,
                             const DetectedBoardParams& p2) -> bool {
    // Check that at least one parameter differs by at least 10% from all the
    // stored samples
    constexpr float epsilon = 1e-6f;
    float pos_x_diff =
        std::abs(p1.pos.x - p2.pos.x) / std::max(std::max(p1.pos.x, p2.pos.x), epsilon);
    float pos_y_diff =
        std::abs(p1.pos.y - p2.pos.y) / std::max(std::max(p1.pos.y, p2.pos.y), epsilon);
    float size_diff = std::abs(p1.size - p2.size) / std::max(std::max(p1.size, p2.size), epsilon);
    float skew_diff = std::abs(p1.skew - p2.skew) / std::max(std::max(p1.skew, p2.skew), epsilon);

    const float diff_thresh = 0.1f;  // 10% difference threshold

    if (verbose_) {
      std::cout << "Comparing to: Pos(" << p2.pos.x << ", " << p2.pos.y
                << "), Size: " << p2.size << ", Skew: " << p2.skew << std::endl;
      std::cout << std::setprecision(3) << "PosX diff: " << pos_x_diff * 100.0f
                << "%, "
                << "PosY diff: " << pos_y_diff * 100.0f << "%, "
                << "Size diff: " << size_diff * 100.0f << "%, "
                << "Skew diff: " << skew_diff * 100.0f << "%";
    }

    if (size_diff > diff_thresh || skew_diff > diff_thresh ||
        pos_x_diff > diff_thresh || pos_y_diff > diff_thresh) {
      if (verbose_) {
        std::cout << " => Different enough." << std::endl;
      }
      return true;  // At least one parameter is sufficiently different
    }

    if (verbose_) {
      std::cout << " => Too similar." << std::endl;
    }

    return false;  // All parameters are too similar
  };

  for (auto& stored_params : paramDb_) {
    // Stop at the first similar sample found
    if (!is_different(params, stored_params)) {
      std::cout << "  Rejected: Too similar to an existing sample" << std::endl;
      return false;
    }
  }

  return true;
}

bool CalibrationChecker::evaluateSampleCollectionStatus(
    float& size_score, float& skew_score, float& pos_score_x,
    float& pos_score_y) const {
  size_score = 0.0f;
  skew_score = 0.0f;
  pos_score_x = 0.0f;
  pos_score_y = 0.0f;
  if (paramDb_.empty()) {
    return false;
  }

  float min_px = 1.0f, max_px = 0.0f;
  float min_py = 1.0f, max_py = 0.0f;
  float min_size = 1.0f, max_size = 0.0f;
  float min_skew = 1.0f, max_skew = 0.0f;

  for (const auto& params : paramDb_) {
    if (params.pos.x < min_px) min_px = params.pos.x;
    if (params.pos.x > max_px) max_px = params.pos.x;
    if (params.pos.y < min_py) min_py = params.pos.y;
    if (params.pos.y > max_py) max_py = params.pos.y;
    if (params.size < min_size) min_size = params.size;
    if (params.size > max_size) max_size = params.size;
    if (params.skew < min_skew) min_skew = params.skew;
    if (params.skew > max_skew) max_skew = params.skew;
  }

  // Don't reward small size or skew
  // min_skew = 0.0f;
  // min_size = 0.0f;

  pos_score_x = std::min((max_px - min_px) / idealParams_.pos.x, 1.0f);
  pos_score_y = std::min((max_py - min_py) / idealParams_.pos.y, 1.0f);
  size_score = std::min((max_size - min_size) / idealParams_.size, 1.0f);
  skew_score = std::min((max_skew - min_skew) / idealParams_.skew, 1.0f);

  std::cout << "Sample Collection Status:" << std::endl;
  std::cout << " - PosX status: [" << min_px << " , " << max_px << "] -> "
            << max_px - min_px << "/" << idealParams_.pos.x << std::endl;
  std::cout << "  * PosX Score : " << std::setprecision(3) << pos_score_x
            << std::endl;
  std::cout << " - PosY status: [" << min_py << " , " << max_py << "] -> "
            << max_py - min_py << "/" << idealParams_.pos.y << std::endl;
  std::cout << "  * PosY Score : " << std::setprecision(3) << pos_score_y
            << std::endl;
  std::cout << " - Size status: [" << min_size << " , " << max_size << "] -> "
            << max_size - min_size << "/" << idealParams_.size << std::endl;
  std::cout << "  * Size Score : " << std::setprecision(3) << size_score
            << std::endl;
  std::cout << " - Skew status: [" << min_skew << " , " << max_skew << "] -> "
            << max_skew - min_skew << "/" << idealParams_.skew << std::endl;
  std::cout << "  * Skew Score : " << std::setprecision(3) << skew_score
            << std::endl;

  if (paramDb_.size() < min_samples_) {
    std::cout << "Sample collection incomplete: not reached the minimum sample "
                 "count ("
              << paramDb_.size() << "/" << min_samples_ << ")" << std::endl;
    return false;
  }

  if (paramDb_.size() >= max_samples_) {
    std::cout
        << "Sample collection complete: Reached the maximum sample count ("
        << paramDb_.size() << "/" << max_samples_ << ")" << std::endl;
    return true;
  }

  if (size_score >= 1.0f && skew_score >= 1.0f && pos_score_x >= 1.0f &&
      pos_score_y >= 1.0f) {
    std::cout << "Sample collection complete: All scores are above threshold"
              << std::endl;
    return true;
  }

  std::cout << "Sample collection incomplete." << std::endl;
  return false;
}

const DetectedBoardParams& CalibrationChecker::getLastDetectedBoardParams()
    const {
  if (paramDb_.empty()) {
    static DetectedBoardParams empty_params = {cv::Point2f(-1.0f, -1.0f), -1.0f,
                                               -1.0f};
    return empty_params;
  }
  return paramDb_.back();
}