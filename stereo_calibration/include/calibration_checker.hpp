#ifndef CALIBRATION_CHECKER_HPP
#define CALIBRATION_CHECKER_HPP

#include <opencv2/opencv.hpp>

typedef struct _board {
  cv::Size board_size = {
      0, 0};  // Number of inner corners per a chessboard row and column
  float square_size =
      0.0f;  // Size of a square in your defined unit (point, millimeter,etc).
  std::vector<cv::Point3f> objp;            // 3D points in real world space
  cv::Size2f board_size_mm = {0.0f, 0.0f};  // Physical size of the board in mm
} Board;

typedef struct _detected_board_params {
  cv::Point2f pos = {
      -1.0f, -1.0f};   // Normalized position of the checkerboard in the image
  float size = -1.0f;  // Normalized size of the checkerboard
  float skew = -1.0f;  // Normalized skew of the checkerboard
} DetectedBoardParams;

class CalibrationChecker {
 public:
  CalibrationChecker(cv::Size board_size, float square_size, bool verbose);
  ~CalibrationChecker() = default;

  // Test if the detected corners form a valid sample
  bool testSample(const std::vector<cv::Point2f>& corners, cv::Size image_size);

  // Retrieve valid corners
  const std::vector<std::vector<cv::Point2f>>& getValidCorners() const {
    return validCorners_;
  }

  // Retrieve valid sample count
  size_t getValidSampleCount() const { return validCorners_.size(); }

  // Calculate the sample collection status according to the stored samples
  bool evaluateSampleCollectionStatus(float& size_score, float& skew_score,
                                      float& pos_score_x,
                                      float& pos_score_y) const;

 private:
  // Calculate the parameter of a detected checkerboard
  DetectedBoardParams getDetectedBoarParams(
      const std::vector<cv::Point2f>& corners, cv::Size image_size);

  // Check if the detected corners are valid
  bool isGoodSample(const DetectedBoardParams& params,
                    const std::vector<cv::Point2f>& corners,
                    const std::vector<cv::Point2f>& prev_corners);

  // Helper functions
  std::vector<cv::Point2f> get_outside_corners(
      const std::vector<cv::Point2f>&
          corners);  // Get the 4 outside corners of a checkerboard
  float compute_skew(
      const std::vector<cv::Point2f>&
          outside_corners);  // Compute skew based on the 4 outside corners
  float compute_area(
      const std::vector<cv::Point2f>&
          outside_corners);  // Compute area based on the 4 outside corners

 private:
  Board board_;

  std::vector<DetectedBoardParams>
      paramDb_;  // Database of previously detected board parameters
  std::vector<std::vector<cv::Point2f>>
      validCorners_;  // All the corners associated to the single parameters in
                      // paramDb_

  const DetectedBoardParams idealParams_ = {
      cv::Point2f(
          0.65f,  // Checkerboard X position should cover 65% of the image width
          0.65f  // Checkerboard Y position should cover 65% of the image height
          ),
      0.4f,  // Checkerboard size variation should be at least 40%
      0.6f   // Checkerboard skew variation should be at least 70%
  };         // Ideal parameters for a good sample database
  const size_t min_samples_ =
      20;  // Minimum number of samples to consider the database complete
  const size_t max_samples_ =
      50;  // Maximum number of samples to consider the database complete (used if it's not possible to reach the ideal parameters)

  bool verbose_ = false;
};

#endif  // CALIBRATION_CHECKER_HPP
