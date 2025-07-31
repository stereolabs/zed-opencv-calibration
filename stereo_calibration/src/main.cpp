#include "opencv_calibration.hpp"
#include <filesystem>

namespace fs = std::filesystem;

// CHANGE THIS PARAM BASED ON THE CHECKERBOARD USED
// https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html
constexpr int target_w = 9; // number of horizontal inner edges
constexpr int target_h = 6; // number of vertical inner edges
constexpr float square_size = 24.0; // mm

std::string folder = "zed-images/";

int verbose = 0;

struct extrinsic_checker {
    float rot_x_min;
    float rot_y_min;
    float rot_z_min;
    float rot_x_max;
    float rot_y_max;
    float rot_z_max;

    float rot_x_delta;
    float rot_y_delta;
    float rot_z_delta;

    float d_min;
    float d_max;
    float distance_tot;
};

std::map<std::string, std::string> parseArguments(int argc, char* argv[]);
bool writeRotText(cv::Mat& image, float rot_x, float rot_y, float rot_z, float distance, int fontSize);
float CheckCoverage(const std::vector<std::vector<cv::Point2f>>& pts, const cv::Size& imgSize);
bool updateRT(extrinsic_checker& checker_, cv::Mat r, cv::Mat t);

/// Calibration condition
const float min_coverage = 10; // in percentage
const float min_rotation = 60; // in degrees
const float acceptable_rotation = 50; // in degrees
const float min_distance = 200; // in mm
const float acceptable_distance = 150; // in mm

std::vector<std::vector<cv::Point2f>> pts_detected;

std::vector<cv::Point2f> square_valid;
int bucketsize = 480;
const int MinPts = 10;
const int MaxPts = 90;
const cv::Scalar info_color = cv::Scalar(50,205,50);
const cv::Scalar warn_color = cv::Scalar(0, 128, 255);

const bool image_stack_horizontal = true; // true for horizontal, false for vertical


void scaleKP(std::vector<cv::Point2f> &pts, cv::Size in, cv::Size out){

    float rx = out.width / static_cast<float>(in.width);
    float ry = out.height / static_cast<float>(in.height);

    for(auto &it: pts){
        it.x *= rx;
        it.y *= ry;
    }
}


struct Args {
    std::string svo_path = "";
    bool is_fisheye = false;

    void parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--svo" && i + 1 < argc) {
                svo_path = argv[++i];
            } else if (arg == "--fisheye") {
                is_fisheye = true;
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [options]\n";
                std::cout << "  --svo <file>         Path to the SVO file\n";
                std::cout << "  --help, -h          Show this help message\n";
                exit(0);
            }
        }
    }
};


int main(int argc, char *argv[]) {
    Args args;
    args.parse(argc, argv);

    setenv("ZED_SDK_ALLOW_UNCALIBRATED_MODE", "1", 1);

    std::cout << std::endl;
    std::cout << "The calibration process requires a checkerboard of known characteristics." << std::endl;
    std::cout << "Expected checkerboard size: " << target_w << "x" << target_h << " - " << square_size << "mm" << std::endl;
    std::cout << "Change those values in the code depending on the checkerboard you are using!" << std::endl;
    std::cout << std::endl;

    sl::Camera zed_camera;
    sl::InitParameters init_params;
    init_params.depth_mode = sl::DEPTH_MODE::NONE; // No depth required for calibration
    init_params.camera_resolution = sl::RESOLUTION::AUTO; // Use the camera's native resolution
    init_params.camera_fps = 30; // Set the camera FPS
    init_params.enable_image_validity_check = false; // Disable image validity check for performance
    init_params.camera_disable_self_calib = true;

    /* Setup virtual stereo camera from two ZED One cameras
    init_params.input.setVirtualStereoFromSerialNumbers(302045219, 306208880, 666);
    */

    auto status = zed_camera.open(init_params);

    // in case of a virtual stereo camera, the calibration file can be not available
    if(status != sl::ERROR_CODE::SUCCESS && status != sl::ERROR_CODE::INVALID_CALIBRATION_FILE) {
        std::cerr << "Error opening ZED camera: " << sl::toString(status) << std::endl;
        return 1;
    }

    // change can_use_calib_prior if you dont want to use the calibration file
    const bool can_use_calib_prior = status != sl::ERROR_CODE::INVALID_CALIBRATION_FILE;
    bool need_intrinsic_estimation = !can_use_calib_prior;

    std::cout << "Using prior calibration: " << (can_use_calib_prior ? "Yes" : "No") << std::endl;

    auto zed_info = zed_camera.getCameraInformation();
    sl::Resolution camera_resolution = zed_info.camera_configuration.resolution;

    StereoCalib calib;
    if (args.is_fisheye) {
        std::cout << "Using fisheye model for calibration." << std::endl;
        calib.left.disto_model_RadTan = false;
        calib.right.disto_model_RadTan = false;
    } else {
        std::cout << "Using radial-tangential model for calibration." << std::endl;
        calib.left.disto_model_RadTan = true;
        calib.right.disto_model_RadTan = true;
    }

    calib.initDefault();
    if(can_use_calib_prior)
        calib.setFrom(zed_info.camera_configuration.calibration_parameters_raw);

    sl::Mat zed_imageL(camera_resolution, sl::MAT_TYPE::U8_C4, sl::MEM::CPU);
    auto rgb_l = cv::Mat(camera_resolution.height, camera_resolution.width, CV_8UC4, zed_imageL.getPtr<sl::uchar1>());

    sl::Mat zed_imageR(camera_resolution, sl::MAT_TYPE::U8_C4, sl::MEM::CPU);
    auto rgb_r = cv::Mat(camera_resolution.height, camera_resolution.width, CV_8UC4, zed_imageR.getPtr<sl::uchar1>());

    // Number of area to fill 4 horizontally
    bucketsize = camera_resolution.width/4;

    bool frames_rot_good=true;
    cv::Mat blank = cv::Mat::zeros(camera_resolution.height, camera_resolution.width, CV_8UC1);

    cv::Mat rgb_d, rgb2_d, rgb_d_fill, display, rendering_image;


    extrinsic_checker checker;
    float cov_left = 100;
    bool angle_clb = false;
    std::vector<cv::Point3f> pts_obj_;
    for (int i = 0; i < target_h; i++) {
        for (int j = 0; j < target_w; j++) {
            pts_obj_.push_back(cv::Point3f(square_size*j, square_size*i, 0));
        }
    }
    
    // Check if the temp image folder exists and clear it
    if (fs::exists(folder)) {
        std::uintmax_t n{fs::remove_all(folder)};
    }
    // Create the temp image folder
    if(!fs::create_directories(folder)) 
    {
        std::cerr << "Error creating storage folder!";
        return 1;
    }
    
    std::vector<std::vector<cv::Point2f>> pts_init_im_l, pts_init_im_r;
    std::vector<std::vector<cv::Point3f>> pts_init_obj;


    const cv::Size display_size(720, 404);

    char key = ' ';
    int image_count = 0;
    bool coverage_mode = false;
    bool missing_left_target_on_last_pic = false;
    bool missing_right_target_on_last_pic = false;

    const std::string window_name = "ZED Calibration";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1280, 960); // Set a larger window size
    while (key != 'q' && key != 27) {

        if (zed_camera.grab() == sl::ERROR_CODE::SUCCESS) {
            zed_camera.retrieveImage(zed_imageL, sl::VIEW::LEFT_UNRECTIFIED);
            zed_camera.retrieveImage(zed_imageR, sl::VIEW::RIGHT_UNRECTIFIED);
            
            cv::resize(rgb_l, rgb_d, display_size);
            cv::resize(rgb_r, rgb2_d, display_size);

            if (!angle_clb) {
                cv::Mat rgb_with_lack_of_pts;
                std::vector<cv::Mat> channels;
                cv::split(rgb_l, channels);
                blank.setTo(0);
                float x_end,y_end;
                float x_max = 0;
                for (int i = 0; i < square_valid.size(); i++) {
                    if(square_valid.at(i).x + bucketsize > blank.size[1])
                        x_end = blank.size[1];
                    else
                        x_end = square_valid.at(i).x + bucketsize;
                    if(square_valid.at(i).y + bucketsize > blank.size[0])
                        y_end = blank.size[0];
                    else
                        y_end = square_valid.at(i).y + bucketsize;
                    if(square_valid.at(i).x>x_max)
                        x_max = square_valid.at(i).x;
                    cv::rectangle(blank, square_valid.at(i), cv::Point(x_end, y_end), cv::Scalar(128, 0, 128), -1);
                }
                channels[0] = channels[0] - blank;
                channels[2] = channels[2] - blank;
                cv::merge(channels, rgb_with_lack_of_pts);
                cv::resize(rgb_with_lack_of_pts, rgb_d_fill, display_size);
            } else {
                cv::resize(rgb_l, rgb_d_fill, display_size);
            }

            std::vector<cv::Point2f> pts;
            bool found = cv::findChessboardCorners(rgb_d, cv::Size(target_w, target_h), pts, 3);
            drawChessboardCorners(rgb_d_fill, cv::Size(target_w, target_h), cv::Mat(pts), found);

            if(image_stack_horizontal)
                cv::hconcat(rgb_d_fill, rgb2_d, display);
            else
                cv::vconcat(rgb_d_fill, rgb2_d, display);

            cv::Mat text_info = cv::Mat::ones(cv::Size(display.size[1], 200), display.type());

            if (angle_clb) {
                coverage_mode = false;
                bool ready_to_calibrate = writeRotText(text_info, checker.rot_x_delta, checker.rot_y_delta, checker.rot_z_delta, checker.distance_tot, 1);
                if(ready_to_calibrate) {
                    if(image_count >= MIN_IMAGE)                    
                        break;
                    else
                        cv::putText(text_info, "Not enough images for calibration", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, warn_color, 2);
                }
            }

            cv::vconcat(display, text_info, rendering_image);

            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << square_size;
            
            if(missing_right_target_on_last_pic)
                cv::putText(rendering_image, "Missing target on image right", cv::Point(10, display.size[0]+140), cv::FONT_HERSHEY_SIMPLEX, 0.8, warn_color, 2);
            if(missing_left_target_on_last_pic)
                cv::putText(rendering_image, "Missing target on image left", cv::Point(10, display.size[0]+110), cv::FONT_HERSHEY_SIMPLEX, 0.8, warn_color, 2);

            cv::putText(rendering_image, "Press 's' to save the current frames", cv::Point(10, display.size[0]+25), cv::FONT_HERSHEY_SIMPLEX, 0.8, info_color, 2);
            if(coverage_mode)
                cv::putText(rendering_image, "Keep going until the green covers the image, it represents coverage", cv::Point(10, display.size[0]+55), cv::FONT_HERSHEY_SIMPLEX, 0.6, info_color, 1);
            if(!frames_rot_good)
                cv::putText(rendering_image, "!!! Do not rotate the checkerboard more than 45 deg around Z !!!", 
                    cv::Point(600, display.size[0]+25), cv::FONT_HERSHEY_SIMPLEX, 0.8, warn_color, 2);
                        
            cv::imshow(window_name, rendering_image);
            key = cv::waitKey(10);

            if ((key == 's' || key == 'S')) {
                if (!angle_clb) coverage_mode = true;
                std::vector<cv::Point2f> pts_l, pts_r;
                bool found_l = cv::findChessboardCorners(rgb_d, cv::Size(target_w, target_h), pts_l);
                bool found_r = cv::findChessboardCorners(rgb2_d, cv::Size(target_w, target_h), pts_r);

                if(!found_r)
                    missing_right_target_on_last_pic = true;
                else
                    missing_right_target_on_last_pic = false;
                if(!found_l)
                    missing_left_target_on_last_pic = true;
                else
                    missing_left_target_on_last_pic = false;

                if (found_l && found_r) {

                    scaleKP(pts_l, display_size, cv::Size(camera_resolution.width, camera_resolution.height));
                    
                    if(need_intrinsic_estimation) {
                        pts_init_im_l.push_back(pts_l);

                        scaleKP(pts_r, display_size, cv::Size(camera_resolution.width, camera_resolution.height));
                        pts_init_im_r.push_back(pts_r);

                        pts_init_obj.push_back(pts_obj_);
                        // wait 3 images before running the estimate
                        if(pts_init_im_l.size()>3){
                            calib.left.K = cv::initCameraMatrix2D(pts_init_obj, pts_init_im_l, cv::Size(camera_resolution.width, camera_resolution.height));
                            calib.right.K = cv::initCameraMatrix2D(pts_init_obj, pts_init_im_r, cv::Size(camera_resolution.width, camera_resolution.height));
                            need_intrinsic_estimation = false;
                        }
                    }else{
                        if (!angle_clb) {
                            pts_detected.push_back(pts_l);
                            cov_left = CheckCoverage(pts_detected, cv::Size(camera_resolution.width, camera_resolution.height));
                            std::cout << "coverage : " << (1-cov_left)*100 << "%" << std::endl;
                            if (cov_left < 0.1) {
                                cv::Mat rvec(1, 3, CV_64FC1);
                                cv::Mat tvec(1, 3, CV_64FC1);
                                
                                auto undist_pts = calib.left.undistortPoints(pts_l);
                                
                                cv::Mat empty_dist;
                                cv::solvePnP(pts_obj_, undist_pts, calib.left.K, empty_dist, rvec, tvec, false, cv::SOLVEPNP_EPNP);

                                double rx = rvec.at<double>(0)*(180 / M_PI);
                                double ry = rvec.at<double>(1)*(180 / M_PI);
                                double rz = rvec.at<double>(2)*(180 / M_PI);

                                checker.rot_x_min = rx;
                                checker.rot_x_max = rx;
                                checker.rot_y_min = ry;
                                checker.rot_y_max = ry;
                                checker.rot_z_min = rz;
                                checker.rot_z_max = rz;

                                checker.d_min = sqrt(pow(tvec.at<double>(0), 2) + pow(tvec.at<double>(1), 2) + pow(tvec.at<double>(2), 2));
                                checker.d_max = checker.d_min;

                                angle_clb = true;
                            }
                        } else {
                            cv::Mat rvec(1, 3, CV_64FC1);
                            cv::Mat tvec(1, 3, CV_64FC1);
                            auto undist_pts = calib.left.undistortPoints(pts_l);
                            cv::Mat empty_dist;
                            bool found_ = cv::solvePnP(pts_obj_, undist_pts, calib.left.K, empty_dist, rvec, tvec, false, cv::SOLVEPNP_EPNP);
                            if (found_) {
                                frames_rot_good = updateRT(checker, rvec, tvec);
                                if(frames_rot_good)
                                    pts_detected.push_back(pts_l);
                            }
                        }
                    }
                }

                if(frames_rot_good) {
                    // saves the images
                    cv::imwrite(folder + "image_left_" + std::to_string(image_count) + ".png", rgb_l);
                    cv::imwrite(folder + "image_right_" + std::to_string(image_count) + ".png", rgb_r);
                    std::cout << " * Images saved" << std::endl;
                    image_count++;
                }
            }
        }
        sl::sleep_ms(10); 
    }

    int err = calibrate(folder, calib, target_w, target_h, square_size, zed_info.serial_number);
    if (err == 0) 
        std::cout << "CALIBRATION success" << std::endl;
        else 
        std::cout << "CALIBRATION fail" << std::endl;
    zed_camera.close();

    return 0;
}

// Function to perform linear interpolation
inline float interpolate(float x, float x0, float x1, float y0=0, float y1=100) {
     float interpolatedValue = y0 + (x - x0) * (y1 - y0) / (x1 - x0);
    interpolatedValue = (interpolatedValue < y0) ? y0 : ((interpolatedValue > y1) ? y1 : interpolatedValue); // clamp
    return interpolatedValue;
}

bool writeRotText(cv::Mat& image, float rot_x, float rot_y, float rot_z, float distance,  int fontSize) {
    bool status = false;
    // Define text from rotation and distance

    // Convert float values to string with two decimal places
    std::stringstream ss_rot_x, ss_rot_y, ss_rot_z, ss_distance;

    int rot_x_idx = interpolate(rot_x, 0, min_rotation);
    int rot_y_idx = interpolate(rot_y, 0, min_rotation);
    int rot_z_idx = interpolate(rot_z, 0, min_rotation);
    int distance_idx = interpolate(distance, 0, min_distance);

    ss_rot_x << "Rotation X: " << std::fixed << std::setprecision(1) << rot_x_idx << "%   ";
    ss_rot_y << " / Rotation Y: " << std::fixed << std::setprecision(1) << rot_y_idx << "%   ";
    ss_rot_z << " / Rotation Z: " << std::fixed << std::setprecision(1) << rot_z_idx << "%   ";
    ss_distance << " / Distance: " << std::fixed << std::setprecision(1) << distance_idx << "%";

    std::string text1 = ss_rot_x.str();
    std::string text2 = ss_rot_y.str();
    std::string text3 = ss_rot_z.str();
    std::string text4 = ss_distance.str();

    std::string text = text1 + text2 + text3 + text4;

    cv::Scalar color1, color2, color3, color4;

    // Condition on colors
    if (rot_x > min_rotation)
        color1 = cv::Scalar(0, 255, 0);
    else if (rot_x >= acceptable_rotation)
        color1 = warn_color;
    else
        color1 = cv::Scalar(0, 0, 255);

    if (rot_y > min_rotation)
        color2 = cv::Scalar(0, 255, 0);
    else if (rot_y >= acceptable_rotation)
        color2 = warn_color;
    else
        color2 = cv::Scalar(0, 0, 255);

    if (rot_z > min_rotation)
        color3 = cv::Scalar(0, 255, 0);
    else if (rot_z >= acceptable_rotation)
        color3 = warn_color;
    else
        color3 = cv::Scalar(0, 0, 255);

    if (distance > min_distance)
        color4 = cv::Scalar(0, 255, 0);
    else if (distance >= acceptable_distance)
        color4 = warn_color;
    else
        color4 = cv::Scalar(0, 0, 255);

    // Get image dimensions
    int width = image.cols;
    int height = image.rows;

    // Calculate text size
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontSize, 1, &baseline);

    // Calculate text position
    int x = (width - textSize.width) / 2;
    int y = (height + textSize.height) / 2;

    status = (rot_x > min_rotation) && (rot_y > min_rotation) && (rot_z > min_rotation) && (distance > min_distance);

    // Draw text on image with different colors
    cv::putText(image, text1, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, fontSize, color1, 2);
    cv::putText(image, text2, cv::Point(x + textSize.width / 4, y), cv::FONT_HERSHEY_SIMPLEX, fontSize, color2, 2);
    cv::putText(image, text3, cv::Point(x + textSize.width / 2, y), cv::FONT_HERSHEY_SIMPLEX, fontSize, color3, 2);
    cv::putText(image, text4, cv::Point(x + 3 * textSize.width / 4, y), cv::FONT_HERSHEY_SIMPLEX, fontSize, color4, 2);

    if(status)
        cv::putText(image, "Ready to calibrate...", cv::Point(10, y-40), cv::FONT_HERSHEY_SIMPLEX, 0.8, info_color, 2);

    return status;
}

bool CheckBucket(int min_h, int max_h, int min_w, int max_w, bool min, std::vector<std::vector<cv::Point2f>> pts) {
    int compteur = 0;

    for (int i = 0; i < pts.size(); i++) {
        for (int j = 0; j < pts.at(i).size(); j++) {
            if ((pts.at(i).at(j).x < max_w)&&(pts.at(i).at(j).x > min_w)) {
                if ((pts.at(i).at(j).y < max_h)&&(pts.at(i).at(j).y > min_h)) {
                    compteur = compteur + 1;
                }
            }
        }
    }

    if (min) {
        if (compteur > MinPts) return true;
        else return false;
    } else {
        if (compteur < MaxPts) return true;
        else return false;
    }

}

float CheckCoverage(const std::vector<std::vector<cv::Point2f>>& pts, const cv::Size& imgSize) {
    int min_h_ = 0;
    int max_h_ = bucketsize;
    float tot = 0;
    float error = 0;

    while (min_h_ < imgSize.height) {
        if (max_h_ > imgSize.height)
            max_h_ = imgSize.height;
        int min_w_ = 0;
        int max_w_ = bucketsize;
        while (min_w_ < imgSize.width) {
            if (max_w_ > imgSize.width)
                max_w_ = imgSize.width;
            if (!CheckBucket(min_h_, max_h_, min_w_, max_w_, true, pts)) {
                error++;
            } else
                square_valid.push_back(cv::Point(min_w_, min_h_));
            min_w_ += bucketsize;
            max_w_ += bucketsize;
            tot++;

        }
        min_h_ += bucketsize;
        max_h_ += bucketsize;
    }
    return error / tot;
}

bool updateRT(extrinsic_checker& checker_, cv::Mat r, cv::Mat tvec) {
    double rx = r.at<double>(0)*(180 / M_PI);
    double ry = r.at<double>(1)*(180 / M_PI);
    double rz = r.at<double>(2)*(180 / M_PI);

    std::cout << "* Rot X: " << rx << "°" << std::endl;
    std::cout << "* Rot Y: " << ry << "°" << std::endl;
    std::cout << "* Rot Z: " << rz << "°" << std::endl;

    if(fabs(rz)>45.0)
    {
        std::cerr << " * Images ignored: Rot Z > 45° ["<< rz <<"°]" << std::endl;
        return false;
    }

    // check min
    if (checker_.rot_x_min > rx)
        checker_.rot_x_min = rx;
    if (checker_.rot_y_min > ry)
        checker_.rot_y_min = ry;
    if (checker_.rot_z_min > rz)
        checker_.rot_z_min = rz;

    //check max
    if (checker_.rot_x_max < rx)
        checker_.rot_x_max = rx;
    if (checker_.rot_y_max < ry)
        checker_.rot_y_max = ry;
    if (checker_.rot_z_max < rz)
        checker_.rot_z_max = rz;

    if (checker_.d_min > sqrt(pow(tvec.at<double>(0), 2) + pow(tvec.at<double>(1), 2) + pow(tvec.at<double>(2), 2)))
        checker_.d_min = sqrt(pow(tvec.at<double>(0), 2) + pow(tvec.at<double>(1), 2) + pow(tvec.at<double>(2), 2));
    if (checker_.d_max < sqrt(pow(tvec.at<double>(0), 2) + pow(tvec.at<double>(1), 2) + pow(tvec.at<double>(2), 2)))
        checker_.d_max = sqrt(pow(tvec.at<double>(0), 2) + pow(tvec.at<double>(1), 2) + pow(tvec.at<double>(2), 2));

    // compute deltas
    checker_.rot_x_delta = checker_.rot_x_max - checker_.rot_x_min;
    checker_.rot_y_delta = checker_.rot_y_max - checker_.rot_y_min;
    checker_.rot_z_delta = checker_.rot_z_max - checker_.rot_z_min;
    checker_.distance_tot = checker_.d_max - checker_.d_min;

    std::cout << " * delta rot x: " << checker_.rot_x_delta << std::endl;
    std::cout << " * delta rot y: " << checker_.rot_y_delta << std::endl;
    std::cout << " * delta rot z: " << checker_.rot_z_delta << std::endl;
    std::cout << " * delta dist: " << checker_.distance_tot << std::endl;

    return true;
}
