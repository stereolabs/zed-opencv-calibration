///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2025, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "GLViewer.hpp"

#include <opencv2/opencv.hpp>

// Using std and sl namespaces
using namespace std;
using namespace sl;


struct Args {
    std::string optional_settings_path = "";
    std::string svo_path = "";

    void parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--calib_path" && i + 1 < argc) {
                optional_settings_path = argv[++i];
            } else if (arg == "--svo" && i + 1 < argc) {
                svo_path = argv[++i];
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [options]\n";
                std::cout << "  --calib_path <file>  Path to the optional calibration file\n";
                std::cout << "  --svo <file>         Path to the SVO file\n";
                std::cout << "  --help, -h          Show this help message\n";
                exit(0);
            }
        }
    }
};

cv::Mat cvtDisto(sl::CameraParameters &camera_param, bool fisheye) {
    cv::Mat disto;
    if(!fisheye){
        const int num_disto_coeffs = 12; // Number of distortion coefficients in ZED SDK
        disto = cv::Mat(1, num_disto_coeffs, CV_64F);
        for(int d = 0; d < num_disto_coeffs; ++d)
            disto.at<double>(0, d) = camera_param.disto[d];
    }else{
        const double *ptr = camera_param.disto;
        disto = cv::Mat (1, 4, CV_64FC1);
        disto.at<double>(0) = ptr[0];
        disto.at<double>(1) = ptr[1];
        disto.at<double>(2) = ptr[4];
        disto.at<double>(3) = ptr[5];
    }
    
    std::cout<<"Distortion Coefficients:\n"<<disto<<std::endl;
    
    return disto;
}

cv::Mat cvtCameraParam(sl::CameraParameters &camera_param) {
    // Convert the ZED camera parameters to OpenCV format
    cv::Mat camera_matrix(3, 3, CV_64F);
    camera_matrix.at<double>(0, 0) = camera_param.fx;
    camera_matrix.at<double>(1, 1) = camera_param.fy;
    camera_matrix.at<double>(0, 2) = camera_param.cx;
    camera_matrix.at<double>(1, 2) = camera_param.cy;
    camera_matrix.at<double>(2, 2) = 1.0;
    return camera_matrix;
}

// Conversion function between sl::Mat and cv::Mat
cv::Mat slMat2cvMat(sl::Mat &input) {
    int cv_type = -1;
    switch (input.getDataType()) {
        case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM::CPU));
}

void reproDepth(cv::Mat &pc, cv::Mat &im, cv::Mat &disto, cv::Mat &camera_matrix, bool fisheye) {
    const int w = pc.cols;
    const int h = pc.rows;

    float d_min = 9999.f; // Minimum depth threshold
    float d_max = 0.0f; // Maximum depth threshold

    std::vector<cv::Point3f> world_points;
    world_points.reserve(w * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const auto &p = pc.at<cv::Vec4f>(y, x);
            if (std::isfinite(p[2])){  // Check if the point is valid
                cv::Point3f point(p[0], -p[1], -p[2]);

                // convert to image coordinates
                world_points.emplace_back(point);
                d_min = std::min(d_min, point.z);
                d_max = std::max(d_max, point.z);
            }
        }
    }
    world_points.shrink_to_fit();

    cv::Scalar blue(0, 0, 255, 30);
    cv::Scalar red(255, 0, 0, 30);

    std::vector<cv::Point2f> image_points;
    // Undistort the point cloud using OpenCV
    auto tvec = cv::Mat::zeros(3, 1, CV_64F); // Translation vector
    auto rvec = cv::Mat::zeros(3, 1, CV_64F); // Rotation vector
    if(fisheye) {
        // For fisheye cameras, use the fisheye projection
        cv::fisheye::projectPoints(world_points, image_points, rvec, tvec, camera_matrix, disto);
    } else {
        // For non-fisheye cameras, use the standard projectPoints function
        cv::projectPoints(world_points, rvec, tvec, camera_matrix, disto, image_points);
    }

    int count = 0;
    for (const auto &p : image_points) {
        if (p.x >= 0 && p.x < im.cols && p.y >= 0 && p.y < im.rows) {
            // Draw the projected points on the image
            //scale pt
            float scale = (world_points[count].z - d_min) / (d_max - d_min);
            scale = std::max(0.0f, std::min(1.0f, scale)); // Clamp to [0, 1]
            cv::Scalar color = blue * scale + red * (1.0f - scale); // Interpolate between blue and red
            

            auto &color_im = im.at<cv::Vec4b>(static_cast<int>(p.y), static_cast<int>(p.x));

            // fade the color based on depth
            color_im = (color_im * 0.5 + cv::Vec4b(color[0], color[1], color[2], color[3]) * 0.5);
        }
        count++;
    }    
    cv::imshow("Projected Points", im);    
}

cv::Mat createMaskUsingUndistortPoints( int width, int height, cv::Mat& camera_matrix, cv::Mat& dist_coeffs, cv::Mat &new_camera_matrix) {
    cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);

    try {
        // Create all pixel coordinates
        std::vector<cv::Point2f> distorted_points;
        distorted_points.reserve(width * height);
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) 
                distorted_points.emplace_back(static_cast<float>(x), static_cast<float>(y));            
        }

        // Undistort points to check validity
        std::vector<cv::Point2f> undistorted_points;
        cv::fisheye::undistortPoints(distorted_points, undistorted_points, camera_matrix, dist_coeffs, cv::Mat(), new_camera_matrix);

        // Collect valid undistorted points (inliers)
        
        for (const auto &pt : undistorted_points) {
            if (pt.x >= 0 && pt.x < width && pt.y >= 0 && pt.y < height) {
                mask.at<uchar>(static_cast<int>(pt.y), static_cast<int>(pt.x)) = 255; // Mark valid points as 0 (black)
            }
        }
                    
        // Apply morphological operations to smooth the mask
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error in undistortPoints: " << e.what() << std::endl;
        mask = cv::Mat::zeros(height, width, CV_8UC1);
    }

    return mask;
}


int main(int argc, char **argv) {
    Args args;
    args.parse(argc, argv);

    Camera zed;
    // Set configuration parameters for the ZED
    InitParameters init_parameters;
    init_parameters.depth_mode = DEPTH_MODE::NEURAL;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed
    init_parameters.sdk_verbose = 1;
    init_parameters.maximum_working_resolution = sl::Resolution(0, 0);

    if (!args.svo_path.empty()) {
        std::cout << "Using SVO file: " << args.svo_path << std::endl;
        init_parameters.input.setFromSVOFile(sl::String(args.svo_path.c_str()));
    } else {
        init_parameters.camera_resolution = RESOLUTION::AUTO; // Set the camera resolution
        init_parameters.camera_fps = 30; // Set the camera FPS
    }

    if (!args.optional_settings_path.empty()) {
        std::cout << "Using optional settings from: " << args.optional_settings_path << std::endl;
        init_parameters.optional_opencv_calibration_file = sl::String(args.optional_settings_path.c_str());
    }


    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    // Automatically set to the optimal resolution
    sl::Resolution res(720, 404); 
    
    auto camera_config = zed.getCameraInformation(res).camera_configuration;

    Mat point_cloud(res, sl::MAT_TYPE::F32_C4, MEM::BOTH);
    auto pc_ocv = slMat2cvMat(point_cloud);
    

    Mat image(res, sl::MAT_TYPE::U8_C4, MEM::CPU);
    auto im_ocv = slMat2cvMat(image);


    Mat image_rect(res, sl::MAT_TYPE::U8_C4, MEM::CPU);
    auto im_rect_ocv = slMat2cvMat(image_rect);

    const bool fisheye = true;

    auto stream = zed.getCUDAStream();

    auto K = cvtCameraParam(camera_config.calibration_parameters_raw.left_cam);
    auto disto = cvtDisto(camera_config.calibration_parameters_raw.left_cam, fisheye);

    auto K_new = cvtCameraParam(camera_config.calibration_parameters.left_cam);
    auto mask_cv = createMaskUsingUndistortPoints(res.width, res.height, K, disto, K_new);
    
    // Convert cv::Mat mask to sl::Mat
    cv::imshow("Mask", mask_cv);
    cv::imwrite("mask.png", mask_cv);

    sl::Mat mask_sl;
    mask_sl.read("mask.png");
    zed.setRegionOfInterest(mask_sl);

    // Point cloud viewer
    GLViewer viewer;
    // Initialize point cloud viewer 
    GLenum errgl = viewer.init(argc, argv, camera_config.calibration_parameters.left_cam, stream, res);
    if (errgl != GLEW_OK) {
        print("Error OpenGL: " + std::string((char*)glewGetErrorString(errgl)));
        return EXIT_FAILURE;
    }

    RuntimeParameters runParameters;

    // Main Loop
    while (viewer.isAvailable()) {        
        // Check that a new image is successfully acquired
        if (zed.grab(runParameters) <= ERROR_CODE::SUCCESS) {
            // retrieve the current 3D coloread point cloud in GPU
            zed.retrieveMeasure(point_cloud, MEASURE::XYZRGBA, MEM::GPU, res);
            viewer.updatePointCloud(point_cloud);

            point_cloud.updateCPUfromGPU();

            zed.retrieveImage(image, VIEW::LEFT_UNRECTIFIED, MEM::CPU, res);

            reproDepth(pc_ocv, im_ocv, disto, K, fisheye);
            zed.retrieveImage(image_rect, VIEW::LEFT, MEM::CPU, res);
            
            // Apply mask to rectified image
            cv::Mat im_rect_masked = im_rect_ocv.clone();
            for (int y = 0; y < im_rect_masked.rows; ++y) {
                for (int x = 0; x < im_rect_masked.cols; ++x) {
                    if (mask_cv.at<uchar>(y, x) == 0) { // If mask is 0 (outside ROI)
                        im_rect_masked.at<cv::Vec4b>(y, x) = cv::Vec4b(0, 0, 0, 0); // Set to black/transparent
                    }
                }
            }
            
            cv::imshow("Rectified Image", im_rect_ocv);
            cv::imshow("Masked Rectified Image", im_rect_masked);
            cv::waitKey(10); // Wait for a short time to update the image display
        }
    }
    // free allocated memory before closing the ZED
    point_cloud.free();

    // close the ZED
    zed.close();

    return EXIT_SUCCESS;
}
