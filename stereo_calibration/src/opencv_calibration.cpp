#include "opencv_calibration.hpp"

int calibrate(const std::string& folder, StereoCalib &calib_data, int target_w, int target_h, float square_size, int serial, bool save_calib_mono, bool use_intrinsic_prior){

    std::vector<cv::Mat> left_images, right_images;

    /// Read images
    cv::Size imageSize = cv::Size(0, 0);
    int img_number = 0;

    while (1) {
        cv::Mat grey_l = cv::imread(folder + "image_left_" + std::to_string(img_number) + ".png", cv::IMREAD_GRAYSCALE);
        cv::Mat grey_r = cv::imread(folder + "image_right_" + std::to_string(img_number) + ".png", cv::IMREAD_GRAYSCALE);
        if (!grey_l.empty() && !grey_r.empty()) {
            if (imageSize.width == 0)
                imageSize = grey_l.size();
            else
                if (imageSize != left_images.back().size()) {
                std::cout << "Image number " << img_number << " does not have the same size as the previous ones"<< imageSize<<" vs "<< left_images.back().size() << std::endl;
                break;
            }

            left_images.push_back(grey_l);
            right_images.push_back(grey_r);
        }else break;
        img_number++;
    }

    std::cout << "\n\t"<< left_images.size() << " images opened" << std::endl;

    // Define object points of the target
    std::vector<cv::Point3f> pattern_points;
    for (int i = 0; i < target_h; i++) {
        for (int j = 0; j < target_w; j++) {
            pattern_points.push_back(cv::Point3f(square_size*j, square_size*i, 0));
        }
    }

    std::vector<std::vector < cv::Point3f>> object_points;
    std::vector<std::vector < cv::Point2f>> pts_l, pts_r;

    cv::Size t_size(target_w, target_h);

    for (int i = 0; i < left_images.size(); i++) {
        std::vector<cv::Point2f> pts_l_, pts_r_;
        bool found_l = cv::findChessboardCorners(left_images.at(i), t_size, pts_l_, 3);
        bool found_r = cv::findChessboardCorners(right_images.at(i), t_size, pts_r_, 3);

        if (found_l && found_r) {
            cv::cornerSubPix(left_images.at(i), pts_l_, cv::Size(5, 5), cv::Size(-1, -1),
                    cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001));

            cv::cornerSubPix(right_images.at(i), pts_r_, cv::Size(5, 5), cv::Size(-1, -1),
                    cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001));

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
        std::cout << " !!! Not enough images with the target detected !!!" << std::endl;
        std::cout << " Please perform a new data acquisition." << std::endl << std::endl;
        return EXIT_FAILURE;
    } else {
        std::cout << " * Enough points detected" << std::endl;
    
        auto flags = use_intrinsic_prior ? cv::CALIB_USE_INTRINSIC_GUESS : 0;
        auto rms_l = calib_data.left.calibrate(object_points, pts_l, imageSize, flags);
        auto rms_r = calib_data.right.calibrate(object_points, pts_r, imageSize, flags);
        std::cout << " * Reprojection error:\t Left "<<rms_l<<" Right "<<rms_r<< std::endl;

        auto err = calib_data.calibrate(object_points, pts_l, pts_r, imageSize, cv::CALIB_USE_INTRINSIC_GUESS + cv::CALIB_ZERO_DISPARITY);
        std::cout << " * Reprojection error:\t Stereo " << err << std::endl;

        if(rms_l > 0.5f || rms_r > 0.5f || err > 0.5f)
            std::cout<<"\n\t !! Warning !!\n The reprojection error looks too high, check that the lens are clean (sharp images) and that the pattern is printed/mounted on a strong and flat surface."<<std::endl;
        
        std::cout << " ** Camera parameters **" << std::endl;
        std::cout << "  * Intrinsic mat left:\t" << calib_data.left.K << std::endl;
        std::cout << "  * Distortion mat left:\t" << calib_data.left.D << std::endl;
        std::cout << "  * Intrinsic mat right:\t" << calib_data.right.K << std::endl;
        std::cout << "  * Distortion mat right:\t" << calib_data.right.D << std::endl;
        std::cout << " ** Extrinsic parameters **" << std::endl;
        std::cout << "  * Translation:\t" << calib_data.T << std::endl;
        std::cout << "  * Rotation:\t" << calib_data.Rv << std::endl;

        std::cout << std::endl << "*** Calibration file ***" << std::endl;        
        std::string calib_filename = "zed_calibration_" + std::to_string(serial) + ".yml";

        std::cout<<" * Saving calibration file: " << calib_filename << std::endl;
        std::cout<<" * If you want to use this calibration with the ZED SDK, you can load it by using sl::InitParameters::optional_opencv_calibration_file" << std::endl;

        cv::FileStorage fs(calib_filename, cv::FileStorage::WRITE);
        if (fs.isOpened()) {
            fs << "Size" << imageSize;
            fs << "K_LEFT" << calib_data.left.K<< "K_RIGHT" << calib_data.right.K;

            if(calib_data.left.disto_model_RadTan) {
                fs << "D_LEFT" << calib_data.left.D << "D_RIGHT" << calib_data.right.D;
            } else {
                fs << "D_LEFT_FE" << calib_data.left.D << "D_RIGHT_FE" << calib_data.right.D;
            }

            fs << "R" << calib_data.Rv << "T" << calib_data.T;
            fs.release();
        } else
            std::cout << "Error: can not save the extrinsic parameters\n";
    }
    return EXIT_SUCCESS;
}