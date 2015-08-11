//  main.cpp
//  OpenSeqSLAM
//
//  Created by Saburo Okita on 14/05/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

#include "OpenSeqSLAM.h"

using namespace std;
using namespace cv;


/**
 * Load the Nordland dataset
 **/
vector<Mat> loadDataset( string path ) {
    char temp[100];
    vector<Mat> images;

    for( int i = 1; i < 35700; i += 100) {

        sprintf( temp, "images-%05d.png", i );
        std::string filepath =  path + temp;
        Mat image = imread( path + temp );
        images.push_back( image );
    }

    return images;
}



int main()
{
    vector<Mat> spring = loadDataset( "spring/" );
    vector<Mat> winter = loadDataset( "winter/" );

    OpenSeqSLAM seq_slam;

    /* Preprocess the image set first */
    vector<Mat> preprocessed_spring = seq_slam.preprocess( spring );
    vector<Mat> preprocessed_winter = seq_slam.preprocess( winter );

    /* Find the matches */
    Mat matches = seq_slam.apply( preprocessed_spring, preprocessed_winter );


    //CvFont font = cvFont("Helvetica", 20.0, CV_RGB(255, 0, 0) );
    cv::namedWindow("");
    moveWindow("", 0, 0);

    float threshold = 0.99;

    float * index_ptr = matches.ptr<float>(0);
    float * score_ptr = matches.ptr<float>(1);

    for( size_t x = 0; x < spring.size(); x++ ) {
        int index = static_cast<int>(index_ptr[x]);

        /* Append the images together */
        Mat appended( 32, 64 * 2, CV_8UC3, Scalar(0) );
        spring[x].copyTo( Mat(appended, Rect(0, 0, 64, 32) ));

        if( score_ptr[x] < threshold )
            winter[index].copyTo( Mat(appended, Rect(64, 0, 64, 32) ));

        resize(appended, appended, Size(), 8.0, 8.0 );


        std::cout << "Spring [" << std::to_string(x) << "]\n";


        /* The lower the score, the lower the differences between images */
        if( score_ptr[x] < threshold ) std::cout << "Winter [" << std::to_string(index) << "]" << std::endl;
        else std::cout << "Winter [None]" << std::endl;

        imshow( "", appended );
        waitKey(100);
    }


    return 0;
}

