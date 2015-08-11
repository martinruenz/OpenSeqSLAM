/*
     OpenSeqSLAM
     Copyright 2013, Niko Sünderhauf Chemnitz University of Technology niko@etit.tu-chemnitz.de

     OpenSeqSLAM is an open source Matlab implementation of the original SeqSLAM algorithm published by Milford and Wyeth at ICRA12 [1]. SeqSLAM performs place recognition by matching sequences of images.
*/

//
//  OpenSeqSLAM.h
//  OpenSeqSLAM
//
//  Created by Saburo Okita on 20/05/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

class OpenSeqSLAM {
public:
    OpenSeqSLAM();
    void init( int patch_size, int local_radius, int matching_distance );
    
    
    cv::Mat preprocess( cv::Mat& image );
    std::vector<cv::Mat> preprocess( std::vector<cv::Mat>& images );
    cv::Mat calcDifferenceMatrix( std::vector<cv::Mat>& set_1, std::vector<cv::Mat>& set_2 );
    
    cv::Mat enhanceLocalContrast( cv::Mat& diff_matrix, int local_radius = 10 );
    std::pair<int, double> findMatch( cv::Mat& diff_mat, int N, int matching_dist );
    cv::Mat findMatches( cv::Mat& diff_mat, int matching_dist = 10 );
    
    cv::Mat apply( std::vector<cv::Mat>& set_1, std::vector<cv::Mat>& set_2 );
    
protected:
    int patchSize;
    int localRadius;
    int matchingDistance;
    int RWindow;
    float minVelocity;
    float maxVelocity;
    cv::Size imageSize;
    
    double convertToSampleStdDev( double pop_stddev, int N );
    cv::Mat normalizePatch( cv::Mat& image, int patch_size );
    
};
