//
//  OpenCVWrapper.m
//  ObjectDetection
//
//  Created by A. Ichwan Yasir on 28/06/21.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//

#ifdef __cplusplus
#undef NO
#undef YES
#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#endif
#include <thread>
#import "ImageProcessing.h"
#import "SAMCubicSpline.h"

using namespace std;
using namespace cv;

@implementation ImageProcessing

+ (NSString *)openCVVersionString {
return [NSString stringWithFormat:@"OpenCV Version %s",  CV_VERSION];
}
 
#pragma mark Public
 
+ (UIImage *)toGray:(UIImage *) source {
    cout << "OpenCV: ";
    return [ImageProcessing _imageFrom:[ImageProcessing _grayFrom:[ImageProcessing _matFrom:source]]];
}

+ (UIImage *)drawRect:(UIImage *) source andReacts: (CGRect *) rects andPoints: (CGPoint *) points {
    cout << "OpenCV: ";
    Mat matSource = [ImageProcessing _matFrom:source];
    int i;
    int thickness = 10;
    int sz = (sizeof(rects));
    for (i = 0; i < sz; ++i) {
        CGRect rec = rects[i];
        int x = rec.origin.x;
        int y = rec.origin.y;
        int width = rec.size.width;
        int height = rec.size.width;
        cv::Point pt1(x, y);
        cv::Point pt2(x + width, y + height);
        
        rectangle(matSource, pt1, pt2, cv::Scalar(0, 255, 0), thickness);
    }
    
    int j;
    int szp = 21;
    for (j = 0; j < szp; ++j) {
        CGPoint point = points[j];
        cv::Point pt(point.x, point.y);
        cv::circle(matSource, pt, 10, cv::Scalar(255,255,255), FILLED , 11,0);
    }
    
    
    return [ImageProcessing _imageFrom: matSource];
}


/**
 * This function performs image processing that paint nails.
 *
 * @param _frame     //Image in RGB format containing a single hand whose nails we'll add color.
 *            The wrist of the hand needs to be visible for the algorithm to work properly.
 * @param nailsize //String indicating the desired size of the final nail (can be "small", "medium", or "large").
 *
 * @param orgsize //Original size of frame before processing.
 *                   of the bounding boxes or any other value >0.
 *                   This parameter exists because it is more efficient to perform image processing
 *                   with a low resolution version of frame and at the end resize the result to the original size
 *                   and paste it into frame so frame is never resized.
 *
 * @return //The processed image with nails painted.
 */
+ (UIImage*) paintNailDynamic: (UIImage*) _frame andMC: (ModelClass *) mc andNailSize: (NSString*) nailsize andOrgSize: (float*) orgsize andResolution: (int) resolution andColor: (NSString*) _color {
    cv::Mat frame = [ImageProcessing _matFrom:_frame];
    cv::cvtColor(frame, frame, cv::COLOR_RGBA2BGR);
    cv::Mat artColor;
    bool useArt = mc.get_useart;
    if (useArt) {
        artColor = [ImageProcessing _matFrom:mc.get_artcolor];
        cv::cvtColor(artColor, artColor, cv::COLOR_RGBA2BGR);
    }
    
    
    cv::Size modelsize = frame.size();
    if (modelsize.width < 2000){
        modelsize.width = 2048;
        modelsize.height = 2048;
    }
    cv::Mat image;
//    NSString *_color = @"#05ab42";
    cv::Mat color;
    color = [ImageProcessing colorWithHexString: _color andWeight: 150 andHeight:150];

    cv::Mat imageOrg = frame.clone(); // original image never resized so the its resolution is preserved
    if(frame.size() != modelsize){
        resize(frame, image, modelsize);
    } else{
        image = frame.clone(); // create a copy of the image to resize (so original image is never resized)
    }
    
    NSMutableArray *boxespreds = [mc get_boundingBox];
    NSMutableArray *nailkeypoints= [mc get_nailkeypoints];
    
    _drawBoxLine(imageOrg, boxespreds, nailkeypoints,  image, frame, color, useArt, artColor, false);
    
    // resize hand outline contour
    cv::Mat handContour;
    cv::Size finalsize(frame.cols, frame.rows);

//    Mat finalimage;

    cv::cvtColor(imageOrg, imageOrg, cv::COLOR_RGB2BGRA);
    return [ImageProcessing _imageFrom:imageOrg];

}

#pragma mark Private

+ (Mat)_grayFrom:(Mat)source {
    cout << "-> grayFrom ->";
     
    Mat result;
    cvtColor(source, result, COLOR_BGR2GRAY);
     
    return result;
}

+ (Mat) colorWithHexString:(NSString *)str_HEX  andWeight:(int) w andHeight: (int) h{
    long red = 0;
    long green = 0;
    long blue = 0;
    sscanf([str_HEX UTF8String], "#%02lX%02lX%02lX", &red, &green, &blue);
    Scalar rgb(blue, green, red);
    
    cv::Mat m(w, h, CV_8UC3);
    m.setTo(rgb);
    return m;
}

/**
 * This function rotates image by angle.
 *
 * @param source     //Image to be rotated counter-clockwise.
 * @param angle     //Angle to rotate image in degrees. Negative values mean clockwise rotation.
 */
+ (Mat) _rotateImage:(Mat)source andAngle:(float) angle {
    cv::Point2f centerPoint(source.cols/2., source.rows/2.);
    cv::Mat r = cv::getRotationMatrix2D(centerPoint, angle, 1.0);
    cv::warpAffine(source, source, r, source.size(), INTER_LINEAR);
    return source;
}


/**
 * This function computes from a mean value the alpha and beta values needed to perform illumination change.
 *
 * @param mean    //Mean value used to compute alpha and beta.
 * @return //Array containing the alpha and beta values computed.
 */
float * getBetaAlpha(float mean, float* output) {
    //Log.i("Mean= ", String.valueOf(mean));
    float r[2];
    r[0] = 2 - 2*(mean / 255.f);
    r[1] = 2*(mean / 255.f);
    output = r;
    return output;
}

/**
 * This function computes the mean value of mat.
 *
 * @param mat     //1D mat.
 * @return //Mean value of mat.
 */
+ (float) meanMat:(Mat ) mat {
    int cols = mat.cols;
    int rows = mat.rows;
    float mean = 0;

    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            double mp = mat.at<double>(i,j);
            mean += mp;
        }
    }

    mean = mean/(cols*rows);
    return mean;
}


+ (Mat)_matFrom:(UIImage *)source {
   Mat imageMat;
   UIImageToMat(source, imageMat);
   return imageMat;
}

+ (UIImage *)_imageFrom:(Mat)source {
   return MatToUIImage(source);
}

+ (Mat)_mat8UC3From:(UIImage *)source {
   cout << "matFrom ->";
    
   CGImageRef image = CGImageCreateCopy(source.CGImage);
   CGFloat cols = CGImageGetWidth(image);
   CGFloat rows = CGImageGetHeight(image);
   Mat result(rows, cols, CV_8UC3);
    
   CGBitmapInfo bitmapFlags = kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault;
   size_t bitsPerComponent = 8;
   size_t bytesPerRow = result.step[0];
   CGColorSpaceRef colorSpace = CGImageGetColorSpace(image);
    
   CGContextRef context = CGBitmapContextCreate(result.data, cols, rows, bitsPerComponent, bytesPerRow, colorSpace, bitmapFlags);
   CGContextDrawImage(context, CGRectMake(0.0f, 0.0f, cols, rows), image);
   CGContextRelease(context);
    
   return result;
}

void _drawBoxLine(Mat imageOrg, NSMutableArray *boxespreds, NSMutableArray *nailkeypoints, cv::Mat image, cv::Mat frame, cv::Mat color, bool useArt, Mat artColor, bool drawBox) {
    
    std::vector<std::thread> threads;
    int height = image.rows;
    int width = image.cols;
    
    dispatch_group_t group = dispatch_group_create();
    for(int i = 0; i < boxespreds.count; i++)
    {
        NSArray *bs = [boxespreds objectAtIndex:i];
        // [ymin, xmin, ymax, xmax]
        float ymin = [[bs objectAtIndex:0] floatValue] * height;
        float xmin = [[bs objectAtIndex:1] floatValue] * width;
        float ymax = [[bs objectAtIndex:2] floatValue] * height;
        float xmax = [[bs objectAtIndex:3] floatValue] * width;
        
        float maxside = 150;
        
        NSArray *nks = [nailkeypoints objectAtIndex:i];
        float x4 = [[nks objectAtIndex:6] floatValue] * maxside;
        float y4 = [[nks objectAtIndex:7] floatValue] * maxside;
        float x8 = [[nks objectAtIndex:14] floatValue] * maxside;
        float y8 = [[nks objectAtIndex:15] floatValue] * maxside;
        
        float angle = _getAngle(CGPointMake(x8, y8), CGPointMake(x4, y4));
        
        threads.push_back(std::thread(_paintSingleNail, std::ref(imageOrg), xmin, ymin, xmax, ymax, image, frame, nailkeypoints, i, color, useArt, artColor, angle));
    }
    
    for (auto &th : threads) {
      th.join();
    }
}

float _getAngle(CGPoint point1, CGPoint point2){
    float x1 = (float) point1.x;
    float y1 = (float) point1.y;
    float x2 = (float) point2.x;
    float y2 = (float) point2.y;

    // Get angles in degrees between -90° to 90°
    float angle = (float) (atan((y2 - y1) / (x2 - x1)) * 360 / (2 * M_PI));

    if((x1 < x2) && (y2 < y1)){
        angle = -angle;

    } else if((x2 < x1) && (y2 < y1)){
        angle = 180 - angle;

    } else if((x2 < x1) && (y1 < y2)){
        angle = 180 + abs(angle);

    } else if((x1 < x2) && (y1 < y2)){
        angle = 360 - angle;

    }
    return  angle;
}

void _paintSingleNail(cv::Mat imageOrg, float xmin,float ymin,float xmax,float ymax, cv::Mat image, cv::Mat frame, NSMutableArray* nailkeypoints, int pred, cv::Mat color, bool useArt, Mat artColor, float angle){
    int xminoff=2;
    int yminoff=2;
    int xmaxoff=2;
    int ymaxoff=2;
    int ymin_raw = ymin - yminoff;
    int ymax_raw = ymax + ymaxoff;
    int xmin_raw = xmin - xminoff;
    int xmax_raw = xmax + xmaxoff;
    
    // assert we get correct values
    if(ymin<0){ //check values
        ymin=0;
    } if(xmin<0){
        xmin=0;
    } if(ymax > image.rows){
        ymax = image.rows;
    } if(xmax > image.cols){
        xmax = image.cols;
    } if(ymin_raw<0){ // check raw values
        ymin_raw=0;
    } if(xmin_raw<0){
        xmin_raw=0;
    } if(ymax_raw > image.rows){
        ymax_raw = image.rows;
    } if(xmax_raw > image.cols){
        xmax_raw = image.cols;
    }
    
    // Check submat size will not equal image size
    if(((int) round(image.cols))<=(xmax-xmin) || ((int) round(image.rows)<=(ymax-ymin))){
        return;
    }
    if(((int) round(image.cols))<=(xmax_raw-xmin_raw) || ((int) round(image.rows)<=(ymax_raw-ymin_raw))){
        return;
    }
    
    
    // Factor Frame to image original
    double xfactor = (double)frame.cols/(double)image.cols;
    double yfactor = (double)frame.rows/(double)image.rows;
    
    // Declare final image
    Mat finalImage = image.clone();
    
    // Crop Nail
    cv::Rect ROI((int)xmin, (int)ymin, (int)xmax - (int)xmin , (int)ymax - (int)ymin);
    cv::Mat nail = image(ROI);
    cv::Rect ROIRaw((int)xmin_raw, (int)ymin_raw, (int)xmax_raw - (int)xmin_raw, (int)ymax_raw - (int)ymin_raw);
    cv::Mat nail_raw = image(ROIRaw);
    
    // Create Mask
    NSArray *nks = [nailkeypoints objectAtIndex:pred];
    //cv::Mat maskPart = [ImageProcessing _createMask: nks andImage: imageOrg andXmin:xmin andYmin:ymin andXmax:xmax andYmax:ymax andAnchor: -1 andDilate: 1 andKernel: 1 andUseDelate: false andXFac: xfactor andYFac: yfactor];
    cv::Mat mask = [ImageProcessing _createMask: nks andImage: imageOrg andXmin:xmin andYmin:ymin andXmax:xmax andYmax:ymax andAnchor: -1 andDilate: 2 andKernel: 3 andUseDelate: true andXFac: xfactor andYFac: yfactor];
    cv::Mat nail_rawmask = [ImageProcessing _createMask: nks andImage: imageOrg andXmin:xmin andYmin:ymin andXmax:xmax andYmax:ymax andAnchor: -1 andDilate: 1 andKernel: 3 andUseDelate: true andXFac: xfactor andYFac: yfactor];
    
    // Resize the mask
    cv::Size maskSize(nail.cols, nail.rows);
    cv::Size rawmaskSize(nail_raw.cols, nail_raw.rows);
    cv::resize(mask, mask, maskSize);
    //cv::resize(maskPart, maskPart, maskSize);
    cv::resize(nail_rawmask, nail_rawmask, rawmaskSize);
    
    // Apply mask
    cv::threshold(mask, mask, 0, 255, THRESH_BINARY_INV);
    //cv::threshold(maskPart, maskPart, 0, 255, THRESH_BINARY_INV);
    cv::threshold(nail_rawmask, nail_rawmask, 0, 255, THRESH_BINARY_INV);
    //cv::Mat maskPart = mask.clone();
    cv::Mat mask_inv;
    cv::bitwise_not(mask, mask_inv);
    cv::Mat nail_rawmask_inv;
    cv::bitwise_not(nail_rawmask, nail_rawmask_inv);
    cv::Mat nailmasked;
    cv::bitwise_and(nail, nail, nailmasked, mask);
    /////////////////////////
    ///
    /// Add COLOR
    ///
    cv::Size newcolorSize = cv::Size(nail.cols, nail.rows);
    cv::Mat newcolor;
    cv::Mat newcolor2;
    cv::resize(color, newcolor2, newcolorSize);
    
    if(!useArt){
        cv::resize(color, newcolor, newcolorSize);
    } else {
        //rotateImage(artColor.clone(), newcolor, angle);
        newcolor = [ImageProcessing _rotateImage: artColor.clone() andAngle: 180];
        newcolor = [ImageProcessing _rotateImage: newcolor andAngle: angle];
        resize(newcolor, newcolor, newcolorSize);

        // add the original art image to fill possible black spaces
        // when rotating the artwork
        cv::Mat fillcolor;
        cv::resize(artColor.clone(), fillcolor, newcolorSize);
        cv::Mat inversenewcolormask;
        cv::threshold(newcolor, inversenewcolormask, 0, 255, THRESH_BINARY_INV);
        cv::bitwise_and(inversenewcolormask, fillcolor, inversenewcolormask);
        cv::bitwise_or(fillcolor, inversenewcolormask, newcolor);
    }
    
    cv::Mat finalcolor;
    cv::bitwise_and(newcolor, newcolor, finalcolor, mask_inv);
    cv::Mat nail_rawmaskedinv;
    cv::bitwise_and(nail_raw, nail_raw, nail_rawmaskedinv, nail_rawmask_inv);
//    cv::seamlessClone(nail_raw, finalcolor, nail_rawmaskedinv, point, finalcolor, MONOCHROME_TRANSFER);
    
    //////////
    Mat mixed;
    add(finalcolor, nailmasked, mixed);
    // Apply Antialiasing to eliminate jagged results
    Mat org_mixed = mixed.clone();
    Mat edges;
    cv::Canny(finalcolor, edges, 100, 200);
    cv::Mat kernel = Mat::ones(5, 5, CV_8UC(1));
    cv::Point anchor(-1,-1); // default value
    cv::dilate(edges, edges, kernel, anchor, 3); // dilate edges to make them larger
    cv::Mat inv_edges;
    cv::bitwise_not(edges, inv_edges);
    cv::Size kSize(5, 5);
    cv::GaussianBlur(mixed, mixed, kSize, 0);
    cv::Mat mixed_masked;
    cv::bitwise_and(mixed, mixed, mixed_masked, edges);
    cv::Mat mixed_masked_inv;
    cv::bitwise_and(org_mixed, org_mixed, mixed_masked_inv, inv_edges);
    cv::add(mixed_masked, mixed_masked_inv, mixed);

    // copy illum info
    if(round(nail_raw.cols/2)+xminoff >=nail_raw.cols || round(nail_raw.rows/2)+yminoff >=nail_raw.rows){
        cv::Point point(round(nail_raw.cols/2), round(nail_raw.rows/2));
        cv::seamlessClone(nail_raw, mixed, nail_rawmaskedinv, point, mixed, MONOCHROME_TRANSFER);
    } else{
        cv::Point point(round(nail_raw.cols/2)+xminoff, round(nail_raw.rows/2)+yminoff);
        cv::resize(mixed, mixed, nail_raw.size());
        cv::seamlessClone(nail_raw, mixed, nail_rawmaskedinv, point, mixed, MONOCHROME_TRANSFER);
    }
    
    //////////
    // mixed.copyTo(image.submat(ymin, ymax, xmin, xmax));
    cv::Size mixed_size = mixed.size();
    // Warning: round error may cause the copyTo function to not work!!
    cv::Size fullmixed((int)(round(xmax*xfactor) - round(xmin*xfactor)), (int)(round(ymax*yfactor) - round(ymin*yfactor)));
    cv::resize(mixed, mixed, fullmixed);
    org_mixed = mixed.clone();
    
    cv::resize(edges, edges, fullmixed);
    cv::threshold(edges, edges, 0, 255, THRESH_BINARY);
    cv::bitwise_not(edges, inv_edges);
    
    cv::GaussianBlur(mixed, mixed, kSize, 0);
    cv::bitwise_and(mixed, mixed, mixed_masked, edges);
    cv::bitwise_and(org_mixed, org_mixed, mixed_masked_inv, inv_edges);
    cv::add(mixed_masked, mixed_masked_inv, mixed);
    // sharpen edges to enhance results
    cv::Mat mixed_copy = mixed.clone();
    //Mat mixed_blur = new Mat();
    // option 1 use unsharp masking
    cv::Mat mixed_blur;
    cv::Size size(0, 0);
    cv::GaussianBlur(mixed_copy, mixed_blur, size, 3);
    cv::Mat mixed_final;
    cv::addWeighted(mixed, 1.5, mixed_blur, -0.5, 0, mixed_final);
    
    cv::Rect ROIfinal((int) round(xmin*xfactor),
                      (int) round(ymin*yfactor),
                      (int) mixed_final.cols,
                      (int) mixed_final.rows);
    // copy result to original image with full resolution
    
    cv::Mat nmask_inv;
    cv::Size mask_finalSize = mixed_final.size();
    cv::resize(mask_inv, nmask_inv, mask_finalSize);
    
    
    //cv::resize(maskPart, maskPart, mask_finalSize);
    
//     dilate final mask to avoid new jaggedness
//    cv::Mat nkernel;
//    nkernel = cv::Mat::ones(7, 7, CV_8UC(1));
//    cv::Point nanchor;
//    nanchor = cv::Point(-1,-1); // default value
//    dilate(nmask_inv, nmask_inv, nkernel, nanchor, 2);
//
    
//    cv::Mat mixed_final_final;
    //cv::bitwise_and(nmask_inv, nmask_inv, nmask_inv, maskPart);
    
    mixed_final.copyTo(imageOrg(ROIfinal), nmask_inv);
    // copy result to image so result in current nail can't be cropped by another nail's result
    resize(mixed_final, mixed_final, mixed_size);
}

NSMutableArray *_keyPoints2MutableArray (NSArray *nks, float maxside){
    float x1 = [[nks objectAtIndex:0] floatValue] * maxside;
    float y1 = [[nks objectAtIndex:1] floatValue] * maxside;
    float x2 = [[nks objectAtIndex:2] floatValue] * maxside;
    float y2 = [[nks objectAtIndex:3] floatValue] * maxside;
    float x3 = [[nks objectAtIndex:4] floatValue] * maxside;
    float y3 = [[nks objectAtIndex:5] floatValue] * maxside;
    float x4 = [[nks objectAtIndex:6] floatValue] * maxside;
    float y4 = [[nks objectAtIndex:7] floatValue] * maxside;
    float x5 = [[nks objectAtIndex:8] floatValue] * maxside;
    float y5 = [[nks objectAtIndex:9] floatValue] * maxside;
    float x6 = [[nks objectAtIndex:10] floatValue] * maxside;
    float y6 = [[nks objectAtIndex:11] floatValue] * maxside;
    float x7 = [[nks objectAtIndex:12] floatValue] * maxside;
    float y7 = [[nks objectAtIndex:13] floatValue] * maxside;
    float x8 = [[nks objectAtIndex:14] floatValue] * maxside;
    float y8 = [[nks objectAtIndex:15] floatValue] * maxside;
    
    NSMutableArray *keyCGPoints = [NSMutableArray array];
    [keyCGPoints addObject:[NSValue valueWithCGPoint:CGPointMake( x1, y1)]];
    [keyCGPoints addObject:[NSValue valueWithCGPoint:CGPointMake( x2, y2)]];
    [keyCGPoints addObject:[NSValue valueWithCGPoint:CGPointMake( x3, y3)]];
    [keyCGPoints addObject:[NSValue valueWithCGPoint:CGPointMake( x4, y4)]];
    [keyCGPoints addObject:[NSValue valueWithCGPoint:CGPointMake( x5, y5)]];
    [keyCGPoints addObject:[NSValue valueWithCGPoint:CGPointMake( x6, y6)]];
    [keyCGPoints addObject:[NSValue valueWithCGPoint:CGPointMake( x7, y7)]];
    [keyCGPoints addObject:[NSValue valueWithCGPoint:CGPointMake( x8, y8)]];
    return keyCGPoints;
}


+ (Mat)_createMask: (NSArray *) nks andImage: (Mat) imageOrg andXmin: (NSInteger) xmin andYmin: (NSInteger) ymin andXmax: (NSInteger) xmax andYmax: (NSInteger) ymax andAnchor: (int) panchor andDilate: (int) pdilate andKernel: (int) pkernel andUseDelate: (bool) useDelate andXFac: (double) XFactor andYFac: (double) YFactor {
    
    float oh = ymax - ymin;
    float ow = xmax - xmin;
    
    float maxside = (oh > ow ? oh : ow)*2;
    
    float sh = oh/maxside;
    float sw = ow/maxside;
    
    NSMutableArray *keyCGPoints = _keyPoints2MutableArray(nks, maxside);
    
    NSMutableArray *maskPoints = [NSMutableArray array];
    for (int z = 0; z < keyCGPoints.count; z++) {
        NSMutableArray *maskPnt = [ImageProcessing _getSplines:keyCGPoints andSide: (z+1) andH: maxside andW: maxside];
        [maskPoints addObjectsFromArray:maskPnt];
    }
    
    maskPoints = deleteDuplicatedXY(maskPoints);
    
    // create mask by joining points
    cv::Mat mask = cv::Mat::zeros(oh, ow, CV_8UC(1));
    //cv::Mat mask(cv::Size(oh, ow),CV_8UC1);
    Scalar color = cv::Scalar(255);
    
    // Convert Muttable Array to vector
    long masksize = maskPoints.count;
    std::vector<cv::Point2i> pts2;
    for (int z = 0; z < masksize; z++) {
        CGPoint chpnt_ = [maskPoints[z] CGPointValue];
        pts2.push_back(cv::Point2f(chpnt_.x * sw, chpnt_.y * sh));
        //cv::circle(imageOrg, cv::Point(round(XFactor * xmin) + (chpnt_.x), round(YFactor * ymin) + (chpnt_.y)), 2, cv::Scalar(234));
    }
    
    vector<vector<cv::Point2i>> contours;
    mask = 0;
    contours.push_back(pts2);
    drawContours(mask,contours,0,color,-1);
    //cv::fillConvexPoly(mask, pts2, color, LINE_4);
    //cv::fillPoly(mask, pts2, color)/;
    // dilate final mask
    //cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC(1));
    if (useDelate){
        cv::Mat kernel = cv::Mat::ones(pkernel, pkernel, CV_8UC(1));
        cv::Point anchor = cv::Point(panchor, panchor); // default value
        cv::dilate(mask, mask, kernel, anchor, pdilate);

    }
    return mask;
}


+ (Mat)_createMask2: (NSArray *) nks andXmin: (NSInteger) xmin andYmin: (NSInteger) ymin andXmax: (NSInteger) xmax andYmax: (NSInteger) ymax {
    
    float oh = ymax - ymin;
    float ow = xmax - xmin;
    
    float maxside = (oh > ow ? oh : ow)*2;
    
    float sh = oh/maxside;
    float sw = ow/maxside;
    
    NSMutableArray *keyCGPoints = _keyPoints2MutableArray(nks, maxside);
    
    NSMutableArray *maskPoints = [NSMutableArray array];
    for (int z = 0; z < keyCGPoints.count; z++) {
        NSMutableArray *maskPnt = [ImageProcessing _getSplines:keyCGPoints andSide: (z+1) andH: maxside andW: maxside];
        [maskPoints addObjectsFromArray:maskPnt];
    }
    
    maskPoints = deleteDuplicatedXY(maskPoints);
    
    // create mask by joining points
    cv::Mat mask = cv::Mat::zeros(oh, ow, CV_8UC(1));
    Scalar color = cv::Scalar(255);
    
    // Convert Muttable Array to vector
    long masksize = maskPoints.count;
    std::vector<cv::Point2i> pts2;
    for (int z = 0; z < masksize; z++) {
        CGPoint chpnt_ = [maskPoints[z] CGPointValue];
        pts2.push_back(cv::Point2f(chpnt_.x * sw, chpnt_.y * sh));
    }
    
    cv::fillConvexPoly(mask, pts2, color, LINE_4);
    // dilate final mask
    cv::Mat kernel = cv::Mat::ones(2, 2, CV_8UC(1));
    cv::Point anchor = cv::Point(-1, -1); // default value
    cv::dilate(mask, mask, kernel, anchor, 1);
    return mask;
}

/**
 * determine if it is necessary to rotate points to avoid overshooting during interpolation
 */
+ (bool)_shouldRotate: (CGPoint) point1 andPoint: (CGPoint) point2 {
    int dx = point1.x - point2.x;
    int dy = point1.y - point2.y;

    if(dy!=0){
        float slope = ((float)dx)/((float)dy);
        return abs(slope) < 1.0;
    } else{
        return false;
    }
}

/////////////////////////////////////////////////////////////////////
// Methods to process predicted coordinates to generate final mask //
/////////////////////////////////////////////////////////////////////

/**
 * Get extra x values to use for interpolation within the interval [x1, x2]
 */
NSMutableArray * extraX (int x1, int x2, int numExtraX) {
    int interval, intervals;
    float dx;

    if(x2 > x1){
        interval = x2 - x1;
    } else{
        interval = x1 - x2;
    }

    // Avoid getting repeated values
    /*if(numExtraX > interval){
        numExtraX = Math.round(interval);
    }*/

    intervals = numExtraX + 1;
    dx = ((float) interval) / ((float) intervals);
    NSMutableArray *x_values = [NSMutableArray array];

    for(int i=0; i<intervals+1; i++){
        if(x2 > x1){
            int newValue = round(x1 + dx*i);
            if(newValue > x2) {
                [x_values addObject: [NSNumber numberWithInt:x2]];
            } else {
                [x_values addObject: [NSNumber numberWithInt:newValue]];
            }

        } else{
            int newValue = round(x1 - dx*i);

            if(newValue > x1){
                [x_values addObject: [NSNumber numberWithInt:x1]];
            } else {
                [x_values addObject: [NSNumber numberWithInt:newValue]];
            }
        }
    }
    return x_values;
}

/**
 * Sort by using x values
 */
NSMutableArray * sortbyX (NSMutableArray *coords) {
    [coords sortUsingComparator:^(id obj1, id obj2) {
        CGPoint firstPoint = [obj1 CGPointValue];
        CGPoint secondPoint = [obj2 CGPointValue];
        if ( firstPoint.x < secondPoint.x) {
            return (NSComparisonResult)NSOrderedAscending;
        } else if ( firstPoint.x > secondPoint.x) {
            return (NSComparisonResult)NSOrderedDescending;
        }
        return (NSComparisonResult)NSOrderedSame;
    }];
    return coords;
}

/**
 * Delete duplicated pairs of coordinates (x, y) without sorting.
 * If a pair of coordinates is repeated then the first pair is deleted.
 * e.g: deleteDuplicatedXY({{9,6,3,6,1,6,7,7,1}, {1,2,4,2,5,2,7,7,4}})
 * gives {{9,3,1,6,7,1},{1,4,5,2,7,4}}
 */
NSMutableArray * deleteDuplicatedXY (NSMutableArray *coords) {
    NSMutableArray *coordsnew = [NSMutableArray array];
    
    for (int i = 0; i < coords.count; i++) {
        bool Exist = false;
        CGPoint pnt1 = [coords[i] CGPointValue];
        for (int z = i+1; z < coords.count; z++) {
            CGPoint pnt2 = [coords[z] CGPointValue];
            if ( pnt2.x == pnt1.x && pnt2.y == pnt1.y ){
                Exist = true;
            }
        }
        if (!Exist){
            [coordsnew addObject:[NSValue valueWithCGPoint:pnt1]];
        }
    }
    return coordsnew;
}

CGPoint rotate90Clockwise(CGPoint point){
    return CGPointMake(point.y, -1 *point.x);
}

CGPoint rotate90CounterClockwise(CGPoint point){
    return CGPointMake(-1 *point.y, point.x);
}

NSMutableArray * rotateList90Clockwise(NSMutableArray * coords){
    NSMutableArray * outcoords = [NSMutableArray array];
    for (NSValue * coord in coords) {
        CGPoint newcord = rotate90Clockwise([coord CGPointValue]);
        
        [outcoords addObject:[NSValue valueWithCGPoint:newcord]];
    }
    return outcoords;
}

NSMutableArray * cubicSpline(NSMutableArray * coords, NSMutableArray * xcoords){
    SAMCubicSpline *spline = [[SAMCubicSpline alloc] initWithPoints:coords];

    // interpolate just one value at each time ( the value must lie inside the interval of x)
    NSMutableArray *newcoords = [NSMutableArray array];

    for (NSInteger i = 0; i < [xcoords count]; i++) {
        CGFloat x = [[xcoords objectAtIndex:i] floatValue];
        CGFloat y = [spline interpolate:x];
        [newcoords addObject:[NSValue valueWithCGPoint:CGPointMake(x,y)]];
    }

    return newcoords;
}

NSMutableArray * deletePointsFromArray(NSMutableArray * newcoords, int x1, int y1, int x2, int y2){
    NSMutableArray * results = [NSMutableArray array];
    long size = newcoords.count;

    for (int i=0; i<size; i++){
        if ((([newcoords[i] CGPointValue].x==x1) && ([newcoords[i] CGPointValue].y==y1)) || (([newcoords[i] CGPointValue].x==x2) && ([newcoords[i] CGPointValue].y==y2))){
            continue;
        } else {
            [results addObject:(newcoords[i])];
        }
    }
    return results;
}

NSMutableArray * rotateList90CounterClockwise(NSMutableArray * pointCoord){
    NSMutableArray * result = [NSMutableArray array];

    for(NSValue * pnt in pointCoord){
        CGPoint point= [pnt CGPointValue];
        CGPoint newpoint = rotate90CounterClockwise(point);
        [result addObject:[NSValue valueWithCGPoint:newpoint]];
    }
    return result;
}

void debugData (NSMutableArray * newcoords){
    long masksize = newcoords.count;
    CGPoint pts2[masksize];
    for (int z = 0; z < masksize; z++) {
        CGPoint chpnt_ = [newcoords[z] CGPointValue];
        pts2[z] = CGPointMake(chpnt_.x, chpnt_.y);
    }
    
}



+ (NSMutableArray *)_getSplines: (NSMutableArray *) keypoints andSide: (int) side andH: (float) h andW: (float) w {
    
    CGPoint point1 = [keypoints[0] CGPointValue];
    CGPoint point2 = [keypoints[1] CGPointValue];
    CGPoint point3 = [keypoints[2] CGPointValue];
    CGPoint point4 = [keypoints[3] CGPointValue];
    CGPoint point5 = [keypoints[4] CGPointValue];
    CGPoint point6 = [keypoints[5] CGPointValue];
    CGPoint point7 = [keypoints[6] CGPointValue];
    CGPoint point8 = [keypoints[7] CGPointValue];

    NSMutableArray* coords = [NSMutableArray array];
    
    float x1, x2;
    float y1, y2;

    if(side == 1){
        x1 = point1.x;
        x2 = point2.x;
        y1 = point1.y;
        y2 = point2.y;
        [coords addObject:[NSValue valueWithCGPoint:point1]];
        [coords addObject:[NSValue valueWithCGPoint:point2]];
        [coords addObject:[NSValue valueWithCGPoint:point3]];
    } else if(side == 2){
        x1 = point2.x;
        x2 = point3.x;
        y1 = point2.y;
        y2 = point4.y;
        [coords addObject:[NSValue valueWithCGPoint:point1]];
        [coords addObject:[NSValue valueWithCGPoint:point2]];
        [coords addObject:[NSValue valueWithCGPoint:point3]];
    } else if(side == 3){
        x1 = point3.x;
        x2 = point4.x;
        y1 = point3.y;
        y2 = point4.y;
        [coords addObject:[NSValue valueWithCGPoint:point3]];
        [coords addObject:[NSValue valueWithCGPoint:point4]];
        [coords addObject:[NSValue valueWithCGPoint:point5]];
    } else if(side == 4){
        x1 = point4.x;
        x2 = point5.x;
        y1 = point4.y;
        y2 = point5.y;
        [coords addObject:[NSValue valueWithCGPoint:point3]];
        [coords addObject:[NSValue valueWithCGPoint:point4]];
        [coords addObject:[NSValue valueWithCGPoint:point5]];
    } else if(side == 5){
        x1 = point5.x;
        x2 = point6.x;
        y1 = point5.y;
        y2 = point6.y;
        [coords addObject:[NSValue valueWithCGPoint:point5]];
        [coords addObject:[NSValue valueWithCGPoint:point6]];
        [coords addObject:[NSValue valueWithCGPoint:point7]];
    } else if(side == 6){
        x1 = point6.x;
        x2 = point7.x;
        y1 = point6.y;
        y2 = point7.y;
        [coords addObject:[NSValue valueWithCGPoint:point5]];
        [coords addObject:[NSValue valueWithCGPoint:point6]];
        [coords addObject:[NSValue valueWithCGPoint:point7]];
    } else if(side == 7){
        x1 = point7.x;
        x2 = point8.x;
        y1 = point7.y;
        y2 = point8.y;
        [coords addObject:[NSValue valueWithCGPoint:point7]];
        [coords addObject:[NSValue valueWithCGPoint:point8]];
        [coords addObject:[NSValue valueWithCGPoint:point1]];
    } else if(side == 8){
        x1 = point8.x;
        x2 = point1.x;
        y1 = point8.y;
        y2 = point1.y;
        [coords addObject:[NSValue valueWithCGPoint:point7]];
        [coords addObject:[NSValue valueWithCGPoint:point8]];
        [coords addObject:[NSValue valueWithCGPoint:point1]];
    } else{
        x1 = point1.x;
        x2 = point2.x;
        y1 = point1.y;
        y2 = point2.y;
        [coords addObject:[NSValue valueWithCGPoint:point1]];
        [coords addObject:[NSValue valueWithCGPoint:point2]];
        [coords addObject:[NSValue valueWithCGPoint:point3]];
    }

    // check if points should be rotated to avoid overshooting
    // option 1: use only points on the side
    // boolean rotate = shouldRotate(new int[]{x1, y1}, new int[]{x2, y2});
    // option 2: use end points

    bool rotate = [ImageProcessing _shouldRotate: [coords[0] CGPointValue] andPoint: [coords[2] CGPointValue]];

    if (rotate) {
        log(rotate);
        log(side);
    }
    
    NSMutableArray* xcoords = [NSMutableArray array];
    xcoords = extraX(x1, x2, w);

    /////////////////////////////////////////////////////////////////
    /* PREPROCESS COORDS: splines need strictly increasing x values*/
    /////////////////////////////////////////////////////////////////

    // sort by using x values
    coords = sortbyX(coords);
    
    // delete duplicated pairs of coordinates (x, y)
    coords = deleteDuplicatedXY(coords);

    if(rotate){
        if(side==1 || side==3 || side==5 || side==7){
            coords = sortbyX(coords);
            coords = rotateList90Clockwise(coords);
            x1 = [coords[0] CGPointValue].x;
            x2 = [coords[1] CGPointValue].x;
            y1 = [coords[0] CGPointValue].y;
            y2 = [coords[1] CGPointValue].y;
            xcoords = extraX(x1, x2, h);
        }  else{
            coords = sortbyX(coords);
            coords = rotateList90Clockwise(coords);
            x1 = [coords[1] CGPointValue].x;
            x2 = [coords[2] CGPointValue].x;
            y1 = [coords[1] CGPointValue].y;
            y2 = [coords[2] CGPointValue].y;
            xcoords = extraX(x1, x2, h);
        }
//         get new dx values (corresponding to original y ranges)
    }

    // check there are not repeated x values
    // sort by using x values
    coords = sortbyX(coords);
//
//    // Interpolation //
//    //find cubic splines
    NSMutableArray* newcoords = cubicSpline(coords, xcoords);

    // delete original points to get a smooth mask without pointed corners
    newcoords = deletePointsFromArray(newcoords, x1, y1, x2, y2);
    //debugData(newcoords);
    if(rotate){
        newcoords = rotateList90CounterClockwise(newcoords);
    }
    //debugData(newcoords);

    return newcoords;
}

@end
