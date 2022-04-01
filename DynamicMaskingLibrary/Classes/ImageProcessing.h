//
//  OpenCVWrapper.h
//  ObjectDetection
//
//  Created by A. Ichwan Yasir on 28/06/21.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//

#import "ImageProcessing.h"
#import "models.h"
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface ImageProcessing : NSObject
//@property float *angle;
//@property float *coordinate;
+ (UIImage *)toGray:(UIImage *)source;
+ (UIImage *)drawRect:(UIImage *) source andReacts: (CGRect *) rects andPoints: (CGPoint *) points;

+ (UIImage *)paintNailDynamic:(UIImage*) _frame andMC: (ModelClass *) mc andNailSize: (NSString*) nailsize andOrgSize: (float*) orgsize andResolution: (int) resolution andColor: (NSString*) _color ;

//+ (UIImage *)paintNail:(UIImage*) _frame andColor: (NSString*) _color andNailMask: (UIImage*) _nail_mask andRthumbMask: (UIImage*) _rthumb_mask andLthumbMask: (UIImage*) _lthumb_mask andMC: (ModelClass *) mc andisLeftHand: (bool) isLeftHand andNailSize: (NSString*) nailsize andOrgSize: (float*) orgsize andNailPatch: (int) nailpatches andIllumOffset: (int) illum_offset andResolution: (int) resolution andMinScore: (float) minscore andShowBoxs: (bool) showBoxes andHoutLineMask: (UIImage*) _houtlinemask andShowOutline: (bool) showOutline;

@end

NS_ASSUME_NONNULL_END
