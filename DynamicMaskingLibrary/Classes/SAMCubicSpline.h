//
//  SAMCubicSpline.h
//  ObjectDetection
//
//  Created by A. Ichwan Yasir on 13/10/21.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//
#import <Foundation/Foundation.h>
#if TARGET_OS_IOS
#import <UIKit/UIKit.h>
#endif

@interface SAMCubicSpline : NSObject

/**
 Initialize a new cubic spline.

 @param points An array of `NSValue` objects containing `CGPoint` structs. These points are the control points of the
 curve.

 @return A new cubic spline.
 */
- (instancetype)initWithPoints:(NSArray *)points;

/**
 Input an X value between 0 and 1.

 @return The corresponding Y value.
 */
- (CGFloat)interpolate:(CGFloat)x;

@end
