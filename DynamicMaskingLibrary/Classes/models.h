//
//  model.h
//  ObjectDetection
//
//  Created by A. Ichwan Yasir on 27/07/21.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//

#ifndef models_h
#define models_h
#import <UIKit/UIKit.h>

#endif /* models_h */
@interface ModelClass:NSObject {
    // All Params to made painte image
    NSMutableArray *boundingBox;
    NSMutableArray *keypoints;
    NSMutableArray *scores;
    NSMutableArray *labels;
    NSMutableArray *angels;
    NSMutableArray *point2coords;
    NSMutableArray *nailkeypoints;
    UIImage *artColor;
    bool useArt;
    
}


-(void) addNail:(NSArray *) bbox andScores:(NSNumber *) score andLabel: (NSNumber *) label;
-(void) addKeyPoint:(NSArray *) point;

-(void) set_boundingBox: (NSMutableArray *) boundingBox;
-(void) set_keypoints: (NSMutableArray *) keypoints;
-(void) set_scores: (NSMutableArray *) scores;
-(void) set_labels: (NSMutableArray *) labels;
-(void) set_angels: (NSMutableArray *) angels;
-(void) set_point2coords: (NSMutableArray *) point2coords;
-(void) set_nailkeypoints: (NSMutableArray *) point2coords;
-(void) set_artcolor: (UIImage *) artcolor;

-(NSMutableArray *) get_boundingBox;
-(NSMutableArray *) get_keypoints;
-(NSMutableArray *) get_scores;
-(NSMutableArray *) get_labels;
-(NSMutableArray *) get_angels;
-(NSMutableArray *) get_point2coords;
-(NSMutableArray *) get_nailkeypoints;
-(UIImage *) get_artcolor;
-(bool) get_useart;

@end
