//
//  models.m
//  ObjectDetection
//
//  Created by A. Ichwan Yasir on 27/07/21.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "models.h"

@implementation ModelClass
-(id)init {
    self = [super init];
    boundingBox = [[NSMutableArray alloc] init];
    scores = [[NSMutableArray alloc] init];
    labels = [[NSMutableArray alloc] init];
    angels = [[NSMutableArray alloc] init];
    point2coords = [[NSMutableArray alloc] init];
    nailkeypoints = [[NSMutableArray alloc] init];
    useArt = false;
    
    return self;
}

-(void) addNail:(NSArray *) bbox andScores:(NSNumber *) score andLabel: (NSNumber *) label {
    [boundingBox addObject:bbox];
    [scores addObject:score];
    [labels addObject:label];
}

-(void) addKeyPoint:(NSArray *) point {
    [keypoints addObject:point];
}

-(void) set_boundingBox:(NSMutableArray *) bbox {
    boundingBox = bbox;
}
-(void) set_keypoints:(NSMutableArray *) keypoint{
    keypoints = keypoint;
}
-(void) set_scores:(NSMutableArray *) score {
    scores = score;
}
-(void) set_labels:(NSMutableArray *) label {
    labels = label;
}
-(void) set_angels:(NSMutableArray *) angel {
    angels = angel;
}
-(void) set_point2coords:(NSMutableArray *) point2coord {
    point2coords = point2coord;
}

-(void) set_nailkeypoints:(NSMutableArray *) nailkeypoint {
    nailkeypoints = nailkeypoint;
}

-(void) set_artcolor:(UIImage *) artcolor {
    artColor = artcolor;
    useArt = true;
}


-(NSMutableArray *) get_boundingBox {
    return boundingBox;
}
-(NSMutableArray *) get_scores {
    return scores;
}
-(NSMutableArray *) get_labels {
    return labels;
}
-(NSMutableArray *) get_keypoints {
    return keypoints;
}
-(NSMutableArray *) get_point2coords {
    return point2coords;
}
-(NSMutableArray *) get_angels {
    return angels;
}
-(NSMutableArray *) get_nailkeypoints {
    return nailkeypoints;
}
-(UIImage *) get_artcolor {
    return artColor;
}
-(bool) get_useart {
    return useArt;
}

@end
