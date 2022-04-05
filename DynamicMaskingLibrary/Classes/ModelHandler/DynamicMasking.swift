//
//  DynamicMasking.swift
//  Dynamic Masking
//
//  Created by Oscar Montes Camberos on 1/31/22.
//  Copyright © 2022 Y Media Labs. All rights reserved.
//

import UIKit

public class DynamicMasking {
    
    private var modelSSDHandler: SSDDataHandler?
    private var modelDCPHandler: DCPModelDataHandler?
    
    // Holds the results at any time
    private var resultSSD: ResultSSD?
    private var resultDCP: DCPInterference?
    
    //Timings
    private var ssdTime = 0.0
    private var landMarkTime = 0.0
    private var coloringTime = 0.0
    private var inferenceTime = 0.0
    
    public init() {
        modelSSDHandler = SSDDataHandler(modelFileInfo: MobilwTflite.modelInfo, labelsFileInfo: MobilwTflite.labelsInfo, threadCount: 4)
        modelDCPHandler = DCPModelDataHandler(modelFileInfo: MobilwTflite.modelDCPInfo, threadCount: 4)
        
        guard modelSSDHandler != nil else {
          fatalError("Failed to load model")
        }
    }
    
    ///Get predictRunProcess timings tuples
    public func getTimings() -> (ssdTime: Double, landMarkTime: Double, coloringTime: Double, inferenceTime: Double) {
        return (ssdTime, landMarkTime, coloringTime, inferenceTime)
    }
    
    ///Paint image with selected parameters
    public func predictRunProcess(handImage: UIImage?, threadCount: Int, color: String, artColor: UIImage, artMode: Bool) -> UIImage? {
        guard let pixelBuffer = handImage?.pixelBuffer() else {
            fatalError()
        }
        
        modelSSDHandler = SSDDataHandler(modelFileInfo: MobilwTflite.modelInfo, labelsFileInfo: MobilwTflite.labelsInfo, threadCount: threadCount)
        
        modelDCPHandler = DCPModelDataHandler(modelFileInfo: MobilwTflite.modelDCPInfo, threadCount: threadCount)
        
        let orgsizeArr: [Float] = [Float((handImage?.size.width)!), Float((handImage?.size.height)!)];
        
        resultSSD = self.modelSSDHandler?.runModel(onFrame: pixelBuffer)
        
        
        var startDate = Date()
        for bbox in (resultSSD!.inferences.boundingboxs as NSArray as! [NSArray]) {
            // [ymin, xmin, ymax, xmax]
            let y = bbox.object(at: 0) as! Float * orgsizeArr[1]
            let x = bbox.object(at: 1) as! Float * orgsizeArr[0]
            let height_ = (bbox.object(at: 2) as! Float * orgsizeArr[1]) - y
            let width_ = (bbox.object(at: 3) as! Float * orgsizeArr[0]) - x
            
            
            let Rect = CGRect(x: CGFloat(x), y: CGFloat(y), width: CGFloat(width_), height: CGFloat(height_))
            
            let pix = handImage?.crop(rect: Rect)
            
            guard let pixbuffer = pix!.pixelBuffer() else {
                fatalError()
            }
            
            resultDCP = self.modelDCPHandler?.runModel(onFrame: pixbuffer)
            resultSSD?.inferences.resultDCP.add(resultDCP!.keypoints)
            
        }
        let intervalDCP = Date().timeIntervalSince(startDate) * 1000
        
        guard let tempSSDResult = resultSSD else {
          return nil
        }
        
        let resolution: Int = 2048;
        
        let nailsize: String = "medium";
        
        let orgsize = UnsafeMutablePointer<Float>.allocate(capacity: orgsizeArr.count)
        orgsize.initialize(from: orgsizeArr, count: orgsizeArr.count)
        
        
        let mc: ModelClass = ModelClass();
        mc.set_boundingBox(tempSSDResult.inferences.boundingboxs)
        mc.set_labels(tempSSDResult.inferences.classes)
        mc.set_scores(tempSSDResult.inferences.scores)
        mc.set_nailkeypoints(tempSSDResult.inferences.resultDCP)
        if (artMode) {
            mc.set_artcolor(artColor)
        }
        startDate = Date()
        
        let resultImage: UIImage = ImageProcessing.paintNailDynamic(handImage!, andMC: mc, andNailSize: nailsize, andOrgSize: orgsize, andResolution: Int32(resolution), andColor: color)
        
        let intervalColoring = Date().timeIntervalSince(startDate) * 1000
        
        ssdTime = tempSSDResult.inferenceTime
        landMarkTime = intervalDCP
        coloringTime = intervalColoring
        inferenceTime = tempSSDResult.inferenceTime + intervalDCP + intervalColoring
        
        return resultImage
    }
}
