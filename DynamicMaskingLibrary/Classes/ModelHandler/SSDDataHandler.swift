//
//  SSDDataHandler.swift
//  ObjectDetection
//
//  Created by A. Ichwan Yasir on 08/07/21.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//

import Foundation

import CoreImage
import TensorFlowLite
import UIKit
import Accelerate

class SSDDataHandler: NSObject {
    // MARK: - Internal Properties
    /// The current thread count used by the TensorFlow Lite Interpreter.
    let threadCount: Int
    let threadCountLimit = 10

    let threshold: Float = 0.5

    // MARK: Model parameters
    let batchSize = 1
    let inputChannels = 3
    let inputWidth = 640
    let inputHeight = 640
//    let delegates: [Delegate]

    // image mean and std for floating model, should be consistent with parameters used in model training
    let imageMean: Float = 127.5
    let imageStd:  Float = 127.5

    // MARK: Private properties
    private var labels: [String] = ["???", "nail"]

    /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
    private var interpreter: Interpreter

    private let bgraPixel = (channels: 4, alphaComponent: 3, lastBgrComponent: 2)
    private let rgbPixelChannels = 3
    private let colorStrideValue = 10
    private let colors = [
      UIColor.red
    ]
    
    // MARK: - Initialization

    /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
    /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
    init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, delegates: [Delegate], threadCount: Int = 1) {
      let modelFilename = modelFileInfo.name
      let modelExt = modelFileInfo.extension
        
      
        guard let modelPath = ModelBundleClass.resourceBundle.path(
          forResource: modelFilename,
          ofType: modelExt
        ) else {
            print("Failed to load the model file with name: \(modelFilename).")
            return nil
          }
      // Construct the path to the nail model file.
//      guard let modelPath = Bundle.main.path(
//        forResource: modelFilename,
//        ofType: modelExt
//      ) else {
//        print("Failed to load the model file with name: \(modelFilename).")
//        return nil
//      }
//      var optionsDel = CoreMLDelegate.Options()
//      optionsDel.enabledDevices = .all
//      let coreMLDelegate = CoreMLDelegate(options: optionsDel)!
//
//      // Specify the options for the `Interpreter`.
//      self.delegates = [coreMLDelegate]
      self.threadCount = threadCount
      var options = Interpreter.Options()
      options.threadCount = threadCount
      do {
        // Create the `Interpreter`.
        interpreter = try Interpreter(modelPath: modelPath, options: options, delegates: delegates)
        
        // Allocate memory for the model's input `Tensor`s.
        try interpreter.allocateTensors()
      } catch let error {
        print("Failed to create the interpreter with error: \(error.localizedDescription)")
        return nil
      }

      super.init()

      // Load the classes listed in the labels file.
      // loadLabels(fileInfo: labelsFileInfo)
    }
    
    
    /// This class handles all data preprocessing and makes calls to run inference on a given frame
    /// through the `Interpreter`. It then formats the inferences obtained and returns the top N
    /// results for a successful inference.
    func runModel(onFrame pixelBuffer: CVPixelBuffer) -> ResultSSD? {
      let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
      let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
      let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
      assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
               sourcePixelFormat == kCVPixelFormatType_32BGRA ||
                 sourcePixelFormat == kCVPixelFormatType_32RGBA)


      let imageChannels = 4
      assert(imageChannels >= inputChannels)

      // Crops the image to the biggest square in the center and scales it down to model dimensions.
      let scaledSize = CGSize(width: inputWidth, height: inputHeight)
      guard let scaledPixelBuffer = pixelBuffer.resized(to: scaledSize) else {
        return nil
      }

      let interval: TimeInterval
      let outputBoundingBox: Tensor
      let outputClasses: Tensor
      let outputScores: Tensor
      let outputCount: Tensor
      do {
        let inputTensor = try interpreter.input(at: 0)
        
        // Remove the alpha component from the image buffer to get the RGB data.
        guard let rgbData = modelHelper.rgbDataFromBuffer(
          scaledPixelBuffer,
          byteCount: batchSize * inputWidth * inputHeight * inputChannels,
          isModelQuantized: inputTensor.dataType == .uInt8
        ) else {
          print("Failed to convert the image buffer to RGB data.")
          return nil
        }
        

        // Copy the RGB data to the input `Tensor`.
        try interpreter.copy(rgbData, toInputAt: 0)
        
        // Run inference by invoking the `Interpreter`.
        let startDate = Date()
        try interpreter.invoke()
        interval = Date().timeIntervalSince(startDate) * 1000

        outputBoundingBox = try interpreter.output(at: 0)
        outputClasses = try interpreter.output(at: 1)
        outputScores = try interpreter.output(at: 2)
        outputCount = try interpreter.output(at: 3)
          
      } catch let error {
        print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
        return nil
      }

      // Formats the results
      let resultArray = formatResults(
        boundingBox: [Float](unsafeData: outputBoundingBox.data) ?? [],
        outputClasses: [Float](unsafeData: outputClasses.data) ?? [],
        outputScores: [Float](unsafeData: outputScores.data) ?? [],
        outputCount: Int(([Float](unsafeData: outputCount.data) ?? [0])[0]),
        width: CGFloat(imageWidth),
        height: CGFloat(imageHeight),
        inputW: CGFloat(inputWidth),
        inputH: CGFloat(inputHeight)
      )
      let result = ResultSSD(inferenceTime: interval, inferences: resultArray)
      return result
    }
    
    /// Filters out all the results with confidence score < threshold and returns the top N results
    /// sorted in descending order.
    func formatResults(boundingBox: [Float], outputClasses: [Float], outputScores: [Float], outputCount: Int, width: CGFloat, height: CGFloat, inputW: CGFloat, inputH: CGFloat) -> SSDInterference {
        let resultInterference: SSDInterference = SSDInterference(scores: [], classes: [], boundingboxs: [], resultDCP: []);
      if (outputCount == 0) {
        return resultInterference
      }
      for i in 0...outputCount - 1 {

        let score = outputScores[i]

        // Filters results with confidence < threshold.
        guard score >= threshold else {
          continue
        }
        
        // Gets the output class names for detected classes from labels list.
        let outputClassIndex = Int(outputClasses[i])

        // Position [ymin, xmin, ymax, xmax]
        let bBox: NSArray = [boundingBox[4*i], boundingBox[4*i+1], boundingBox[4*i+2], boundingBox[4*i+3]]

        resultInterference.boundingboxs.add(bBox)
        resultInterference.classes.add(outputClassIndex)
        resultInterference.scores.add(score)

      }

      return resultInterference
    }
    
    /// Loads the labels from the labels file and stores them in the `labels` property.
    private func loadLabels(fileInfo: FileInfo) {
      let filename = fileInfo.name
      let fileExtension = fileInfo.extension
        
      guard let fileURL = ModelBundleClass.resourceBundle.url(
          forResource: filename,
          withExtension: fileExtension
      )
      else {
          fatalError("Labels file not found in bundle. Please add a labels file with name " +
                       "\(filename).\(fileExtension) and try again.")
      }
      do {
        let contents = try String(contentsOf: fileURL, encoding: .utf8)
        labels = contents.components(separatedBy: .newlines)
      } catch {
        fatalError("Labels file named \(filename).\(fileExtension) cannot be read. Please add a " +
                     "valid labels file and try again.")
      }
    }
    
    /// This assigns color for a particular class.
    private func colorForClass(withIndex index: Int) -> UIColor {

      // We have a set of colors and the depending upon a stride, it assigns variations to of the base
      // colors to each object based on its index.
      let baseColor = colors[index % colors.count]

      var colorToAssign = baseColor

      let percentage = CGFloat((colorStrideValue / 2 - index / colors.count) * colorStrideValue)

      if let modifiedColor = baseColor.getModified(byPercentage: percentage) {
        colorToAssign = modifiedColor
      }

      return colorToAssign
    }
}
