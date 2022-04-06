//
//  DCPModelDataHandler.swift
//  ObjectDetection
//
//  Created by A. Ichwan Yasir on 01/10/21.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//

import CoreImage
import TensorFlowLite
import UIKit
import Accelerate


/// A result from invoking the `Interpreter`.
struct DCPResult {
  let inferenceTime: Double
  let inferences: [Int]
}

/// An inference from invoking the `Interpreter`.
/*struct Inference {
  let confidence: Float
  let label: String
}*/

/// Information about a model file or labels file.
typealias DCPFileInfo = (name: String, extension: String)

/// Information about the DCP model.
enum DCP {
  static let modelInfo: DCPFileInfo = (name: "dcp_8-keypoints_1.0.2", extension: "tflite")
}

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`
class DCPModelDataHandler {

  // MARK: - Internal Properties

  /// The current thread count used by the TensorFlow Lite Interpreter.
  let threadCount: Int
  //let resultCount = 3
  //let threadCountLimit = 10

  // MARK: - Model Parameters

  let batchSize = 1
  let inputChannels = 3
  let inputWidth = 150
  let inputHeight = 150

  // MARK: - Private Properties

  /// List of labels from the given labels file.
  //private var labels: [String] = []

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter
  private var delegates: [Delegate]

  /// Information about the alpha component in RGBA data.
  private let alphaComponent = (baseOffset: 4, moduloRemainder: 3)

  // MARK: - Initialization

  /// A failable initializer for `DCPModelDataHandler`. A new instance is created if the model is
  /// successfully loaded from the app's main bundle. Default `threadCount` is 4.
  init?(modelFileInfo: FileInfo, threadCount: Int = 4) {
    let modelFilename = modelFileInfo.name

      guard let modelPath = ModelBundleClass.resourceBundle.path(
        forResource: modelFilename,
        ofType: modelFileInfo.extension
      ) else {
          print("Failed to load the model file with name: \(modelFilename).")
          return nil
        }
    // Construct the path to the model file.
//    guard let modelPath = Bundle.main.path(
//      forResource: modelFilename,
//      ofType: modelFileInfo.extension
//    ) else {
//      print("Failed to load the model file with name: \(modelFilename).")
//      return nil
//    }
    
    var optionsDel = CoreMLDelegate.Options()
    optionsDel.enabledDevices = .all
    let coreMLDelegate = CoreMLDelegate(options: optionsDel)!
      
    // Specify the options for the `Interpreter`.
    self.delegates = [coreMLDelegate]
    self.threadCount = threadCount
    var options = Interpreter.Options()
    options.threadCount = threadCount
    
    do {
      // Create the `Interpreter`.
      interpreter = try Interpreter(modelPath: modelPath, options: options)
      // Allocate memory for the model's input `Tensor`s.
      try interpreter.allocateTensors()
    } catch let error {
      print("Failed to create the interpreter with error: \(error.localizedDescription)")
      return nil
    }
    // Load the classes listed in the labels file.
    //loadLabels(fileInfo: labelsFileInfo)
  }

  // MARK: - Internal Methods

  /// Performs image preprocessing, invokes the `Interpreter`, and processes the inference results.
  func runModel(onFrame pixelBuffer: CVPixelBuffer) -> DCPInterference? {
    
    let startDate = Date()
    
    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
             sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)

    let imageChannels = 4
    assert(imageChannels >= inputChannels)

    // Crops the image to the biggest square in the center and scales it down to model dimensions.
    /*let scaledSize = CGSize(width: inputWidth, height: inputHeight)
    guard let thumbnailPixelBuffer = pixelBuffer.centerThumbnail(ofSize: scaledSize) else {
      return nil
    }*/
    
    // Resize the image down to model dimensions.
    // by using the custom scale function in CVPixelBufferExtension file (not working!)
    /*guard let thumbnailPixelBuffer = pixelBuffer.scale(targetHeight: inputHeight, targetWidth: inputWidth) else {
      return nil
    }*/
    // by using the CoreMLHelpers library
    guard let thumbnailPixelBuffer = pixelBuffer.resized(to: CGSize(width: inputWidth, height: inputHeight)) else {
      return nil
    }
    
    let interval: TimeInterval
    let outputTensor: Tensor
    do {
      let inputTensor = try interpreter.input(at: 0)

      // Remove the alpha component from the image buffer to get the RGB data.
      guard let rgbData = rgbDataFromBuffer(
        thumbnailPixelBuffer,
        byteCount: batchSize * inputWidth * inputHeight * inputChannels,
        isModelQuantized: inputTensor.dataType == .uInt8
      ) else {
        print("Failed to convert the image buffer to RGB data.")
        return nil
      }
        
      // Copy the RGB data to the input `Tensor`.
      try interpreter.copy(rgbData, toInputAt: 0)
      // Run inference by invoking the `Interpreter`.
      try interpreter.invoke()
        
      // Get the output `Tensor` to process the inference results.
      outputTensor = try interpreter.output(at: 0)
    } catch let error {
      print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return nil
    }

    let results: [Float]
    switch outputTensor.dataType {
    case .uInt8:
      guard let quantization = outputTensor.quantizationParameters else {
        print("No results returned because the quantization values for the output tensor are nil.")
        return nil
      }
      let quantizedResults = [UInt8](outputTensor.data)
      results = quantizedResults.map {
        quantization.scale * Float(Int($0) - quantization.zeroPoint)
      }
    case .float32:
      results = [Float32](unsafeData: outputTensor.data) ?? []
    default:
      print("Output tensor data type \(outputTensor.dataType) is unsupported for this example app.")
      return nil
    }

    // Process the results.
    //let topNInferences = getTopN(results: results)

    interval = Date().timeIntervalSince(startDate) * 1000
    
      
    let resultsArray = NSArray(array: results)
      
    // Return the inference time and inference results.
    return DCPInterference(inferenceTime: interval, keypoints: resultsArray)
      
  }

  // MARK: - Private Methods

    /// Processes the array with results
    func processDCPResults(results: [Float])-> [Int]{
        var intResults: [Int] = []
        
        for coord in results {
            intResults.append(Int(round(coord*150)))
        }
        
        return intResults
    }

  /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
  ///
  /// - Parameters
  ///   - buffer: The pixel buffer to convert to RGB data.
  ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
  ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
  ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
  ///       floating point values).
  /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
  ///     converted.
  private func rgbDataFromBuffer(
    _ buffer: CVPixelBuffer,
    byteCount: Int,
    isModelQuantized: Bool
  ) -> Data? {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer {
      CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
    }
    guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
      return nil
    }
    
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    let destinationChannelCount = 3
    let destinationBytesPerRow = destinationChannelCount * width
    
    var sourceBuffer = vImage_Buffer(data: sourceData,
                                     height: vImagePixelCount(height),
                                     width: vImagePixelCount(width),
                                     rowBytes: sourceBytesPerRow)
    
    guard let destinationData = malloc(height * destinationBytesPerRow) else {
      print("Error: out of memory")
      return nil
    }
    
    defer {
        free(destinationData)
    }

    var destinationBuffer = vImage_Buffer(data: destinationData,
                                          height: vImagePixelCount(height),
                                          width: vImagePixelCount(width),
                                          rowBytes: destinationBytesPerRow)

    let pixelBufferFormat = CVPixelBufferGetPixelFormatType(buffer)

    switch (pixelBufferFormat) {
    case kCVPixelFormatType_32BGRA:
        vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    case kCVPixelFormatType_32ARGB:
        vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    case kCVPixelFormatType_32RGBA:
        vImageConvert_RGBA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    default:
        // Unknown pixel format.
        return nil
    }

    let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
    if isModelQuantized {
        return byteData
    }

    // Not quantized, convert to floats
    let bytes = Array<UInt8>(unsafeData: byteData)!
    var floats = [Float]()
    for i in 0..<bytes.count {
        floats.append(Float(bytes[i])) // here we can normalize by divind the float number by 255, etc.
    }
    return Data(copyingBufferOf: floats)
  }
}

