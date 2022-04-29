//
//  ModelInfo.swift
//  ObjectDetection
//
//  Created by A. Ichwan Yasir on 08/07/21.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//


import CoreImage
import TensorFlowLite
import UIKit
import Accelerate

///// Stores one formatted inference.
struct Inference {
  let confidence: Float
  let className: String
  let rect: CGRect
  let displayColor: UIColor
}
struct SSDInterference {
  let scores: NSMutableArray
  let classes: NSMutableArray
  let boundingboxs: NSMutableArray
  let resultDCP: NSMutableArray
}


typealias FileInfo = (name: String, extension: String)

struct ResultSSD {
  let inferenceTime: Double
  let inferences: SSDInterference
}


struct LandmarkInterference {
  let inferenceTime: Double
  let keypoints: NSMutableArray
  let angles: NSMutableArray
  let point2coords: NSMutableArray
}

struct DCPInterference {
  let inferenceTime: Double
  let keypoints: NSArray
}


/// Information about the MobileNet SSD model.
enum MobilwTflite {
  static let modelInfo: FileInfo = (name: "SSD_modelV4_640px_2000imgs", extension: "tflite")
  static let modelDCPInfo: FileInfo = (name: "dcp_8-keypoints_1.0.2", extension: "tflite")
  static let modelLandMark: FileInfo = (name: "hand_landmark", extension: "tflite")
  static let labelsInfo: FileInfo = (name: "labelmap", extension: "txt")
}


class modelHelper {
    
    // image mean and std for floating model, should be consistent with parameters used in model training
    static let imageMean: Float = 127.5
    static let imageStd:  Float = 127.5
    
    /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
    ///
    /// - Parameters
    ///   - buffer: The BGRA pixel buffer to convert to RGB data.
    ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
    ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
    ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
    ///       floating point values).
    /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
    ///     converted.
    static func rgbDataFromBuffer(
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
      
      if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA){
        vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
      } else if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32ARGB) {
        vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
      }

      let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
      if isModelQuantized {
        return byteData
      }

      // Not quantized, convert to floats
      let bytes = Array<UInt8>(unsafeData: byteData)!
      var floats = [Float]()
      for i in 0..<bytes.count {
        floats.append((Float(bytes[i]) - imageMean) / imageStd)
      }
      return Data(copyingBufferOf: floats)
    }
    
    
}
