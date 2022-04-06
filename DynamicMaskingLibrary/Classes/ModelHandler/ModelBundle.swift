//
//  ModelBundle.swift
//  DynamicMaskingLibrary
//
//  Created by achmad ichwan yasir on 06/04/22.
//

import Foundation

final class ModelBundleClass {
    static let resourceBundle: Bundle = {
        let myBundle = Bundle(for: ModelBundleClass.self)

        guard let resourceBundleURL = myBundle.url(
            forResource: "DynamicMaskingLibrary", withExtension: "bundle")
            else { fatalError("MySDK.bundle not found!") }

        guard let resourceBundle = Bundle(url: resourceBundleURL)
            else { fatalError("Cannot access MySDK.bundle!") }

        return resourceBundle
    }()
}
