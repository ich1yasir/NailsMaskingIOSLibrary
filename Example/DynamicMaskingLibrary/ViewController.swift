//
//  ViewController.swift
//  DynamicMaskingLibrary
//
//  Created by ich1yasir on 04/01/2022.
//  Copyright (c) 2022 ich1yasir. All rights reserved.
//

import UIKit
import DynamicMaskingLibrary

class ViewController: UIViewController {
    @IBOutlet weak var imageViewHand: UIImageView!
    
    // Dynamic Masking module
    private var dynamicMasking: DynamicMasking?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        dynamicMasking = DynamicMasking()
        
        let namehand = "hand_2"
        let nameart = "art_11"
        let color = "#A62C17"
        let artColorTest: UIImage? = UIImage(named: nameart)
        let handTest: UIImage? = UIImage(named: namehand)
        
        if let resultImage: UIImage = dynamicMasking?.predictRunProcess(handImage: handTest, threadCount: 4, color: color, artColor: artColorTest!, artMode: false) {
            imageViewHand.image = resultImage
        } else {
            print("Failed to paint image.")
        }
        
        // Do any additional setup after loading the view, typically from a nib.
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

}

