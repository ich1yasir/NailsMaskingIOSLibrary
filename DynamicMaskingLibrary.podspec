#
# Be sure to run `pod lib lint DynamicMaskingLibrary.podspec' to ensure this is a
# valid spec before submitting.
#
# Any lines starting with a # are optional, but their use is encouraged
# To learn more about a Podspec see https://guides.cocoapods.org/syntax/podspec.html
#

Pod::Spec.new do |s|
  s.name             = 'DynamicMaskingLibrary'
  s.version          = '1.0.0'
  s.summary          = 'A short description of DynamicMaskingLibrary.'

# This description is used to generate tags and improve search results.
#   * Think: What does it do? Why did you write it? What is the focus?
#   * Try to keep it short, snappy and to the point.
#   * Write the description between the DESC delimiters below.
#   * Finally, don't worry about the indent, CocoaPods strips it!

  s.description      = <<-DESC
TODO: Add long description of the pod here.
                       DESC

  s.homepage         = 'https://github.com/ich1yasir/NailsMaskingIOSLibrary'
  # s.screenshots     = 'www.example.com/screenshots_1', 'www.example.com/screenshots_2'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'ich1yasir' => '43979098+ich1yasir@users.noreply.github.com' }
  s.source           = { :git => 'https://github.com/ich1yasir/NailsMaskingIOSLibrary.git', :tag => s.version.to_s }
  # s.social_media_url = 'https://twitter.com/<TWITTER_USERNAME>'

  s.ios.deployment_target = '9.0'

  s.source_files = 'DynamicMaskingLibrary/Classes/**/*'
  
  s.resource_bundles = {
    'DynamicMaskingLibrary' => 'DynamicMaskingLibrary/Assets/**/*'
  }
#  s.source_files = 'XDCoreLib/Pod/Classes/**/*'
#  s.resource_bundles = { 'XDCoreLib' => ['XDCoreLib/Pod/Resources/**/*.{png,storyboard}'] }
  
#  s.resource_bundles = {
#  ‘MyPodName’ => [‘MyPodName/Classes/**/*.js’]
#  }

s.resource_bundles = {
    'DynamicMaskingLibrary' => ['DynamicMaskingLibrary/Assets/**/*']
}

  
  s.public_header_files = 'DynamicMaskingLibrary/Classes/**/*.h'
  s.frameworks = 'UIKit'
  
  
  # Ref: https://github.com/CocoaPods/CocoaPods/issues/7234
  s.static_framework = true
  
  # s.dependency 'AFNetworking', '~> 2.3'
  s.dependency 'TensorFlowLiteSwift', '2.7.0'
  s.dependency 'TensorFlowLiteSwift/CoreML', '2.7.0'
  s.dependency 'OpenCV', '~> 4.3.0'
end
