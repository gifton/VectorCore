#!/usr/bin/env ruby
# One-time scaffold generator. Refuses to modify an existing Xcode project.
require 'xcodeproj'
directory = File.expand_path(__dir__)
path = File.join(directory, 'QualificationHost.xcodeproj')
abort('Project already exists') if File.exist?(path)
project = Xcodeproj::Project.new(path)
app = project.new_target(:application, 'QualificationHost', :ios, '17.0')
witness = project.new_target(:static_library, 'BufferValueWitness', :ios, '17.0')
%w[App.swift ProbeAdapter.swift].each do |source|
  app.source_build_phase.add_file_reference(project.main_group.new_file(source))
end
witness.source_build_phase.add_file_reference(project.main_group.new_file('BufferValueWitness.swift'))
app.add_dependency(witness)
app.frameworks_build_phase.add_file_reference(witness.product_reference)
[app, witness].each do |target|
  target.build_configurations.each do |configuration|
    settings = configuration.build_settings
    settings['SWIFT_VERSION'] = '6.0'
    settings['ARCHS'] = 'arm64'
    settings['ONLY_ACTIVE_ARCH'] = 'YES'
    settings['SWIFT_OPTIMIZATION_LEVEL'] = configuration.name == 'Release' ? '-O' : '-Onone'
    settings['SWIFT_COMPILATION_MODE'] = 'singlefile'
    settings['SWIFT_INCLUDE_PATHS'] = '$(inherited) $(VC_PRODUCER_MODULES)'
    settings['OTHER_SWIFT_FLAGS'] = '$(inherited) $(VC_MODULE_FLAGS)'
    settings['SWIFT_ACTIVE_COMPILATION_CONDITIONS'] = '$(inherited) $(VC_PROBE_CONDITIONS)'
    settings['CODE_SIGN_STYLE'] = 'Automatic'
    settings['DEVELOPMENT_TEAM'] = ''
  end
end
app.build_configurations.each do |configuration|
  settings = configuration.build_settings
  settings['PRODUCT_BUNDLE_IDENTIFIER'] = 'org.vectorcore.qualification.Host'
  settings['GENERATE_INFOPLIST_FILE'] = 'YES'
  settings['ENABLE_DEBUG_DYLIB'] = 'NO'
  settings['INFOPLIST_KEY_CFBundleDisplayName'] = 'VectorCore Qualification'
  settings['INFOPLIST_KEY_UILaunchScreen_Generation'] = 'YES'
  settings['INFOPLIST_KEY_UIApplicationSceneManifest_Generation'] = 'YES'
  settings['INFOPLIST_KEY_UIApplicationSupportsIndirectInputEvents'] = 'YES'
  settings['TARGETED_DEVICE_FAMILY'] = '1,2'
  settings['OTHER_LDFLAGS'] = '$(inherited) $(VC_PRODUCER_OBJECT_FLAGS) -framework Accelerate'
end
project.save
scheme = Xcodeproj::XCScheme.new
scheme.add_build_target(app)
scheme.set_launch_target(app)
scheme.save_as(path, 'QualificationHost', true)
puts path
