// swift-tools-version:5.5
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "NumSwift",
    platforms: [ .iOS(.v14),
                 .tvOS(.v14),
                 .watchOS(.v7),
                 .macOS(.v11)],
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
        .library(
            name: "NumSwift",
            targets: ["NumSwift", "NumSwiftC"])
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages this package depends on.
      .target(
          name: "NumSwiftC",
          dependencies: [],
          cSettings: [
            .define("__ARM_FEATURE_FP16_VECTOR_ARITHMETIC", .when(platforms: [.iOS, .tvOS, .watchOS])),
            .unsafeFlags(["-march=armv8.2-a+fp16"], .when(platforms: [.iOS, .tvOS, .watchOS]))
          ]),
      .target(
            name: "NumSwift",
            dependencies: ["NumSwiftC"],
            resources: [ ]),
        .testTarget(
            name: "NumSwiftTests",
            dependencies: ["NumSwift"])
    ]
)
