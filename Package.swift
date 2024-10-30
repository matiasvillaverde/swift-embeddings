// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "swift-embeddings",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
        .tvOS(.v18),
        .visionOS(.v2),
        .watchOS(.v11),
    ],
    products: [
        .executable(
            name: "embeddings-cli",
            targets: ["EmbeddingsCLI"]
        ),
        .library(
            name: "BertEmbeddings",
            targets: ["BertEmbeddings"]),
        .library(
            name: "MLTensorNN",
            targets: ["MLTensorNN"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/apple/swift-numerics.git",
            from: "1.0.2"
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers.git",
            revision: "2c68d534155b702a504541e60d8eb6e0ab395478"
        ),
        .package(
            url: "https://github.com/jkrukowski/swift-safetensors.git",
            from: "0.0.6"
        ),
        .package(
            url: "https://github.com/apple/swift-argument-parser.git",
            from: "1.5.0"
        ),
    ],
    targets: [
        .executableTarget(
            name: "EmbeddingsCLI",
            dependencies: [
                "BertEmbeddings",
                .product(name: "Safetensors", package: "swift-safetensors"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .target(
            name: "BertEmbeddings",
            dependencies: [
                "MLTensorNN",
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .target(
            name: "MLTensorNN"),
        .target(
            name: "TestingUtils",
            dependencies: [
                .product(name: "Numerics", package: "swift-numerics")
            ]
        ),
        .testTarget(
            name: "BertEmbeddingsTests",
            dependencies: [
                "BertEmbeddings",
                "MLTensorNN",
                "TestingUtils",
                .product(name: "Safetensors", package: "swift-safetensors"),
            ]
        ),
        .testTarget(
            name: "MLTensorNNTests",
            dependencies: [
                "MLTensorNN",
                "TestingUtils",
            ]
        ),
    ]
)
