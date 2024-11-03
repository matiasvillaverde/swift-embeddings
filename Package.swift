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
            name: "Embeddings",
            targets: ["Embeddings"]),
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
            from: "0.1.14"
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
                "Embeddings",
                "MLTensorNN",
                .product(name: "Safetensors", package: "swift-safetensors"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .target(
            name: "Embeddings",
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
            name: "EmbeddingsTests",
            dependencies: [
                "Embeddings",
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
