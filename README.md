# `swift-embeddings`

Run embedding models locally in `Swift` using `MLTensor`.
Inspired by [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings).

## Supported Models Archictectures

- BERT (Bidirectional Encoder Representations from Transformers)

## Installation

Add the following to your `Package.swift` file. In the package dependencies add:

```swift
dependencies: [
    .package(url: "https://github.com/jkrukowski/swift-embeddings", from: "0.0.1")
]
```

In the target dependencies add:

```swift
dependencies: [
    .product(name: "Bert", package: "swift-embeddings")
]
```

## Usage

```swift
import Bert

// load model and tokenizer from Hugging Face
let modelBundle = try await Bert.loadModelBundle(from: "sentence-transformers/all-MiniLM-L6-v2")

// encode text
let result: [Float32] = await modelBundle.encode(text)

// print result
print(result)
```

## Command Line Demo

To run the command line demo, use the following command:

```bash
swift run embeddings-cli [--model-id <model-id>] [--text <text>]
```

Command line options:

```bash
--model-id <model-id>   (default: sentence-transformers/all-MiniLM-L6-v2)
--text <text>           (default: Text to encode)
-h, --help              Show help information.
```

## Code Formatting

This project uses [swift-format](https://github.com/swiftlang/swift-format). To format the code run:

```bash
swift-format format . -i -r --configuration .swift-format
```

## Acknowledgements

This project is based on and uses some of the code from:

- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)
