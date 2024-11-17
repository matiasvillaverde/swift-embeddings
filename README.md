# `swift-embeddings`

[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fjkrukowski%2Fswift-embeddings%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/jkrukowski/swift-embeddings)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fjkrukowski%2Fswift-embeddings%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/jkrukowski/swift-embeddings)

Run embedding models locally in `Swift` using `MLTensor`.
Inspired by [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings).

## Supported Models Archictectures

### BERT (Bidirectional Encoder Representations from Transformers)

Some of the supported models on `Hugging Face`:

- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [sentence-transformers/msmarco-bert-base-dot-v5](https://huggingface.co/sentence-transformers/msmarco-bert-base-dot-v5)
- [thenlper/gte-base](https://huggingface.co/thenlper/gte-base)

### CLIP (Contrastive Language–Image Pre-training)

NOTE: only text encoding is supported for now.
Some of the supported models on `Hugging Face`:

- [jkrukowski/clip-vit-base-patch16](https://huggingface.co/jkrukowski/clip-vit-base-patch16)
- [jkrukowski/clip-vit-base-patch32](https://huggingface.co/jkrukowski/clip-vit-base-patch32)
- [jkrukowski/clip-vit-large-patch14](https://huggingface.co/jkrukowski/clip-vit-large-patch14)

## Installation

Add the following to your `Package.swift` file. In the package dependencies add:

```swift
dependencies: [
    .package(url: "https://github.com/jkrukowski/swift-embeddings", from: "0.0.4")
]
```

In the target dependencies add:

```swift
dependencies: [
    .product(name: "Embeddings", package: "swift-embeddings")
]
```

## Usage

### Encoding

```swift
import Embeddings

// load model and tokenizer from Hugging Face
let modelBundle = try await Bert.loadModelBundle(
    from: "sentence-transformers/all-MiniLM-L6-v2"
)

// encode text
let encoded = modelBundle.encode("The cat is black")
let result = await encoded.cast(to: Float.self).shapedArray(of: Float.self).scalars

// print result
print(result)
```

### Batch Encoding

```swift
import Embeddings
import MLTensorUtils

let texts = [
    "The cat is black",
    "The dog is black",
    "The cat sleeps well"
]
let modelBundle = try await Bert.loadModelBundle(
    from: "sentence-transformers/all-MiniLM-L6-v2"
)
let encoded = modelBundle.batchEncode(texts)
let distance = cosineDistance(encoded, encoded)
let result = await distance.cast(to: Float.self).shapedArray(of: Float.self).scalars
print(result)
```

## Command Line Demo

### BERT

To run the `BERT` command line demo, use the following command:

```bash
swift run embeddings-cli bert [--model-id <model-id>] [--text <text>] [--max-length <max-length>]
```

Command line options:

```bash
--model-id <model-id>                       (default: sentence-transformers/all-MiniLM-L6-v2)
--text <text>                               (default: a photo of a dog)
--max-length <max-length>                   (default: 512)
-h, --help                                  Show help information.
```

### CLIP

To run the `CLIP` command line demo, use the following command:

```bash
swift run embeddings-cli clip [--model-id <model-id>] [--text <text>] [--max-length <max-length>]
```

Command line options:

```bash
--model-id <model-id>                       (default: jkrukowski/clip-vit-base-patch16)
--text <text>                               (default: a photo of a dog)
--max-length <max-length>                   (default: 77)
-h, --help                                  Show help information.
```

## Code Formatting

This project uses [swift-format](https://github.com/swiftlang/swift-format). To format the code run:

```bash
swift format . -i -r --configuration .swift-format
```

## Acknowledgements

This project is based on and uses some of the code from:

- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)
