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
- [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)

NOTE: `google-bert/bert-base-uncased` is supported but `weightKeyTransform` must be provided:

```swift
let modelBundle = try await Bert.loadModelBundle(from: modelId, weightKeyTransform: Bert.googleWeightsKeyTransform)
```

### XLM-RoBERTa (Cross-lingual Language Model - Robustly Optimized BERT Approach)

Some of the supported models on `Hugging Face`:

- [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- [tomaarsen/xlm-roberta-base-multilingual-en-ar-fr-de-es-tr-it](https://huggingface.co/tomaarsen/xlm-roberta-base-multilingual-en-ar-fr-de-es-tr-it)

### CLIP (Contrastive Language–Image Pre-training)

NOTE: only text encoding is supported for now.
Some of the supported models on `Hugging Face`:

- [jkrukowski/clip-vit-base-patch16](https://huggingface.co/jkrukowski/clip-vit-base-patch16)
- [jkrukowski/clip-vit-base-patch32](https://huggingface.co/jkrukowski/clip-vit-base-patch32)
- [jkrukowski/clip-vit-large-patch14](https://huggingface.co/jkrukowski/clip-vit-large-patch14)

### Word2Vec

NOTE: it's a word embedding model. It loads and keeps the whole model in memory.
For the more memory efficient solution, you might want to use [SQLiteVec](https://github.com/jkrukowski/SQLiteVec).
Some of the supported models on `Hugging Face`:

- [jkrukowski/glove-twitter-25](https://huggingface.co/jkrukowski/glove-twitter-25)
- [jkrukowski/glove-twitter-50](https://huggingface.co/jkrukowski/glove-twitter-50)
- [jkrukowski/glove-twitter-100](https://huggingface.co/jkrukowski/glove-twitter-100)
- [jkrukowski/glove-twitter-200](https://huggingface.co/jkrukowski/glove-twitter-200)

## Installation

Add the following to your `Package.swift` file. In the package dependencies add:

```swift
dependencies: [
    .package(url: "https://github.com/jkrukowski/swift-embeddings", from: "0.0.7")
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

To run the command line demo, use the following command:

```bash
swift run embeddings-cli <subcommand> [--model-id <model-id>] [--model-file <model-file>] [--text <text>] [--max-length <max-length>]
```

Subcommands:

```bash
bert                    Encode text using BERT model
clip                    Encode text using CLIP model
xlm-roberta             Encode text using XLMRoberta model
word2vec                Encode word using Word2Vec model
```

Command line options:

```bash
--model-id <model-id>                       Id of the model to use
--model-file <model-file>                   Path to the model file (only for `Word2Vec`)
--text <text>                               Text to encode
--max-length <max-length>                   Maximum length of the input (not for `Word2Vec`)
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
