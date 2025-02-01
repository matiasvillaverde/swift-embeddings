import CoreML
import MLTensorUtils
import Testing
import TestingUtils

@testable import Embeddings

struct StaticEmbeddingsTests {
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    func modelBundle(tokenizedValues: [Int32]) -> StaticEmbeddings.ModelBundle {
        let data: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        let embeddings = MLTensor(shape: [3, 3], scalars: data)
        let model = StaticEmbeddings.Model(embeddings: embeddings)
        let tokenizer = TextTokenizerMock(tokenizedValues: tokenizedValues)
        return StaticEmbeddings.ModelBundle(model: model, tokenizer: tokenizer)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func staticEmbeddings() async throws {
        let modelBundle = modelBundle(tokenizedValues: [0, 1, 2])
        let encoded = try modelBundle.encode("Text")
        let result = await encoded.scalars(of: Float.self)

        #expect(allClose(result, [0.4, 0.5, 0.6]) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func staticEmbeddingsWhenNormalize() async throws {
        let modelBundle = modelBundle(tokenizedValues: [0, 1, 2])
        let encoded = try modelBundle.encode("Text", normalize: true)
        let result = await encoded.scalars(of: Float.self)

        #expect(allClose(result, [0.45584226, 0.56980276, 0.6837634]) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func staticEmbeddingsWhenTruncateDimension() async throws {
        let modelBundle = modelBundle(tokenizedValues: [0, 1, 2])
        let encoded = try modelBundle.encode("Text", truncateDimension: 2)
        let result = await encoded.scalars(of: Float.self)

        #expect(allClose(result, [0.4, 0.5]) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func staticEmbeddingsWhenTokenizerReturnsEmpty() async throws {
        let modelBundle = modelBundle(tokenizedValues: [])
        let encoded = try modelBundle.encode("Text")
        let result = await encoded.scalars(of: Float.self)

        #expect(allClose(result, [0.0, 0.0, 0.0]) == true)
    }
}
