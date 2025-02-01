import CoreML
import MLTensorUtils
import Testing
import TestingUtils

@testable import Embeddings

struct Model2VecTests {
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    func modelBundle(
        tokenizedValues: [Int32],
        unknownTokenId: Int? = nil
    ) -> Model2Vec.ModelBundle {
        let data: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        let embeddings = MLTensor(shape: [3, 3], scalars: data)
        let model = Model2Vec.Model(embeddings: embeddings)
        let tokenizer = TextTokenizerMock(
            tokenizedValues: tokenizedValues,
            unknownTokenId: unknownTokenId
        )
        return Model2Vec.ModelBundle(model: model, tokenizer: tokenizer)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func model2VecEmbeddings() async throws {
        let modelBundle = modelBundle(tokenizedValues: [0, 1, 2])
        let encoded = try modelBundle.encode("Text")
        let result = await encoded.scalars(of: Float.self)

        #expect(allClose(result, [0.4, 0.5, 0.6]) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func model2VecWhenNormalize() async throws {
        let modelBundle = modelBundle(tokenizedValues: [0, 1, 2])
        let encoded = try modelBundle.encode("Text", normalize: true)
        let result = await encoded.scalars(of: Float.self)

        #expect(allClose(result, [0.45584226, 0.56980276, 0.6837634]) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func model2VecWhenUnknownTokenId() async throws {
        let modelBundle = modelBundle(tokenizedValues: [0, 1, 2], unknownTokenId: 0)
        let encoded = try modelBundle.encode("Text")
        let result = await encoded.scalars(of: Float.self)

        #expect(allClose(result, [0.55, 0.65, 0.75]) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func model2VecWhenTokenizerReturnsEmpty() async throws {
        let modelBundle = modelBundle(tokenizedValues: [])
        let encoded = try modelBundle.encode("Text")
        let result = await encoded.scalars(of: Float.self)

        #expect(allClose(result, [0.0, 0.0, 0.0]) == true)
    }
}
