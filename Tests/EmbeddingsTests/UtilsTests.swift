import CoreML
import Testing

@testable import Embeddings

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func googleWeightsKeyTransform() async {
    #expect(Bert.googleWeightsKeyTransform("some.weight.key") == "bert.some.weight.key")
    #expect(Bert.googleWeightsKeyTransform("some.LayerNorm.weight") == "bert.some.LayerNorm.gamma")
    #expect(Bert.googleWeightsKeyTransform("some.LayerNorm.bias") == "bert.some.LayerNorm.beta")
    #expect(Bert.googleWeightsKeyTransform("some.Embedding.weight") == "bert.some.Embedding.weight")
    #expect(Bert.googleWeightsKeyTransform("some.Embedding.bias") == "bert.some.Embedding.bias")
}

final class TextTokenizerMock: TextTokenizer {
    let tokenizedValues: [Int32]
    let unknownTokenId: Int?

    init(tokenizedValues: [Int32], unknownTokenId: Int? = nil) {
        self.tokenizedValues = tokenizedValues
        self.unknownTokenId = unknownTokenId
    }

    func tokenizeText(_ text: String, maxLength: Int?, addSpecialTokens: Bool) throws -> [Int32] {
        tokenizedValues
    }
}
