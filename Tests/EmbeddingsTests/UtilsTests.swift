import CoreML
import Testing

@testable import Embeddings

@Test func googleWeightsKeyTransform() async {
    #expect(Bert.googleWeightsKeyTransform("some.weight.key") == "bert.some.weight.key")
    #expect(Bert.googleWeightsKeyTransform("some.LayerNorm.weight") == "bert.some.LayerNorm.gamma")
    #expect(Bert.googleWeightsKeyTransform("some.LayerNorm.bias") == "bert.some.LayerNorm.beta")
    #expect(Bert.googleWeightsKeyTransform("some.Embedding.weight") == "bert.some.Embedding.weight")
    #expect(Bert.googleWeightsKeyTransform("some.Embedding.bias") == "bert.some.Embedding.bias")
}
