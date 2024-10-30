import CoreML
import MLTensorNN
import Testing
import TestingUtils

@testable import BertEmbeddings

@Test func testPooler() async {
    let pooler1 = BertEmbeddings.Pooler(
        dense: MLTensorNN.linear(
            weight: MLTensor.float32(shape: [5, 5]),
            bias: nil
        )
    )
    let result1 = pooler1(
        MLTensor.float32(shape: [1, 3, 5])
    )
    let data1 = await result1.scalars(of: Float32.self)

    #expect(result1.shape == [1, 5])
    #expect(allClose(data1, [1, 1, 1, 1, 1]) == true)

    let pooler2 = BertEmbeddings.Pooler(
        dense: MLTensorNN.linear(
            weight: MLTensor.float32(shape: [5, 5]),
            bias: MLTensor.float32(shape: [5])
        )
    )
    let result2 = pooler2(
        MLTensor.float32(shape: [1, 3, 5])
    )
    let data2 = await result2.scalars(of: Float32.self)

    #expect(result2.shape == [1, 5])
    #expect(allClose(data2, [1, 1, 1, 1, 1]) == true)
}

@Test func testIntermediate() async {
    let intermediate1 = BertEmbeddings.Intermediate(
        dense: MLTensorNN.linear(
            weight: MLTensor.float32(shape: [2, 3]),
            bias: nil
        )
    )
    let result1 = intermediate1(
        hiddenStates: MLTensor.float32(shape: [1, 2, 3])
    )
    let data1 = await result1.scalars(of: Float32.self)

    #expect(result1.shape == [1, 2, 2])
    #expect(allClose(data1, [5.0, 14.0, 14.0, 50.0]) == true)

    let intermediate2 = BertEmbeddings.Intermediate(
        dense: MLTensorNN.linear(
            weight: MLTensor.float32(shape: [2, 3]),
            bias: MLTensor.float32(shape: [2])
        )
    )
    let result2 = intermediate2(
        hiddenStates: MLTensor.float32(shape: [1, 2, 3])
    )
    let data2 = await result2.scalars(of: Float32.self)

    #expect(result2.shape == [1, 2, 2])
    #expect(allClose(data2, [5.0, 15.0, 14.0, 51.0]) == true)
}

@Test func testOutput() async {
    let output1 = BertEmbeddings.Output(
        dense: MLTensorNN.linear(
            weight: MLTensor.float32(shape: [4, 4]),
            bias: nil
        ),
        layerNorm: MLTensorNN.layerNorm(
            weight: MLTensor.float32(shape: [4]),
            bias: MLTensor.float32(shape: [4]),
            epsilon: 1e-5
        )
    )

    let result1 = output1(
        hiddenStates: MLTensor.float32(shape: [1, 2, 4]),
        inputTensor: MLTensor.float32(shape: [1, 2, 4])
    )
    let data1 = await result1.scalars(of: Float32.self)

    #expect(result1.shape == [1, 2, 4])
    #expect(
        allClose(
            data1,
            [0.0, 0.5527864, 2.8944273, 7.0249224, 0.0, 0.5527864, 2.8944273, 7.0249224])
            == true)

    let output2 = BertEmbeddings.Output(
        dense: MLTensorNN.linear(
            weight: MLTensor.float32(shape: [4, 4]),
            bias: MLTensor.float32(shape: [4])
        ),
        layerNorm: MLTensorNN.layerNorm(
            weight: MLTensor.float32(shape: [4]),
            bias: MLTensor.float32(shape: [4]),
            epsilon: 1e-5
        )
    )

    let result2 = output2(
        hiddenStates: MLTensor.float32(shape: [1, 2, 4]),
        inputTensor: MLTensor.float32(shape: [1, 2, 4])
    )
    let data2 = await result2.scalars(of: Float32.self)

    #expect(result2.shape == [1, 2, 4])
    #expect(
        allClose(
            data2,
            [0.0, 0.55278635, 2.8944273, 7.0249224, 0.0, 0.5527864, 2.8944273, 7.0249224])
            == true)
}

@Test func testEmbeddings() async {
    let wordEmbeddings = MLTensorNN.embedding(weight: MLTensor.float32(shape: [2, 4]))
    let positionEmbeddings = MLTensorNN.embedding(weight: MLTensor.float32(shape: [1, 4]))
    let tokenTypeEmbeddings = MLTensorNN.embedding(weight: MLTensor.float32(shape: [2, 4]))
    let embeddings = BertEmbeddings.Embeddings(
        wordEmbeddings: wordEmbeddings,
        positionEmbeddings: positionEmbeddings,
        tokenTypeEmbeddings: tokenTypeEmbeddings,
        layerNorm: MLTensorNN.layerNorm(
            weight: MLTensor.float32(shape: [4]),
            bias: MLTensor.float32(shape: [4]),
            epsilon: 1e-5
        )
    )

    let result = embeddings(inputIds: MLTensor.int32(shape: [1, 2]))
    let data = await result.scalars(of: Float32.self)

    #expect(result.shape == [1, 2, 4])
    #expect(
        allClose(data, [0, 0.552787, 2.89443, 7.02492, 0, 0.552787, 2.89443, 7.02492])
            == true)
}
