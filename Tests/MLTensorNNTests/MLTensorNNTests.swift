import CoreML
import Numerics
import Testing
import TestingUtils

@testable import MLTensorNN

@Test func l2Norm() async {
    let x = MLTensor(
        shape: [2, 3],
        scalars: [1, 2, 3, 4, 5, 6],
        scalarType: Float.self
    )
    let result = l2Norm(x)

    #expect(result.shape == [2])
    let resultArray = await result.scalars(of: Float.self)
    let expectedArray: [Float] = [3.7417, 8.7749]
    #expect(allClose(resultArray, expectedArray) == true)
}

@Test func cosineSimilarity1D() async {
    let x = MLTensor(
        shape: [1, 6],
        scalars: [1, 2, 3, 4, 5, 6],
        scalarType: Float.self
    )
    let y = MLTensor(
        shape: [1, 6],
        scalars: [4, 5, 6, 7, 8, 9],
        scalarType: Float.self
    )
    let result = cosineSimilarity(x, y)

    #expect(result.shape == [1, 1])
    let resultArray = await result.shapedArray(of: Float.self).scalars
    let expectedArray: [Float] = [0.980653]
    #expect(allClose(resultArray, expectedArray) == true)
}

@Test func cosineSimilarity2D() async {
    let x = MLTensor(
        shape: [2, 3],
        scalars: [1, 2, 3, 4, 5, 6],
        scalarType: Float.self
    )
    let y = MLTensor(
        shape: [2, 3],
        scalars: [4, 5, 6, 7, 8, 9],
        scalarType: Float.self
    )
    let result = cosineSimilarity(x, y)

    #expect(result.shape == [2, 2])
    let resultArray = await result.shapedArray(of: Float.self).scalars
    let expectedArray: [Float] = [0.9746318, 0.4090946, 2.3452077, 0.9981909]
    #expect(allClose(resultArray, expectedArray) == true)
}

@Test func cosineSimilaritySameTensor() async {
    let x = MLTensor(
        shape: [2, 3],
        scalars: [1, 2, 3, 1, 2, 3],
        scalarType: Float.self
    )
    let result = cosineSimilarity(x, x)

    #expect(result.shape == [2, 2])
    let resultArray = await result.shapedArray(of: Float.self).scalars
    let expectedArray: [Float] = [1.0, 1.0, 1.0, 1.0]
    #expect(allClose(resultArray, expectedArray) == true)
}

@Test func embeddingLayer1D() async {
    let embedding = embedding(weight: MLTensor.float(shape: [12]))
    let result = embedding(MLTensor([0, 2, 4] as [Int32]))

    #expect(result.shape == [3])
    let resultArray = await result.scalars(of: Float.self)
    #expect(resultArray == [0, 2, 4])
}

@Test func embeddingLayer2D() async {
    let embedding = embedding(weight: MLTensor.float(shape: [6, 2]))
    let result = embedding(MLTensor([0, 2, 4] as [Int32]))

    #expect(result.shape == [3, 2])
    let resultArray = await result.scalars(of: Float.self)
    #expect(resultArray == [0, 1, 4, 5, 8, 9])
}

@Test func embeddingLayer3D() async {
    let embedding = embedding(weight: MLTensor.float(shape: [2, 2, 2]))
    let result = embedding(MLTensor([0, 1] as [Int32]))

    #expect(result.shape == [2, 2, 2])
    let resultArray = await result.scalars(of: Float.self)
    #expect(resultArray == [0, 1, 2, 3, 4, 5, 6, 7])
}

@Test func layerNorm1D() async {
    let weight = MLTensor(
        shape: [3],
        scalars: [1, 2, 3],
        scalarType: Float.self
    )
    let bias = MLTensor(
        shape: [3],
        scalars: [4, 5, 6],
        scalarType: Float.self
    )
    let layerNorm = layerNorm(weight: weight, bias: bias, epsilon: 1e-5)
    let input = MLTensor(
        shape: [2, 3],
        scalars: [1, 2, 3, 4, 5, 6],
        scalarType: Float.self
    )
    let result = layerNorm(input)

    #expect(result.shape == [2, 3])
    let resultArray = await result.scalars(of: Float.self)
    let expectedArray: [Float] = [2.7753, 5.0000, 9.6742, 2.7753, 5.0000, 9.6742]
    #expect(allClose(resultArray, expectedArray) == true)
}

@Test func layerNorm2D() async {
    let weight = MLTensor(
        shape: [2, 3],
        scalars: [1, 2, 3, 4, 5, 6],
        scalarType: Float.self
    )
    let bias = MLTensor(
        shape: [2, 3],
        scalars: [4, 5, 6, 7, 8, 9],
        scalarType: Float.self
    )
    let layerNorm = layerNorm(weight: weight, bias: bias, epsilon: 1e-5)
    let input = MLTensor(
        shape: [1, 2, 3],
        scalars: [1, 2, 3, 4, 5, 6],
        scalarType: Float.self
    )
    let result = layerNorm(input)

    #expect(result.shape == [1, 2, 3])
    let resultArray = await result.shapedArray(of: Float.self).scalars
    let expectedArray: [Float] = [2.7753, 5.0000, 9.6742, 2.1011, 8.0000, 16.3484]
    #expect(allClose(resultArray, expectedArray) == true)
}

@Test func layerNorm3D() async {
    let weight = MLTensor(
        shape: [1, 2, 3],
        scalars: [1, 2, 3, 4, 5, 6],
        scalarType: Float.self
    )
    let bias = MLTensor(
        shape: [1, 2, 3],
        scalars: [4, 5, 6, 7, 8, 9],
        scalarType: Float.self
    )
    let layerNorm = layerNorm(weight: weight, bias: bias, epsilon: 1e-5)
    let input = MLTensor(
        shape: [1, 2, 3],
        scalars: [1, 2, 3, 4, 5, 6],
        scalarType: Float.self
    )
    let result = layerNorm(input)

    #expect(result.shape == [1, 2, 3])
    let resultArray = await result.shapedArray(of: Float.self).scalars
    let expectedArray: [Float] = [2.7753, 5.0000, 9.6742, 2.1011, 8.0000, 16.3484]
    #expect(allClose(resultArray, expectedArray) == true)
}

@Test func erf() async {
    let input1 = MLTensor(shape: [1], scalars: [0], scalarType: Float.self)
    let result1 = await erf(input1).shapedArray(of: Float.self).scalars
    #expect(result1 == [0])

    let input2 = MLTensor(shape: [1], scalars: [.infinity], scalarType: Float.self)
    let result2 = await erf(input2).shapedArray(of: Float.self).scalars
    #expect(result2 == [1])

    let input3 = MLTensor(shape: [1], scalars: [-.infinity], scalarType: Float.self)
    let result3 = await erf(input3).shapedArray(of: Float.self).scalars
    #expect(result3 == [-1])

    let input4 = MLTensor(
        shape: [6], scalars: [0.9, 0.5, 0.1, -0.1, -0.5, -0.9], scalarType: Float.self)
    let result4 = await erf(input4).shapedArray(of: Float.self).scalars
    let expected4: [Float] = [
        0.7969082124228322,
        0.5204998778130465,
        0.1124629160182849,
        -0.1124629160182849,
        -0.5204998778130465,
        -0.7969082124228322,
    ]
    #expect(allClose(result4, expected4) == true)
}

@Test func gelu() async {
    let input1 = MLTensor(shape: [1], scalars: [0], scalarType: Float.self)
    let result1 = await gelu(input1).shapedArray(of: Float.self).scalars
    #expect(result1 == [0])

    let input2 = MLTensor(shape: [1], scalars: [100], scalarType: Float.self)
    let result2 = await gelu(input2).shapedArray(of: Float.self).scalars
    #expect(result2 == [100])

    let input3 = MLTensor(shape: [1], scalars: [-100], scalarType: Float.self)
    let result3 = await gelu(input3).shapedArray(of: Float.self).scalars
    #expect(result3 == [0])

    let input4 = MLTensor(
        shape: [6], scalars: [0.9, 0.5, 0.1, -0.1, -0.5, -0.9], scalarType: Float.self)
    let result4 = await gelu(input4).shapedArray(of: Float.self).scalars
    let expected4: [Float] = [
        0.734346, 0.345731, 0.0539828, -0.0460172, -0.154269, -0.165654,
    ]
    #expect(allClose(result4, expected4) == true)
}

@Test func geluApproximationFast() async {
    let input1 = MLTensor(shape: [1], scalars: [0], scalarType: Float.self)
    let result1 = await gelu(input1, approximation: .fast).shapedArray(of: Float.self).scalars
    #expect(result1 == [0])

    let input2 = MLTensor(shape: [1], scalars: [100], scalarType: Float.self)
    let result2 = await gelu(input2, approximation: .fast).shapedArray(of: Float.self).scalars
    #expect(result2 == [100])

    let input3 = MLTensor(shape: [1], scalars: [-100], scalarType: Float.self)
    let result3 = await gelu(input3, approximation: .fast).shapedArray(of: Float.self).scalars
    #expect(result3 == [0])

    let input4 = MLTensor(
        shape: [6], scalars: [0.9, 0.5, 0.1, -0.1, -0.5, -0.9], scalarType: Float.self)
    let result4 = await gelu(input4, approximation: .fast).shapedArray(of: Float.self).scalars
    let expected4: [Float] = [
        0.740043, 0.350388, 0.0542448, -0.0457552, -0.149612, -0.159957,
    ]
    #expect(allClose(result4, expected4) == true)
}

@Test func geluApproximationPrecise() async {
    let input1 = MLTensor(shape: [1], scalars: [0], scalarType: Float.self)
    let result1 = await gelu(input1, approximation: .precise).shapedArray(of: Float.self).scalars
    #expect(result1 == [0])

    let input2 = MLTensor(shape: [1], scalars: [100], scalarType: Float.self)
    let result2 = await gelu(input2, approximation: .precise).shapedArray(of: Float.self).scalars
    #expect(result2 == [100])

    let input3 = MLTensor(shape: [1], scalars: [-100], scalarType: Float.self)
    let result3 = await gelu(input3, approximation: .precise).shapedArray(of: Float.self).scalars
    #expect(result3 == [0])

    let input4 = MLTensor(
        shape: [6], scalars: [0.9, 0.5, 0.1, -0.1, -0.5, -0.9], scalarType: Float.self)
    let result4 = await gelu(input4, approximation: .precise).shapedArray(of: Float.self).scalars
    let expected4: [Float] = [
        0.734228, 0.345714, 0.0539828, -0.0460172, -0.154286, -0.165772,
    ]
    #expect(allClose(result4, expected4) == true)
}
