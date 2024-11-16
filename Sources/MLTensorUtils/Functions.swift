import CoreML
import Foundation

public func norm(_ x: MLTensor, alongAxes: Int = 1, keepRank: Bool = false) -> MLTensor {
    x.squared().sum(alongAxes: alongAxes, keepRank: keepRank).squareRoot()
}

public func cosineSimilarity(_ x: MLTensor, _ y: MLTensor, alongAxes: Int = 1) -> MLTensor {
    let normX = norm(x, alongAxes: alongAxes)
    let normY = norm(y, alongAxes: alongAxes)
    return x.matmul(y.transposed()) / (normX * normY)
}

public func dotProduct(_ x: MLTensor, _ y: MLTensor) -> MLTensor {
    x.transposed().matmul(y)
}

public func cosineDistance(_ x: MLTensor, _ y: MLTensor, alongAxes: Int = 1) -> MLTensor {
    1 - cosineSimilarity(x, y, alongAxes: alongAxes)
}

public func euclideanDistance(_ x: MLTensor, _ y: MLTensor, alongAxes: Int = 1) -> MLTensor {
    (x - y).squared().sum(alongAxes: alongAxes).squareRoot()
}

public func additiveCausalMask<Scalar: MLTensorScalar>(
    _ n: Int32,
    scalarType: Scalar.Type = Float.self
) -> MLTensor {
    let indices = MLTensor(0..<n)
    let mask = indices.expandingShape(at: 1) .< indices.expandingShape(at: 0)
    return mask.cast(to: scalarType) * -1e9
}
