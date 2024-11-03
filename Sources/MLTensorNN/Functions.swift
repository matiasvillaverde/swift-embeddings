import CoreML
import Foundation

public func l2Norm(_ x: MLTensor, alongAxes: Int = 1) -> MLTensor {
    return x.squared().sum(alongAxes: alongAxes).squareRoot()
}

public func cosineSimilarity(_ x: MLTensor, _ y: MLTensor, alongAxes: Int = 1) -> MLTensor {
    let normX = l2Norm(x, alongAxes: alongAxes)
    let normY = l2Norm(y, alongAxes: alongAxes)
    return x.matmul(y.transposed()) / (normX * normY)
}
