import CoreML
import Foundation
import MLTensorNN
@preconcurrency import Tokenizers

public enum BertEmbeddings {}

extension BertEmbeddings {
    public struct ModelConfig: Codable {
        public var modelType: String
        public var numHiddenLayers: Int
        public var numAttentionHeads: Int
        public var hiddenSize: Int
        public var intermediateSize: Int
        public var maxPositionEmbeddings: Int
        public var hiddenDropoutProb: Float
        public var attentionProbsDropoutProb: Float
        public var typeVocabSize: Int
        public var initializerRange: Float
        public var layerNormEps: Float
        public var vocabSize: Int

        public init(
            modelType: String,
            numHiddenLayers: Int,
            numAttentionHeads: Int,
            hiddenSize: Int,
            intermediateSize: Int,
            maxPositionEmbeddings: Int,
            hiddenDropoutProb: Float = 0.1,
            attentionProbsDropoutProb: Float = 0.1,
            typeVocabSize: Int = 2,
            initializerRange: Float = 0.02,
            layerNormEps: Float = 1e-12,
            vocabSize: Int = 30522
        ) {
            self.modelType = modelType
            self.numHiddenLayers = numHiddenLayers
            self.numAttentionHeads = numAttentionHeads
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.hiddenDropoutProb = hiddenDropoutProb
            self.attentionProbsDropoutProb = attentionProbsDropoutProb
            self.typeVocabSize = typeVocabSize
            self.initializerRange = initializerRange
            self.layerNormEps = layerNormEps
            self.vocabSize = vocabSize
        }
    }
}

extension BertEmbeddings {
    public struct Pooler: Sendable {
        let dense: MLTensorNN.Layer

        public init(dense: @escaping MLTensorNN.Layer) {
            self.dense = dense
        }

        public func callAsFunction(_ hiddenStates: MLTensor) -> MLTensor {
            let firstTokenTensor = hiddenStates[0..., 0]
            let pooledOutput = dense(firstTokenTensor)
            return pooledOutput.tanh()
        }
    }
}

extension BertEmbeddings {
    public struct Embeddings: Sendable {
        let wordEmbeddings: MLTensorNN.Layer
        let positionEmbeddings: MLTensorNN.Layer
        let tokenTypeEmbeddings: MLTensorNN.Layer
        let layerNorm: MLTensorNN.Layer

        public init(
            wordEmbeddings: @escaping MLTensorNN.Layer,
            positionEmbeddings: @escaping MLTensorNN.Layer,
            tokenTypeEmbeddings: @escaping MLTensorNN.Layer,
            layerNorm: @escaping MLTensorNN.Layer
        ) {
            self.wordEmbeddings = wordEmbeddings
            self.positionEmbeddings = positionEmbeddings
            self.tokenTypeEmbeddings = tokenTypeEmbeddings
            self.layerNorm = layerNorm
        }

        public func callAsFunction(
            inputIds: MLTensor,
            tokenTypeIds: MLTensor? = nil,
            positionIds: MLTensor? = nil
        ) -> MLTensor {
            let seqLength = inputIds.shape[1]
            let positionIds =
                positionIds
                ?? MLTensor(
                    shape: [1, seqLength],
                    scalars: 0..<Int32(seqLength),
                    scalarType: Int32.self
                )
            let tokenTypeIds =
                tokenTypeIds
                ?? MLTensor(
                    zeros: inputIds.shape,
                    scalarType: Int32.self
                )
            let wordsEmbeddings = wordEmbeddings(inputIds)
            let positionEmbeddings = positionEmbeddings(positionIds)
            let tokenTypeEmbeddings = tokenTypeEmbeddings(tokenTypeIds)
            let embeddings = wordsEmbeddings + positionEmbeddings + tokenTypeEmbeddings
            return layerNorm(embeddings)
        }
    }
}

extension BertEmbeddings {
    public struct Output: Sendable {
        let dense: MLTensorNN.Layer
        let layerNorm: MLTensorNN.Layer

        public init(
            dense: @escaping MLTensorNN.Layer,
            layerNorm: @escaping MLTensorNN.Layer
        ) {
            self.dense = dense
            self.layerNorm = layerNorm
        }

        public func callAsFunction(
            hiddenStates: MLTensor,
            inputTensor: MLTensor
        ) -> MLTensor {
            let dense = dense(hiddenStates)
            let layerNormInput = dense + inputTensor
            return layerNorm(layerNormInput)
        }
    }
}

extension BertEmbeddings {
    public struct Intermediate: Sendable {
        let dense: MLTensorNN.Layer

        public init(dense: @escaping MLTensorNN.Layer) {
            self.dense = dense
        }

        public func callAsFunction(hiddenStates: MLTensor) -> MLTensor {
            let dense = dense(hiddenStates)
            return gelu(dense)
        }
    }
}

extension BertEmbeddings {
    public struct SelfOutput: Sendable {
        let dense: MLTensorNN.Layer
        let layerNorm: MLTensorNN.Layer

        public init(
            dense: @escaping MLTensorNN.Layer,
            layerNorm: @escaping MLTensorNN.Layer
        ) {
            self.dense = dense
            self.layerNorm = layerNorm
        }

        public func callAsFunction(
            hiddenStates: MLTensor,
            inputTensor: MLTensor
        ) -> MLTensor {
            let dense = dense(hiddenStates)
            let layerNormInput = dense + inputTensor
            return layerNorm(layerNormInput)
        }
    }
}

extension BertEmbeddings {
    public struct SelfAttention: Sendable {
        let query: MLTensorNN.Layer
        let key: MLTensorNN.Layer
        let value: MLTensorNN.Layer
        let numAttentionHeads: Int
        let attentionHeadSize: Int
        let allHeadSize: Int

        public init(
            query: @escaping MLTensorNN.Layer,
            key: @escaping MLTensorNN.Layer,
            value: @escaping MLTensorNN.Layer,
            numAttentionHeads: Int,
            attentionHeadSize: Int,
            allHeadSize: Int
        ) {
            self.query = query
            self.key = key
            self.value = value
            self.numAttentionHeads = numAttentionHeads
            self.attentionHeadSize = attentionHeadSize
            self.allHeadSize = allHeadSize
        }

        private func transposeForScores(_ x: MLTensor) -> MLTensor {
            let newShape = x.shape.dropLast() + [numAttentionHeads, attentionHeadSize]
            return x.reshaped(to: Array(newShape)).transposed(permutation: 0, 2, 1, 3)
        }

        public func callAsFunction(
            hiddenStates: MLTensor,
            attentionMask: MLTensor?
        ) -> MLTensor {
            let mixedQueryLayer = query(hiddenStates)
            let mixedKeyLayer = key(hiddenStates)
            let mixedValueLayer = value(hiddenStates)

            let queryLayer = transposeForScores(mixedQueryLayer)
            let keyLayer = transposeForScores(mixedKeyLayer)
            let valueLayer = transposeForScores(mixedValueLayer)

            var attentionScores = queryLayer.matmul(keyLayer.transposed(permutation: 0, 1, 3, 2))
            attentionScores = attentionScores / sqrt(Float(attentionHeadSize))
            if let attentionMask {
                attentionScores = attentionScores + attentionMask
            }
            let attentionProbs = attentionScores.softmax(alongAxis: -1)
            var contextLayer = attentionProbs.matmul(valueLayer)
            contextLayer = contextLayer.transposed(permutation: [0, 2, 1, 3])
            let newContextLayerShape = contextLayer.shape.dropLast(2) + [allHeadSize]
            return contextLayer.reshaped(to: Array(newContextLayerShape))
        }
    }
}

extension BertEmbeddings {
    public struct Attention: Sendable {
        let selfAttention: BertEmbeddings.SelfAttention
        let output: BertEmbeddings.SelfOutput

        public init(
            selfAttention: BertEmbeddings.SelfAttention,
            output: BertEmbeddings.SelfOutput
        ) {
            self.selfAttention = selfAttention
            self.output = output
        }

        public func callAsFunction(
            hiddenStates: MLTensor,
            attentionMask: MLTensor?
        ) -> MLTensor {
            let selfOutputs = selfAttention(
                hiddenStates: hiddenStates,
                attentionMask: attentionMask
            )
            return output(
                hiddenStates: selfOutputs,
                inputTensor: hiddenStates
            )
        }
    }
}

extension BertEmbeddings {
    public struct Layer: Sendable {
        let attention: BertEmbeddings.Attention
        let intermediate: BertEmbeddings.Intermediate
        let output: BertEmbeddings.Output

        public init(
            attention: BertEmbeddings.Attention,
            intermediate: BertEmbeddings.Intermediate,
            output: BertEmbeddings.Output
        ) {
            self.attention = attention
            self.intermediate = intermediate
            self.output = output
        }

        public func callAsFunction(
            hiddenStates: MLTensor,
            attentionMask: MLTensor?
        ) -> MLTensor {
            let attentionOutput = attention(
                hiddenStates: hiddenStates,
                attentionMask: attentionMask
            )
            let intermediateOutput = intermediate(
                hiddenStates: attentionOutput
            )
            return output(
                hiddenStates: intermediateOutput,
                inputTensor: attentionOutput
            )
        }
    }
}

extension BertEmbeddings {
    public struct Encoder: Sendable {
        let layers: [BertEmbeddings.Layer]

        public init(layers: [BertEmbeddings.Layer]) {
            self.layers = layers
        }

        public func callAsFunction(
            hiddenStates: MLTensor,
            attentionMask: MLTensor?
        ) -> MLTensor {
            var hiddenStates = hiddenStates
            for layer in layers {
                hiddenStates = layer(
                    hiddenStates: hiddenStates,
                    attentionMask: attentionMask
                )
            }
            return hiddenStates
        }
    }
}

extension BertEmbeddings {
    public struct Model: Sendable {
        let embeddings: BertEmbeddings.Embeddings
        let encoder: BertEmbeddings.Encoder
        let pooler: BertEmbeddings.Pooler

        public init(
            embeddings: BertEmbeddings.Embeddings,
            encoder: BertEmbeddings.Encoder,
            pooler: BertEmbeddings.Pooler
        ) {
            self.embeddings = embeddings
            self.encoder = encoder
            self.pooler = pooler
        }

        public func callAsFunction(
            inputIds: MLTensor,
            tokenTypeIds: MLTensor? = nil,
            attentionMask: MLTensor? = nil
        ) -> (sequenceOutput: MLTensor, pooledOutput: MLTensor) {
            let embeddingOutput = embeddings(inputIds: inputIds, tokenTypeIds: tokenTypeIds)
            if let attentionMask {
                var attentionMask = attentionMask.expandingShape(at: 1, 1)
                attentionMask = (1.0 - attentionMask) * -10000.0
            }
            let encoderOutput = encoder(hiddenStates: embeddingOutput, attentionMask: attentionMask)
            let pooledOutput = pooler(encoderOutput)
            return (encoderOutput, pooledOutput)
        }
    }
}

extension BertEmbeddings {
    public struct ModelBundle: Sendable {
        public let model: BertEmbeddings.Model
        public let tokenizer: any Tokenizer

        public init(
            model: BertEmbeddings.Model,
            tokenizer: any Tokenizer
        ) {
            self.model = model
            self.tokenizer = tokenizer
        }

        public func encode<Scalar: MLShapedArrayScalar & MLTensorScalar>(
            _ text: String
        ) async -> [Scalar] {
            let encodedText = tokenizer.encode(text: text).map { Int32($0) }
            let inputIds = MLTensor(shape: [1, encodedText.count], scalars: encodedText)
            let result = model(inputIds: inputIds)
            return await result.sequenceOutput[0..., 0, 0...].shapedArray(of: Scalar.self).scalars
        }
    }
}
