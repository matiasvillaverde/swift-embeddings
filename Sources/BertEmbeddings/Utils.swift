import CoreML
import Foundation
import Hub
import MLTensorNN
import Safetensors
@preconcurrency import Tokenizers

extension BertEmbeddings {
    public static func loadConfig(at url: URL) throws -> BertEmbeddings.ModelConfig {
        let configData = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(BertEmbeddings.ModelConfig.self, from: configData)
    }
}

extension BertEmbeddings {
    public static func loadModelBundle(
        from hubRepoId: String,
        downloadBase: URL? = nil,
        useBackgroundSession: Bool = false
    ) async throws -> BertEmbeddings.ModelBundle {
        let hubApi = HubApi(downloadBase: downloadBase, useBackgroundSession: useBackgroundSession)
        let repo = Hub.Repo(id: hubRepoId, type: .models)
        let modelUrl = try await hubApi.snapshot(from: repo)
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelUrl)
        // NOTE: just `safetensors` support for now
        let weightsUrl = modelUrl.appendingPathComponent("model.safetensors")
        let configUrl = modelUrl.appendingPathComponent("config.json")
        let config = try BertEmbeddings.loadConfig(at: configUrl)
        let model = try BertEmbeddings.loadModel(weightsUrl: weightsUrl, config: config)
        return BertEmbeddings.ModelBundle(model: model, tokenizer: tokenizer)
    }
}

extension BertEmbeddings {
    public static func loadModel(
        weightsUrl: URL,
        config: BertEmbeddings.ModelConfig
    ) throws -> BertEmbeddings.Model {
        let safetensors = try Safetensors.read(at: weightsUrl)
        let pooler = try BertEmbeddings.Pooler(
            dense: MLTensorNN.linear(
                weight: safetensors.mlTensor(forKey: "pooler.dense.weight"),
                bias: safetensors.mlTensor(forKey: "pooler.dense.bias")))

        let wordEmbeddings = try MLTensorNN.embedding(
            weight: safetensors.mlTensor(
                forKey: "embeddings.word_embeddings.weight"))

        let tokenTypeEmbeddings = try MLTensorNN.embedding(
            weight: safetensors.mlTensor(
                forKey: "embeddings.token_type_embeddings.weight"))

        let positionEmbeddings = try MLTensorNN.embedding(
            weight: safetensors.mlTensor(
                forKey: "embeddings.position_embeddings.weight"))

        let layerNorm = try MLTensorNN.layerNorm(
            weight: safetensors.mlTensor(
                forKey: "embeddings.LayerNorm.weight"),
            bias: safetensors.mlTensor(
                forKey: "embeddings.LayerNorm.bias"),
            epsilon: config.layerNormEps)

        let embeddings = BertEmbeddings.Embeddings(
            wordEmbeddings: wordEmbeddings,
            positionEmbeddings: positionEmbeddings,
            tokenTypeEmbeddings: tokenTypeEmbeddings,
            layerNorm: layerNorm)

        var layers = [BertEmbeddings.Layer]()
        for layer in 0..<config.numHiddenLayers {
            let bertSelfAttention = try BertEmbeddings.SelfAttention(
                query: MLTensorNN.linear(
                    weight: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.self.query.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.self.query.bias")),
                key: MLTensorNN.linear(
                    weight: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.self.key.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.self.key.bias")),
                value: MLTensorNN.linear(
                    weight: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.self.value.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.self.value.bias")),
                numAttentionHeads: config.numAttentionHeads,
                attentionHeadSize: config.hiddenSize / config.numAttentionHeads,
                allHeadSize: config.numAttentionHeads
                    * (config.hiddenSize / config.numAttentionHeads)
            )
            let bertSelfOutput = try BertEmbeddings.SelfOutput(
                dense: MLTensorNN.linear(
                    weight: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.output.dense.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.output.dense.bias")),
                layerNorm: MLTensorNN.layerNorm(
                    weight: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.output.LayerNorm.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.output.LayerNorm.bias"),
                    epsilon: config.layerNormEps)
            )
            let bertAttention = BertEmbeddings.Attention(
                selfAttention: bertSelfAttention,
                output: bertSelfOutput
            )
            let bertIntermediate = try BertEmbeddings.Intermediate(
                dense: MLTensorNN.linear(
                    weight: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).intermediate.dense.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).intermediate.dense.bias"))
            )
            let bertOutput = try BertEmbeddings.Output(
                dense: MLTensorNN.linear(
                    weight: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).output.dense.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).output.dense.bias")),
                layerNorm: MLTensorNN.layerNorm(
                    weight: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).output.LayerNorm.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).output.LayerNorm.bias"),
                    epsilon: config.layerNormEps))

            let bertLayer = BertEmbeddings.Layer(
                attention: bertAttention,
                intermediate: bertIntermediate,
                output: bertOutput
            )
            layers.append(bertLayer)
        }
        return BertEmbeddings.Model(
            embeddings: embeddings,
            encoder: BertEmbeddings.Encoder(layers: layers),
            pooler: pooler)
    }
}
