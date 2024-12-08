import CoreML
import Foundation
import Hub
import MLTensorUtils
import Safetensors

extension Clip {
    public static func loadConfig(at url: URL) throws -> Clip.ModelConfig {
        try loadConfigFromFile(at: url)
    }
}

extension Clip {
    public static func loadModelBundle(
        from hubRepoId: String,
        downloadBase: URL? = nil,
        useBackgroundSession: Bool = false
    ) async throws -> Clip.ModelBundle {
        let modelFolder = try await downloadModelFromHub(
            from: hubRepoId,
            downloadBase: downloadBase,
            useBackgroundSession: useBackgroundSession
        )
        return try await loadModelBundle(from: modelFolder)
    }

    public static func loadModelBundle(from modelFolder: URL) async throws -> Clip.ModelBundle {
        let tokenizer = try loadClipTokenizer(at: modelFolder)
        let weightsUrl = modelFolder.appendingPathComponent("model.safetensors")
        let configUrl = modelFolder.appendingPathComponent("config.json")
        let config = try Clip.loadConfig(at: configUrl)
        // TODO: implement vision model loading
        let textModel = try Clip.loadModel(weightsUrl: weightsUrl, config: config)
        return Clip.ModelBundle(textModel: textModel, tokenizer: tokenizer)
    }
}

extension Clip {
    public static func loadModel(
        weightsUrl: URL,
        config: Clip.ModelConfig
    ) throws -> Clip.TextModel {
        let safetensors = try Safetensors.read(at: weightsUrl)
        let embeddings = try Clip.Embeddings(
            tokenEmbedding: MLTensorUtils.embedding(
                weight: safetensors.mlTensor(
                    forKey: "text_model.embeddings.token_embedding.weight")),
            positionEmbeddingWeight: safetensors.mlTensor(
                forKey: "text_model.embeddings.position_embedding.weight"))
        var encoderLayers = [Clip.EncoderLayer]()
        encoderLayers.reserveCapacity(config.textConfig.numHiddenLayers)
        for i in 0..<config.textConfig.numHiddenLayers {
            let attention = try Clip.Attention(
                qProj: MLTensorUtils.linear(
                    weight: safetensors.mlTensor(
                        forKey: "text_model.encoder.layers.\(i).self_attn.q_proj.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "text_model.encoder.layers.\(i).self_attn.q_proj.bias")),
                kProj: MLTensorUtils.linear(
                    weight: safetensors.mlTensor(
                        forKey: "text_model.encoder.layers.\(i).self_attn.k_proj.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "text_model.encoder.layers.\(i).self_attn.k_proj.bias")),
                vProj: MLTensorUtils.linear(
                    weight: safetensors.mlTensor(
                        forKey: "text_model.encoder.layers.\(i).self_attn.v_proj.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "text_model.encoder.layers.\(i).self_attn.v_proj.bias")),
                outProj: MLTensorUtils.linear(
                    weight: safetensors.mlTensor(
                        forKey: "text_model.encoder.layers.\(i).self_attn.out_proj.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "text_model.encoder.layers.\(i).self_attn.out_proj.bias")),
                numHeads: config.textConfig.numAttentionHeads)
            let mlp = try Clip.MLP(
                fc1: MLTensorUtils.linear(
                    weight: safetensors.mlTensor(
                        forKey: "text_model.encoder.layers.\(i).mlp.fc1.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "text_model.encoder.layers.\(i).mlp.fc1.bias")),
                fc2: MLTensorUtils.linear(
                    weight: safetensors.mlTensor(
                        forKey: "text_model.encoder.layers.\(i).mlp.fc2.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "text_model.encoder.layers.\(i).mlp.fc2.bias")))
            let layer = try Clip.EncoderLayer(
                selfAttnention: attention,
                mlp: mlp,
                layerNorm1: MLTensorUtils.layerNorm(
                    weight: safetensors.mlTensor(
                        forKey: "text_model.encoder.layers.\(i).layer_norm1.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "text_model.encoder.layers.\(i).layer_norm1.bias"),
                    epsilon: config.textConfig.layerNormEps),
                layerNorm2: MLTensorUtils.layerNorm(
                    weight: safetensors.mlTensor(
                        forKey: "text_model.encoder.layers.\(i).layer_norm2.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "text_model.encoder.layers.\(i).layer_norm2.bias"),
                    epsilon: config.textConfig.layerNormEps))
            encoderLayers.append(layer)
        }
        let encoder = Clip.Encoder(layers: encoderLayers)
        return try Clip.TextModel(
            embeddings: embeddings,
            encoder: encoder,
            finalLayerNorm: MLTensorUtils.layerNorm(
                weight: safetensors.mlTensor(
                    forKey: "text_model.final_layer_norm.weight"),
                bias: safetensors.mlTensor(
                    forKey: "text_model.final_layer_norm.bias"),
                epsilon: config.textConfig.layerNormEps),
            textProjection: MLTensorUtils.linear(
                weight: safetensors.mlTensor(
                    forKey: "text_projection.weight")))
    }
}
