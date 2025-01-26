import CoreML
import Foundation
import Hub
import MLTensorUtils
import Safetensors
@preconcurrency import Tokenizers

extension Model2Vec {
    public static func loadConfig(at url: URL) throws -> Model2Vec.ModelConfig {
        try loadConfigFromFile(at: url)
    }
}

extension Model2Vec {
    public static func loadModelBundle(
        from hubRepoId: String,
        downloadBase: URL? = nil,
        useBackgroundSession: Bool = false,
        loadConfig: LoadConfig = LoadConfig()
    ) async throws -> Model2Vec.ModelBundle {
        let modelFolder = try await downloadModelFromHub(
            from: hubRepoId,
            downloadBase: downloadBase,
            useBackgroundSession: useBackgroundSession
        )
        return try await loadModelBundle(
            from: modelFolder,
            loadConfig: loadConfig
        )
    }

    public static func loadModelBundle(
        from modelFolder: URL,
        loadConfig: LoadConfig = LoadConfig()
    ) async throws -> Model2Vec.ModelBundle {
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)
        let weightsUrl = modelFolder.appendingPathComponent(loadConfig.modelFileName)
        let configUrl = modelFolder.appendingPathComponent(loadConfig.configFileName)
        let config = try Model2Vec.loadConfig(at: configUrl)
        let model = try Model2Vec.loadModel(
            weightsUrl: weightsUrl,
            normalize: config.normalize ?? false,
            loadConfig: loadConfig
        )
        return Model2Vec.ModelBundle(
            model: model,
            tokenizer: TokenizerWrapper(tokenizer)
        )
    }
}

extension Model2Vec {
    public static func loadModel(
        weightsUrl: URL,
        normalize: Bool,
        loadConfig: LoadConfig = LoadConfig()
    ) throws -> Model2Vec.Model {
        let data = try Safetensors.read(at: weightsUrl)
        let embeddings = try data.mlTensor(forKey: loadConfig.weightKeyTransform("embeddings"))
        return Model2Vec.Model(embeddings: embeddings, normalize: normalize)
    }
}
