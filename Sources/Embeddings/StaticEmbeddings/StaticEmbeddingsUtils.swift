import CoreML
import Foundation
import Hub
import MLTensorUtils
import Safetensors
@preconcurrency import Tokenizers

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension StaticEmbeddings {
    public static func loadModelBundle(
        from hubRepoId: String,
        downloadBase: URL? = nil,
        useBackgroundSession: Bool = false,
        loadConfig: LoadConfig = LoadConfig()
    ) async throws -> StaticEmbeddings.ModelBundle {
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
    ) async throws -> StaticEmbeddings.ModelBundle {
        let tokenizer =
            if let tokenizerConfig = loadConfig.tokenizerConfig {
                try AutoTokenizer.from(
                    tokenizerDataFile: modelFolder.appendingPathComponent(
                        tokenizerConfig.dataFileName),
                    tokenizerClass: tokenizerConfig.tokenizerClass
                )
            } else {
                try await AutoTokenizer.from(modelFolder: modelFolder)
            }
        let weightsUrl = modelFolder.appendingPathComponent(loadConfig.modelConfig.weightsFileName)
        let model = try StaticEmbeddings.loadModel(
            weightsUrl: weightsUrl,
            loadConfig: loadConfig
        )
        return StaticEmbeddings.ModelBundle(
            model: model,
            tokenizer: TokenizerWrapper(tokenizer)
        )
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension StaticEmbeddings {
    public static func loadModel(
        weightsUrl: URL,
        loadConfig: LoadConfig = LoadConfig()
    ) throws -> StaticEmbeddings.Model {
        let data = try Safetensors.read(at: weightsUrl)
        let embeddings = try data.mlTensor(
            forKey: loadConfig.modelConfig.weightKeyTransform("embedding.weight"))
        return StaticEmbeddings.Model(embeddings: embeddings)
    }
}

extension AutoTokenizer {
    static func from(
        tokenizerDataFile: URL,
        tokenizerClass: String
    ) throws -> any Tokenizer {
        let data = try Data(contentsOf: tokenizerDataFile)
        let parsedData = try JSONSerialization.jsonObject(with: data, options: [])
        guard let tokenizerData = parsedData as? [NSString: Any] else {
            throw EmbeddingsError.invalidFile
        }
        return try AutoTokenizer.from(
            tokenizerConfig: Config(["tokenizerClass": tokenizerClass]),
            tokenizerData: Config(tokenizerData)
        )
    }
}
