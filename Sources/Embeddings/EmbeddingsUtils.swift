import CoreML
import Foundation
import Hub

func downloadModelFromHub(
    from hubRepoId: String,
    downloadBase: URL? = nil,
    useBackgroundSession: Bool = false,
    globs: [String] = Constants.modelGlobs
) async throws -> URL {
    let hubApi = HubApi(downloadBase: downloadBase, useBackgroundSession: useBackgroundSession)
    let repo = Hub.Repo(id: hubRepoId, type: .models)
    return try await hubApi.snapshot(
        from: repo,
        matching: globs
    )
}

enum EmbeddingsError: Error {
    case fileNotFound
    case invalidFile
}

enum Constants {
    static let modelGlobs = [
        "*.json",
        "*.safetensors",
        "*.py",
        "tokenizer.model",
        "sentencepiece*.model",
        "*.tiktoken",
        "*.txt",
    ]
}

func loadConfigFromFile<Config: Codable>(at url: URL) throws -> Config {
    let configData = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(Config.self, from: configData)
}

extension String {
    func replace(suffix: String, with string: String) -> String {
        guard hasSuffix(suffix) else { return self }
        return String(dropLast(suffix.count) + string)
    }
}

public struct TokenizerConfig {
    public let dataFileName: String
    public let tokenizerClass: String

    public init(
        dataFileName: String = "tokenizer.json",
        tokenizerClass: String = "BertTokenizer"
    ) {
        self.dataFileName = dataFileName
        self.tokenizerClass = tokenizerClass
    }
}

public struct ModelConfig {
    public let configFileName: String
    public let weightsFileName: String
    public let weightKeyTransform: ((String) -> String)

    public init(
        configFileName: String = "config.json",
        weightsFileName: String = "model.safetensors",
        weightKeyTransform: @escaping ((String) -> String) = { $0 }
    ) {
        self.configFileName = configFileName
        self.weightsFileName = weightsFileName
        self.weightKeyTransform = weightKeyTransform
    }
}

public struct LoadConfig {
    public let modelConfig: ModelConfig
    public let tokenizerConfig: TokenizerConfig?

    public init(
        modelConfig: ModelConfig = ModelConfig(),
        tokenizerConfig: TokenizerConfig? = nil
    ) {
        self.modelConfig = modelConfig
        self.tokenizerConfig = tokenizerConfig
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension LoadConfig {
    public static var googleBert: LoadConfig {
        LoadConfig(
            modelConfig: ModelConfig(
                weightKeyTransform: Bert.googleWeightsKeyTransform
            )
        )
    }

    public static var staticEmbeddings: LoadConfig {
        LoadConfig(
            modelConfig: ModelConfig(
                weightsFileName: "0_StaticEmbedding/model.safetensors"
            ),
            tokenizerConfig: TokenizerConfig(
                dataFileName: "0_StaticEmbedding/tokenizer.json",
                tokenizerClass: "BertTokenizer"
            )
        )
    }
}
