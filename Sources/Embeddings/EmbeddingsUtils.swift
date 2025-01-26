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

public struct LoadConfig {
    public let modelFileName: String
    public let configFileName: String
    public let weightKeyTransform: ((String) -> String)

    public init(
        modelFileName: String = "model.safetensors",
        configFileName: String = "config.json",
        weightKeyTransform: @escaping ((String) -> String) = { $0 }
    ) {
        self.modelFileName = modelFileName
        self.configFileName = configFileName
        self.weightKeyTransform = weightKeyTransform
    }
}
