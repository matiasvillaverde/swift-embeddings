import ArgumentParser
import Embeddings
import Foundation

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
struct StaticEmbeddingsCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "static-embeddings",
        abstract: "Encode text using StaticEmbeddings model"
    )
    @Option var modelId: String = "sentence-transformers/static-retrieval-mrl-en-v1"
    @Option var text: String = "Text to encode"

    func run() async throws {
        let modelBundle = try await StaticEmbeddings.loadModelBundle(
            from: modelId,
            loadConfig: LoadConfig.staticEmbeddings
        )
        let encoded = try modelBundle.encode(text)
        let result = await encoded.cast(to: Float.self).shapedArray(of: Float.self).scalars
        print(result)
    }
}
