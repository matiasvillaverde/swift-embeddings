import ArgumentParser
import Bert
import CoreML
import Foundation
import Safetensors

@main
struct EmbeddingsCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Encode text using BERT model"
    )
    @Option var modelId: String = "sentence-transformers/all-MiniLM-L6-v2"
    @Option var text: String = "Text to encode"

    func run() async throws {
        let modelBundle = try await Bert.loadModelBundle(from: modelId)
        let result: [Float32] = await modelBundle.encode(text)
        print(result)
    }
}
