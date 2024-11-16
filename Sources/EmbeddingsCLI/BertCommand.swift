import ArgumentParser
import Embeddings
import Foundation

struct BertCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bert",
        abstract: "Encode text using BERT model"
    )
    @Option var modelId: String = "sentence-transformers/all-MiniLM-L6-v2"
    @Option var text: String = "Text to encode"
    @Option var maxLength: Int = 512

    func run() async throws {
        let modelBundle = try await Bert.loadModelBundle(from: modelId)
        let encoded = modelBundle.encode(text, maxLength: maxLength)
        let result = await encoded.cast(to: Float.self).shapedArray(of: Float.self).scalars
        print(result)
    }
}
