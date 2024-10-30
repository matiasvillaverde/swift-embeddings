import ArgumentParser
import BertEmbeddings
import Foundation

struct BertCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bert",
        abstract: "Encode text using BERT model"
    )
    @Option var modelId: String = "sentence-transformers/all-MiniLM-L6-v2"
    @Option var text: String = "Text to encode"

    func run() async throws {
        let modelBundle = try await BertEmbeddings.loadModelBundle(from: modelId)
        let result: [Float32] = await modelBundle.encode(text)
        print(result)
    }
}
