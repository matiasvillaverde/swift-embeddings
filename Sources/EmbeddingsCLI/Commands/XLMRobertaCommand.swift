import ArgumentParser
import Embeddings
import Foundation

struct XLMRobertaCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "xlm-roberta",
        abstract: "Encode text using XLMRoberta model"
    )
    @Option var modelId: String = "tomaarsen/xlm-roberta-base-multilingual-en-ar-fr-de-es-tr-it"
    @Option var text: String = "Text to encode"
    @Option var maxLength: Int = 128

    func run() async throws {
        let modelBundle = try await XLMRoberta.loadModelBundle(from: modelId)
        let encoded = try modelBundle.encode(text, maxLength: maxLength)
        let result = await encoded.cast(to: Float.self).shapedArray(of: Float.self).scalars
        print(result)
    }
}
