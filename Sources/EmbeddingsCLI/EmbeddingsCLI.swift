import ArgumentParser
import BertEmbeddings
import Foundation

@main
struct EmbeddingsCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Encode text using embedding model",
        subcommands: [BertCommand.self]
    )
}
