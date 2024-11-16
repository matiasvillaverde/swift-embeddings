import ArgumentParser
import Foundation

@main
struct EmbeddingsCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Encode text using embedding model",
        subcommands: [BertCommand.self, ClipCommand.self]
    )
}
