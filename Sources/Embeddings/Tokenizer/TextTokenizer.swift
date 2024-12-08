import Foundation
import SentencepieceTokenizer
@preconcurrency import Tokenizers

public protocol TextTokenizer: Sendable {
    func tokenizeText(_ text: String, maxLength: Int) throws -> [Int32]
    func tokenizeTextsPaddingToLongest(
        _ texts: [String], padTokenId: Int, maxLength: Int
    ) throws -> [[Int32]]
}

extension TextTokenizer {
    public func tokenizeTextsPaddingToLongest(
        _ texts: [String],
        padTokenId: Int,
        maxLength: Int
    ) throws -> [[Int32]] {
        var longest = 0
        var result = [[Int32]]()
        result.reserveCapacity(texts.count)
        for text in texts {
            let encoded = try tokenizeText(text, maxLength: maxLength)
            longest = max(longest, encoded.count)
            result.append(encoded)
        }
        return result.map {
            if $0.count < longest {
                return $0 + Array(repeating: Int32(padTokenId), count: longest - $0.count)
            } else {
                return $0
            }
        }
    }
}

public struct TokenizerWrapper {
    private let tokenizer: any Tokenizers.Tokenizer

    public init(_ tokenizer: any Tokenizers.Tokenizer) {
        self.tokenizer = tokenizer
    }
}

extension TokenizerWrapper: TextTokenizer {
    public func tokenizeText(_ text: String, maxLength: Int) throws -> [Int32] {
        var encoded = tokenizer.encode(text: text)
        if encoded.count > maxLength {
            encoded.removeLast(encoded.count - maxLength)
        }
        return encoded.map { Int32($0) }
    }
}
