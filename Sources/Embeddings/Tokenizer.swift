import Foundation
@preconcurrency import Tokenizers

public protocol TextTokenizer: Sendable {
    func tokenize(_ text: String, maxLength: Int) -> [Int32]
    func tokenizePaddingToLongest(
        _ texts: [String], padTokenId: Int, maxLength: Int
    ) -> [[Int32]]
}

enum TextTokenizerType {
    case transformers(any Tokenizers.Tokenizer)
    case clip(ClipTokenizer)
}

extension TextTokenizerType: TextTokenizer {
    public func tokenize(_ text: String, maxLength: Int) -> [Int32] {
        switch self {
        case .transformers(let tokenizer):
            var encoded = tokenizer.encode(text: text)
            if encoded.count > maxLength {
                encoded.removeLast(encoded.count - maxLength)
            }
            return encoded.map { Int32($0) }
        case .clip(let tokenizer):
            return tokenizer.tokenize(text, maxLength: maxLength).map { Int32($0) }
        }
    }

    public func tokenizePaddingToLongest(
        _ texts: [String],
        padTokenId: Int,
        maxLength: Int
    ) -> [[Int32]] {
        var longest = 0
        var result = [[Int32]]()
        result.reserveCapacity(texts.count)
        for text in texts {
            let encoded = tokenize(text, maxLength: maxLength)
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
