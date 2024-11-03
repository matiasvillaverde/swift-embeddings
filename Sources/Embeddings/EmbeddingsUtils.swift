import CoreML
import Foundation
@preconcurrency import Tokenizers

extension Tokenizer {
    func tokenizeWithPadding(
        _ texts: [String],
        padTokenId: Int32,
        maxSequenceLength: Int
    ) -> [[Int32]] {
        var longest = 0
        var result = [[Int32]]()
        for text in texts {
            let encoded = tokenize(text, maxSequenceLength: maxSequenceLength)
            longest = max(longest, encoded.count)
            result.append(encoded)
        }
        return result.map {
            if $0.count < longest {
                return $0 + Array(repeating: padTokenId, count: longest - $0.count)
            } else {
                return $0
            }
        }
    }

    func tokenize(
        _ text: String,
        maxSequenceLength: Int
    ) -> [Int32] {
        var encoded = self(text).map { Int32($0) }
        if encoded.count > maxSequenceLength {
            encoded.removeLast(encoded.count - maxSequenceLength)
        }
        return encoded
    }
}
