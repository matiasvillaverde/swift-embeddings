import Foundation
import SentencepieceTokenizer
import Synchronization

final class XLMRobetaTokenizer: Sendable {
    /// TODO: Make `SentencepieceTokenizer` conform to `Sendable`
    private let tokenizer: Mutex<SentencepieceTokenizer>
    private let addedTokens: [String: Int]

    init(tokenizerModelUrl: URL, addedTokens: [String: Int]) throws {
        let sentencepieceTokenizer = try SentencepieceTokenizer(modelPath: tokenizerModelUrl.path)
        self.tokenizer = Mutex(sentencepieceTokenizer)
        self.addedTokens = addedTokens
    }

    func tokenize(_ text: String, maxLength: Int, padToLength: Int? = nil) throws -> [Int] {
        precondition(
            maxLength >= 2, "maxLength must be at least 2 to accommodate BOS and EOS tokens")
        let tokenIds = try tokenizer.withLock {
            try $0.encode(text)
        }
        var result = [bosTokenId]
        // Truncate to maxLength - 2 to make space for bos and eos tokens
        result.append(contentsOf: tokenIds.prefix(maxLength - 2))
        result.append(eosTokenId)
        // If padToLength is provided, pad the tokenIds with padTokenId
        if let padToLength, padToLength <= maxLength {
            precondition(padToLength - 2 >= 0, "padToLength must be greater than or equal to 2")
            result.append(
                contentsOf: Array(repeating: padTokenId, count: padToLength - result.count))
        }
        return result
    }

    var bosTokenId: Int {
        if let bosTokenId = addedTokens["<s>"] {
            return bosTokenId
        }
        return tokenizer.withLock {
            $0.bosTokenId
        }
    }

    var eosTokenId: Int {
        if let eosTokenId = addedTokens["</s>"] {
            return eosTokenId
        }
        return tokenizer.withLock {
            $0.eosTokenId
        }
    }

    var padTokenId: Int {
        if let padTokenId = addedTokens["<pad>"] {
            return padTokenId
        }
        return tokenizer.withLock {
            $0.padTokenId
        }
    }

    var unkTokenId: Int {
        if let unkTokenId = addedTokens["<unk>"] {
            return unkTokenId
        }
        return tokenizer.withLock {
            $0.unkTokenId
        }
    }
}

extension XLMRobetaTokenizer: TextTokenizer {
    func tokenizeText(_ text: String, maxLength: Int) throws -> [Int32] {
        try tokenize(text, maxLength: maxLength, padToLength: nil).map { Int32($0) }
    }
}
