import Foundation
import Synchronization

extension Regex: @retroactive @unchecked Sendable {}

final class ClipTokenizer: Sendable {
    private let bpeRanks: [Pair<String>: Int]
    private let vocab: [String: Int]
    private let splitStringPattern: Regex<AnyRegexOutput>
    private let emptyStringPattern: Regex<AnyRegexOutput>
    private let bos: String
    private let bosToken: Int
    private let eos: String
    private let eosToken: Int
    private let cache: Mutex<[String: [String]]>

    init(
        bpeRanks: [Pair<String>: Int],
        vocab: [String: Int]
    ) throws {
        self.bpeRanks = bpeRanks
        self.vocab = vocab
        self.splitStringPattern = try Regex(
            "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"
        )
        self.emptyStringPattern = try Regex("\\s+")
        self.bos = "<|startoftext|>"
        self.bosToken = vocab[bos]!
        self.eos = "<|endoftext|>"
        self.eosToken = vocab[eos]!
        self.cache = Mutex([:])
    }

    func tokenize(_ text: String, maxLength: Int, padToLength: Int? = nil) -> [Int] {
        precondition(
            maxLength >= 2, "maxLength must be at least 2 to accommodate BOS and EOS tokens")
        let cleanText = text.lowercased().replacing(emptyStringPattern, with: " ")
        let tokens = cleanText.ranges(of: splitStringPattern).map { String(cleanText[$0]) }
        let bpeTokens = tokens.flatMap { bpe($0) }
        let tokenIds = bpeTokens.map { vocab[$0]! }
        var result = [bosToken]
        // Truncate to maxLength - 2 to make space for bos and eos tokens
        result.append(contentsOf: tokenIds.prefix(maxLength - 2))
        result.append(eosToken)
        // If padToLength is provided, pad the tokenIds with 0s
        if let padToLength, padToLength <= maxLength {
            precondition(padToLength - 2 >= 0, "padToLength must be greater than or equal to 2")
            result.append(
                contentsOf: Array(repeating: 0, count: padToLength - result.count))
        }
        return result
    }

    private func bpe(_ text: String) -> [String] {
        let cachedValue = cache.withLock {
            $0[text]
        }
        if let cachedValue {
            return cachedValue
        }
        var unigrams = text.dropLast().map { String($0) } + ["\(text.suffix(1))</w>"]
        var uniqueBigrams = uniquePairs(from: unigrams)
        while !uniqueBigrams.isEmpty {
            guard let lowestMergePair = findLowestMergePair(in: uniqueBigrams, using: bpeRanks)
            else {
                break
            }
            var newUnigrams = [String]()
            var skip = false
            for (first, second) in zip(unigrams, unigrams.dropFirst()) {
                if skip {
                    skip = false
                    continue
                }
                let pair = Pair(first: first, second: second)
                if pair == lowestMergePair {
                    newUnigrams.append(first + second)
                    skip = true
                } else {
                    newUnigrams.append(first)
                }
            }

            if !skip {
                newUnigrams.append(unigrams.last!)
            }

            unigrams = newUnigrams
            uniqueBigrams = uniquePairs(from: unigrams)
        }

        cache.withLock {
            $0[text] = unigrams
        }
        return unigrams
    }
}

extension ClipTokenizer: TextTokenizer {
    func tokenizeText(_ text: String, maxLength: Int) throws -> [Int32] {
        tokenize(text, maxLength: maxLength, padToLength: nil).map { Int32($0) }
    }
}

func loadClipTokenizer(at url: URL) throws -> ClipTokenizer {
    let mergesData = try String(
        contentsOf: url.appendingPathComponent("merges.txt"),
        encoding: .utf8)
    let merges = mergesData.split(separator: "\n").dropFirst()
    var bpeRanks = [Pair<String>: Int]()
    for (index, line) in merges.enumerated() {
        let pair = line.trimmingCharacters(in: .whitespacesAndNewlines).components(separatedBy: " ")
        if pair.count != 2 {
            fatalError("Malformed data on line \(line)")
        }
        bpeRanks[Pair(first: pair[0], second: pair[1])] = index
    }
    let vocabData = try JSONSerialization.jsonObject(
        with: Data(contentsOf: url.appendingPathComponent("vocab.json")))
    guard let vocab = vocabData as? [String: Int] else {
        fatalError("Malformed vocab data")
    }
    return try ClipTokenizer(bpeRanks: bpeRanks, vocab: vocab)
}

func uniquePairs(from arr: [String]) -> Set<Pair<String>> {
    Set(zip(arr, arr.dropFirst()).map { Pair($0) })
}

func findLowestMergePair(
    in keySet: Set<Pair<String>>,
    using bpeRanks: [Pair<String>: Int]
) -> Pair<String>? {
    var pair: Pair<String>?
    var index: Int?
    for key in keySet {
        guard let mergeIndex = bpeRanks[key] else {
            continue
        }
        if let currentIndex = index {
            if mergeIndex < currentIndex {
                index = mergeIndex
                pair = key
            }
        } else {
            index = mergeIndex
            pair = key
        }
    }
    guard let pair else {
        return nil
    }
    return pair
}

struct Pair<T> {
    let first: T
    let second: T

    init(first: T, second: T) {
        self.first = first
        self.second = second
    }
}

extension Pair {
    init(_ pair: (T, T)) {
        self.init(first: pair.0, second: pair.1)
    }
}

extension Pair: Equatable where T: Equatable {}
extension Pair: Hashable where T: Hashable {}
extension Pair: Sendable where T: Sendable {}
