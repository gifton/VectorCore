// VectorCore: Integration Patterns
//
// This example demonstrates how to integrate VectorCore into real applications

import VectorCore
import Foundation

// MARK: - 1. Building a Vector Database

protocol VectorStorage {
    associatedtype Vector: VectorType
    func store(_ vector: Vector, id: String, metadata: [String: Any]?)
    func retrieve(id: String) -> (vector: Vector, metadata: [String: Any]?)?
    func search(query: Vector, k: Int, filter: ((String, [String: Any]?) -> Bool)?) -> [(id: String, distance: Float)]
}

class InMemoryVectorStore<V: VectorType>: VectorStorage {
    typealias Vector = V
    
    private struct Entry {
        let id: String
        let vector: V
        let metadata: [String: Any]?
    }
    
    private var entries: [Entry] = []
    private let metric: any DistanceMetric
    
    init(metric: any DistanceMetric = CosineDistance()) {
        self.metric = metric
    }
    
    func store(_ vector: V, id: String, metadata: [String: Any]?) {
        entries.append(Entry(id: id, vector: vector, metadata: metadata))
    }
    
    func retrieve(id: String) -> (vector: V, metadata: [String: Any]?)? {
        guard let entry = entries.first(where: { $0.id == id }) else { return nil }
        return (entry.vector, entry.metadata)
    }
    
    func search(query: V, k: Int, filter: ((String, [String: Any]?) -> Bool)? = nil) -> [(id: String, distance: Float)] {
        let filtered = filter != nil ? entries.filter { filter!($0.id, $0.metadata) } : entries
        
        let results = filtered.map { entry in
            (id: entry.id, distance: metric.distance(query, entry.vector))
        }
        .sorted { $0.distance < $1.distance }
        .prefix(k)
        
        return Array(results)
    }
}

func vectorDatabaseExample() {
    print("=== Vector Database Example ===\n")
    
    // Create a vector store for embeddings
    let store = InMemoryVectorStore<Vector768>()
    
    // Simulate storing document embeddings
    let documents = [
        (id: "doc1", text: "Introduction to Swift programming", category: "programming"),
        (id: "doc2", text: "Machine learning with CoreML", category: "ml"),
        (id: "doc3", text: "Building iOS applications", category: "ios"),
        (id: "doc4", text: "Deep learning fundamentals", category: "ml"),
        (id: "doc5", text: "SwiftUI best practices", category: "ios")
    ]
    
    for doc in documents {
        let embedding = Vector768.random(in: -1...1).normalized()
        store.store(
            embedding,
            id: doc.id,
            metadata: ["text": doc.text, "category": doc.category]
        )
    }
    
    // Search for similar documents
    let queryEmbedding = Vector768.random(in: -1...1).normalized()
    
    print("Searching for similar documents...")
    let results = store.search(query: queryEmbedding, k: 3)
    
    for (index, result) in results.enumerated() {
        if let retrieved = store.retrieve(id: result.id) {
            let metadata = retrieved.metadata ?? [:]
            print("\(index + 1). \(result.id) (distance: \(String(format: "%.3f", result.distance)))")
            print("   Text: \(metadata["text"] ?? "N/A")")
            print("   Category: \(metadata["category"] ?? "N/A")")
        }
    }
    
    // Filtered search
    print("\nSearching only ML documents...")
    let mlResults = store.search(
        query: queryEmbedding,
        k: 2,
        filter: { _, metadata in
            (metadata?["category"] as? String) == "ml"
        }
    )
    
    for result in mlResults {
        if let retrieved = store.retrieve(id: result.id) {
            print("- \(result.id): \(retrieved.metadata?["text"] ?? "N/A")")
        }
    }
}

// MARK: - 2. Building an Embedding Service

protocol EmbeddingProvider {
    associatedtype Embedding: VectorType
    func embed(_ text: String) async throws -> Embedding
    func embedBatch(_ texts: [String]) async throws -> [Embedding]
}

// Mock embedding provider
class MockEmbeddingProvider: EmbeddingProvider {
    typealias Embedding = Vector768
    
    func embed(_ text: String) async throws -> Vector768 {
        // Simulate API call delay
        try await Task.sleep(nanoseconds: 10_000_000) // 10ms
        
        // In real implementation, this would call an ML model
        return Vector768.random(in: -1...1).normalized()
    }
    
    func embedBatch(_ texts: [String]) async throws -> [Vector768] {
        // Batch processing is often more efficient
        try await Task.sleep(nanoseconds: 50_000_000) // 50ms for batch
        
        return texts.map { _ in Vector768.random(in: -1...1).normalized() }
    }
}

class EmbeddingService {
    private let provider: any EmbeddingProvider
    private let cache: NSCache<NSString, AnyObject>
    
    init(provider: any EmbeddingProvider) {
        self.provider = provider
        self.cache = NSCache<NSString, AnyObject>()
        self.cache.countLimit = 1000
    }
    
    func embed(_ text: String) async throws -> any VectorType {
        let cacheKey = text as NSString
        
        // Check cache
        if let cached = cache.object(forKey: cacheKey) {
            return cached as! any VectorType
        }
        
        // Generate embedding
        let embedding = try await provider.embed(text)
        
        // Cache result
        cache.setObject(embedding as AnyObject, forKey: cacheKey)
        
        return embedding
    }
}

func embeddingServiceExample() async {
    print("\n\n=== Embedding Service Example ===\n")
    
    let provider = MockEmbeddingProvider()
    let service = EmbeddingService(provider: provider)
    
    // Embed single text
    let text = "Understanding vector embeddings"
    print("Embedding single text...")
    
    let start = Date()
    let embedding = try! await service.embed(text)
    let elapsed = Date().timeIntervalSince(start)
    
    print("Generated embedding with dimension: \(embedding.scalarCount)")
    print("Time taken: \(String(format: "%.3f", elapsed))s")
    
    // Cached access
    let startCached = Date()
    let cachedEmbedding = try! await service.embed(text)
    let elapsedCached = Date().timeIntervalSince(startCached)
    
    print("\nCached access time: \(String(format: "%.3f", elapsedCached))s")
    print("Speedup: \(String(format: "%.0fx", elapsed / elapsedCached))")
    
    // Batch embedding
    let texts = [
        "Vector databases are powerful",
        "Machine learning needs embeddings",
        "Swift is great for performance"
    ]
    
    print("\nBatch embedding \(texts.count) texts...")
    let batchStart = Date()
    let batchEmbeddings = try! await provider.embedBatch(texts)
    let batchElapsed = Date().timeIntervalSince(batchStart)
    
    print("Batch time: \(String(format: "%.3f", batchElapsed))s")
    print("Average per text: \(String(format: "%.3f", batchElapsed / Double(texts.count)))s")
}

// MARK: - 3. Building a Recommendation Engine

class RecommendationEngine<Item: Hashable> {
    private var itemVectors: [Item: any VectorType] = [:]
    private var userProfiles: [String: any VectorType] = [:]
    
    func addItem(_ item: Item, vector: any VectorType) {
        itemVectors[item] = vector
    }
    
    func updateUserProfile(userId: String, interactedItems: [Item], weights: [Float]? = nil) {
        guard !interactedItems.isEmpty else { return }
        
        // Weighted average of item vectors
        let vectors = interactedItems.compactMap { itemVectors[$0] }
        guard !vectors.isEmpty else { return }
        
        let dimension = vectors[0].scalarCount
        var profile = [Float](repeating: 0, count: dimension)
        
        for (index, vector) in vectors.enumerated() {
            let weight = weights?[index] ?? 1.0
            let array = vector.toArray()
            for i in 0..<dimension {
                profile[i] += array[i] * weight
            }
        }
        
        // Normalize
        let totalWeight = weights?.reduce(0, +) ?? Float(vectors.count)
        profile = profile.map { $0 / totalWeight }
        
        // Store as appropriate vector type
        userProfiles[userId] = DynamicVector(profile).normalized()
    }
    
    func recommend(for userId: String, count: Int, excludeInteracted: Set<Item> = []) -> [(item: Item, score: Float)] {
        guard let userProfile = userProfiles[userId] else { return [] }
        
        let recommendations = itemVectors
            .filter { !excludeInteracted.contains($0.key) }
            .map { item, vector in
                let score = userProfile.toArray().enumerated().reduce(Float(0)) { sum, pair in
                    sum + pair.element * vector.toArray()[pair.offset]
                }
                return (item: item, score: score)
            }
            .sorted { $0.score > $1.score }
            .prefix(count)
        
        return Array(recommendations)
    }
}

func recommendationEngineExample() {
    print("\n\n=== Recommendation Engine Example ===\n")
    
    let engine = RecommendationEngine<String>()
    
    // Add items with feature vectors
    let items = [
        ("iPhone 15", "electronics,apple,smartphone"),
        ("MacBook Pro", "electronics,apple,laptop"),
        ("Swift Programming Book", "books,programming,swift"),
        ("AirPods", "electronics,apple,audio"),
        ("iOS Development Course", "courses,programming,ios"),
        ("Samsung Galaxy", "electronics,samsung,smartphone"),
        ("Python Book", "books,programming,python")
    ]
    
    for (item, tags) in items {
        // Simple feature vector based on tags
        let vector = createFeatureVector(from: tags)
        engine.addItem(item, vector: vector)
    }
    
    // User interactions
    let user1Interactions = ["iPhone 15", "MacBook Pro", "AirPods"]
    let user1Weights = [1.0, 0.8, 0.6] // More recent interactions weighted higher
    
    engine.updateUserProfile(
        userId: "user1",
        interactedItems: user1Interactions,
        weights: user1Weights
    )
    
    // Get recommendations
    let recommendations = engine.recommend(
        for: "user1",
        count: 3,
        excludeInteracted: Set(user1Interactions)
    )
    
    print("Recommendations for user1:")
    for (index, rec) in recommendations.enumerated() {
        print("\(index + 1). \(rec.item) (score: \(String(format: "%.3f", rec.score)))")
    }
}

// MARK: - 4. Real-time Similarity Search

class RealtimeSimilaritySearch {
    private let batchOps = SyncBatchOperations()
    private var indexedVectors: [Vector256] = []
    private var metadata: [[String: Any]] = []
    
    func index(_ vectors: [Vector256], metadata: [[String: Any]]) {
        self.indexedVectors.append(contentsOf: vectors)
        self.metadata.append(contentsOf: metadata)
    }
    
    func search(
        query: Vector256,
        k: Int,
        threshold: Float? = nil,
        useApproximateSearch: Bool = false
    ) -> [(index: Int, distance: Float, metadata: [String: Any])] {
        
        if useApproximateSearch && indexedVectors.count > 1000 {
            // Use approximate search for large datasets
            return approximateSearch(query: query, k: k, threshold: threshold)
        } else {
            // Exact search for smaller datasets
            return exactSearch(query: query, k: k, threshold: threshold)
        }
    }
    
    private func exactSearch(
        query: Vector256,
        k: Int,
        threshold: Float?
    ) -> [(index: Int, distance: Float, metadata: [String: Any])] {
        
        let distances = batchOps.batchDistance(
            from: query,
            to: indexedVectors,
            using: CosineDistance()
        )
        
        var results = distances.enumerated()
            .map { (index: $0, distance: $1, metadata: metadata[$0]) }
        
        if let threshold = threshold {
            results = results.filter { $0.distance <= threshold }
        }
        
        return Array(results.sorted { $0.distance < $1.distance }.prefix(k))
    }
    
    private func approximateSearch(
        query: Vector256,
        k: Int,
        threshold: Float?
    ) -> [(index: Int, distance: Float, metadata: [String: Any])] {
        
        // Simple approximate search using sampling
        let sampleRate = 0.1
        let sampleSize = Int(Double(indexedVectors.count) * sampleRate)
        let sampledIndices = (0..<indexedVectors.count).shuffled().prefix(sampleSize)
        
        let sampledResults = sampledIndices.map { index in
            let distance = CosineDistance().distance(query, indexedVectors[index])
            return (index: index, distance: distance, metadata: metadata[index])
        }
        
        return Array(sampledResults.sorted { $0.distance < $1.distance }.prefix(k))
    }
}

// MARK: - Helper Functions

func createFeatureVector(from tags: String) -> Vector128 {
    // Simple hash-based feature vector
    var features = [Float](repeating: 0, count: 128)
    
    for tag in tags.split(separator: ",") {
        let hash = abs(tag.hashValue)
        let index = hash % 128
        features[index] = 1.0
    }
    
    return Vector128(features).normalized()
}

// MARK: - Main

@main
struct IntegrationExamples {
    static func main() async {
        // Synchronous examples
        vectorDatabaseExample()
        recommendationEngineExample()
        
        // Asynchronous example
        await embeddingServiceExample()
        
        print("\n\n=== Integration Patterns Summary ===")
        print("""
        1. Vector Database:
           - Store vectors with metadata
           - Efficient similarity search
           - Support for filtered queries
        
        2. Embedding Service:
           - Async/await support
           - Caching for performance
           - Batch processing
        
        3. Recommendation Engine:
           - User profile generation
           - Item similarity scoring
           - Personalized recommendations
        
        4. Real-time Search:
           - Exact vs approximate search
           - Threshold-based filtering
           - Scalable architecture
        
        VectorCore provides the foundation for building
        sophisticated vector-based applications!
        """)
    }
}