// VectorCore: Machine Learning Use Cases
//
// This example demonstrates VectorCore in ML/AI applications

import VectorCore
import Foundation

// MARK: - 1. Text Embeddings

struct TextEmbedding {
    let text: String
    let vector: Vector768  // Common dimension for BERT-like models
    let metadata: [String: Any]
}

func textEmbeddingExample() {
    print("=== Text Embeddings Example ===\n")
    
    // Simulate embeddings from a language model
    let documents = [
        "VectorCore is a high-performance vector library for Swift",
        "Machine learning models use embeddings to represent text",
        "Swift is a powerful programming language for Apple platforms",
        "Embeddings capture semantic meaning in vector space",
        "Performance optimization is crucial for vector operations"
    ]
    
    // Create mock embeddings (in practice, these come from your ML model)
    let embeddings = documents.map { doc in
        TextEmbedding(
            text: doc,
            vector: Vector768.random(in: -1...1).normalized(),
            metadata: ["length": doc.count, "words": doc.split(separator: " ").count]
        )
    }
    
    // Semantic search
    let query = "How to optimize vector performance in Swift?"
    let queryEmbedding = Vector768.random(in: -1...1).normalized()
    
    // Find most similar documents
    let similarities = embeddings.map { embedding in
        (
            text: embedding.text,
            similarity: queryEmbedding.cosineSimilarity(to: embedding.vector)
        )
    }.sorted { $0.similarity > $1.similarity }
    
    print("Query: \"\(query)\"")
    print("\nTop 3 similar documents:")
    for (index, result) in similarities.prefix(3).enumerated() {
        print("\(index + 1). Similarity: \(String(format: "%.3f", result.similarity))")
        print("   \"\(result.text)\"")
    }
}

// MARK: - 2. Image Embeddings and Similarity

struct ImageEmbedding {
    let filename: String
    let vector: Vector512  // Common for image models
    let category: String
}

func imageSearchExample() {
    print("\n\n=== Image Similarity Search ===\n")
    
    // Simulate image embeddings database
    let imageDatabase = [
        ImageEmbedding(filename: "cat_01.jpg", vector: Vector512.random(in: -1...1).normalized(), category: "animal"),
        ImageEmbedding(filename: "dog_01.jpg", vector: Vector512.random(in: -1...1).normalized(), category: "animal"),
        ImageEmbedding(filename: "car_01.jpg", vector: Vector512.random(in: -1...1).normalized(), category: "vehicle"),
        ImageEmbedding(filename: "tree_01.jpg", vector: Vector512.random(in: -1...1).normalized(), category: "nature"),
        ImageEmbedding(filename: "cat_02.jpg", vector: Vector512.random(in: -1...1).normalized(), category: "animal"),
    ]
    
    // Query with an image
    let queryImage = imageDatabase[0] // Query with first cat image
    
    // Find similar images using different metrics
    let metrics: [(name: String, metric: any DistanceMetric)] = [
        ("Euclidean", EuclideanDistance()),
        ("Cosine", CosineDistance()),
        ("Manhattan", ManhattanDistance())
    ]
    
    print("Finding images similar to: \(queryImage.filename)")
    
    for (metricName, metric) in metrics {
        print("\n\(metricName) Distance:")
        
        let results = imageDatabase
            .filter { $0.filename != queryImage.filename }
            .map { image in
                (
                    filename: image.filename,
                    category: image.category,
                    distance: metric.distance(queryImage.vector, image.vector)
                )
            }
            .sorted { $0.distance < $1.distance }
            .prefix(3)
        
        for (index, result) in results.enumerated() {
            print("  \(index + 1). \(result.filename) (\(result.category)) - distance: \(String(format: "%.3f", result.distance))")
        }
    }
}

// MARK: - 3. Clustering Embeddings

func clusteringExample() {
    print("\n\n=== Clustering Example ===\n")
    
    // Generate synthetic embeddings for 3 clusters
    let cluster1 = (0..<10).map { _ in Vector128(randomIn: -1...0, bias: [-0.5, -0.5]) }
    let cluster2 = (0..<10).map { _ in Vector128(randomIn: 0...1, bias: [0.5, 0.5]) }
    let cluster3 = (0..<10).map { _ in Vector128(randomIn: -0.5...0.5, bias: [0, 0]) }
    
    let allVectors = cluster1 + cluster2 + cluster3
    
    // Simple k-means style centroid calculation
    func calculateCentroid(_ vectors: [Vector128]) -> Vector128 {
        guard !vectors.isEmpty else { return Vector128.zeros() }
        
        let sum = vectors.reduce(Vector128.zeros()) { $0 + $1 }
        return sum / Float(vectors.count)
    }
    
    // Calculate centroids
    let centroids = [
        calculateCentroid(cluster1),
        calculateCentroid(cluster2),
        calculateCentroid(cluster3)
    ]
    
    // Assign vectors to nearest centroid
    var assignments = [Int: [Vector128]]()
    for vector in allVectors {
        let distances = centroids.enumerated().map { index, centroid in
            (index: index, distance: vector.distance(to: centroid))
        }
        let nearest = distances.min(by: { $0.distance < $1.distance })!
        assignments[nearest.index, default: []].append(vector)
    }
    
    print("Cluster assignments:")
    for (cluster, vectors) in assignments.sorted(by: { $0.key < $1.key }) {
        let centroid = calculateCentroid(vectors)
        print("Cluster \(cluster + 1): \(vectors.count) vectors")
        print("  Centroid magnitude: \(String(format: "%.3f", centroid.magnitude))")
        print("  Average distance to centroid: \(String(format: "%.3f", vectors.map { $0.distance(to: centroid) }.reduce(0, +) / Float(vectors.count)))")
    }
}

// MARK: - 4. Recommendation System

struct User {
    let id: String
    let preferences: Vector256
}

struct RecommendationItem {
    let id: String
    let name: String
    let features: Vector256
}

func recommendationExample() {
    print("\n\n=== Recommendation System ===\n")
    
    // Create users with preference vectors
    let users = [
        User(id: "user1", preferences: Vector256.random(in: -1...1).normalized()),
        User(id: "user2", preferences: Vector256.random(in: -1...1).normalized()),
        User(id: "user3", preferences: Vector256.random(in: -1...1).normalized())
    ]
    
    // Create items with feature vectors
    let items = [
        RecommendationItem(id: "item1", name: "Swift Programming Book", features: Vector256.random(in: -1...1).normalized()),
        RecommendationItem(id: "item2", name: "Machine Learning Course", features: Vector256.random(in: -1...1).normalized()),
        RecommendationItem(id: "item3", name: "Vector Database Tutorial", features: Vector256.random(in: -1...1).normalized()),
        RecommendationItem(id: "item4", name: "iOS Development Guide", features: Vector256.random(in: -1...1).normalized()),
        RecommendationItem(id: "item5", name: "Data Science Handbook", features: Vector256.random(in: -1...1).normalized())
    ]
    
    // Generate recommendations for each user
    for user in users {
        print("\nRecommendations for \(user.id):")
        
        let recommendations = items
            .map { item in
                (
                    item: item,
                    score: user.preferences.cosineSimilarity(to: item.features)
                )
            }
            .sorted { $0.score > $1.score }
            .prefix(3)
        
        for (index, rec) in recommendations.enumerated() {
            print("  \(index + 1). \(rec.item.name) (score: \(String(format: "%.3f", rec.score)))")
        }
    }
}

// MARK: - 5. Anomaly Detection

func anomalyDetectionExample() {
    print("\n\n=== Anomaly Detection ===\n")
    
    // Create normal data points
    let normalData = (0..<50).map { _ in
        Vector64(randomIn: -1...1, scale: 0.3)  // Low variance
    }
    
    // Add some anomalies
    let anomalies = [
        Vector64(randomIn: -5...5, scale: 2.0),  // High variance
        Vector64(repeating: 10.0),               // Extreme values
        Vector64(randomIn: -10...10, scale: 3.0) // Another outlier
    ]
    
    let allData = normalData + anomalies
    
    // Calculate statistics for anomaly detection
    let meanVector = allData.reduce(Vector64.zeros()) { $0 + $1 } / Float(allData.count)
    let distances = allData.map { $0.distance(to: meanVector) }
    let meanDistance = distances.reduce(0, +) / Float(distances.count)
    let stdDistance = sqrt(distances.map { pow($0 - meanDistance, 2) }.reduce(0, +) / Float(distances.count))
    
    // Detect anomalies (> 2 standard deviations)
    let threshold = meanDistance + 2 * stdDistance
    
    print("Anomaly detection statistics:")
    print("Mean distance: \(String(format: "%.3f", meanDistance))")
    print("Std deviation: \(String(format: "%.3f", stdDistance))")
    print("Threshold (2σ): \(String(format: "%.3f", threshold))")
    
    print("\nDetected anomalies:")
    for (index, distance) in distances.enumerated() {
        if distance > threshold {
            print("  Index \(index): distance = \(String(format: "%.3f", distance)) (anomaly!)")
        }
    }
}

// MARK: - Helper Extensions

extension Vector128 {
    init(randomIn range: ClosedRange<Float>, bias: [Float]) {
        var values = [Float](repeating: 0, count: 128)
        for i in 0..<128 {
            values[i] = Float.random(in: range) + (i < bias.count ? bias[i] : 0)
        }
        self.init(values)
    }
}

extension Vector64 {
    init(randomIn range: ClosedRange<Float>, scale: Float) {
        let values = (0..<64).map { _ in Float.random(in: range) * scale }
        self.init(values)
    }
}

extension Vector256 {
    init(randomIn range: ClosedRange<Float>, bias: Float = 0) {
        let values = (0..<256).map { _ in Float.random(in: range) + bias }
        self.init(values)
    }
}

// MARK: - Main

func main() {
    textEmbeddingExample()
    imageSearchExample()
    clusteringExample()
    recommendationExample()
    anomalyDetectionExample()
}

// Run the example
main()