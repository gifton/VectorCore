import Foundation
import VectorCoreBenchSupport

@main
struct App {
    static func main() async {
        exit(await BenchmarkRunner.runCLI())
    }
}
