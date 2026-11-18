public protocol MinimalVector: Collection where Index == Int {
    func toArray() -> [Element]
}
public extension MinimalVector {
    var startIndex: Int { 0 }
    var endIndex: Int { toArray().count }
    func index(after i: Int) -> Int { i + 1 }
    subscript(index: Int) -> Element { toArray()[index] }
}
