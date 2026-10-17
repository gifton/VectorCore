import VectorCore

@main struct L02PublicConsumer {
    static func main() throws {
        let data = [DynamicVector([Float(16777216)]), DynamicVector([Float(16777218)])]
        let model = try PCAModel.fit(data, components: 1, config: PCAConfig(oversampling: Int.max))
        let mean: [Double] = model.mean
        precondition(mean == [16777217])
        let single = try model.transform([Float(16777216)])
        let batch = try model.transform(data)
        precondition(single.count == 1 && single[0].isFinite && abs(single[0]+1) < 1e-5)
        precondition(batch.count == 2 && batch[1].count == 1 && batch[1][0].isFinite && abs(batch[1][0]-1) < 1e-5)
        print("L02_PUBLIC_COMPLETE")
    }
}
