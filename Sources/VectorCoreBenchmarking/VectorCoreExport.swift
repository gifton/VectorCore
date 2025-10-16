//
//  VectorCoreExport.swift
//  VectorCoreBenchmarking
//
//  Re-exports VectorCore to ensure proper dependency linking
//

// This re-export ensures that when a client imports VectorCoreBenchmarking,
// they also get VectorCore, and the dependency chain is properly established
// for the build system.
@_exported import VectorCore