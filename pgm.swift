import os
import RealityKit
import ModelIO
import Metal

private var timeFormatter  = {
    let formatter = DateFormatter()
    formatter.dateFormat = "hh:mm:ss SSSS"
    return formatter
}()

private var percentFormatter = {
    let formatter = NumberFormatter()
    formatter.numberStyle = .percent
    formatter.minimumIntegerDigits = 2
    formatter.maximumIntegerDigits = 2
    formatter.minimumFractionDigits = 2
    formatter.maximumFractionDigits = 2

    return formatter
}()

private var secondsFormatter = {
    let formatter = NumberFormatter()
    formatter.numberStyle = .decimal
    formatter.maximumFractionDigits = 2

    return formatter
}()

private var timeIntervalFormatter = {
    let formatter = DateComponentsFormatter()
    formatter.allowedUnits = [.minute, .second]
    formatter.zeroFormattingBehavior = .pad

    return formatter
}()

let LOG = false;

private func info(_ message: String, terminator: String = "\n") {
    if (LOG) {
        print("\u{001B}[0;37m[\(timeFormatter.string(from: Date()))] \(message)\u{001B}[0;0m", terminator: terminator)
    }
}

private func warn(_ message: String, terminator: String = "\n") {
    print("\u{001B}[0;33m[\(timeFormatter.string(from: Date()))] \(message)\u{001B}[0;0m", terminator: terminator)
}


private func err(_ message: String, terminator: String = "\n") {
    print("\u{001B}[0;31m[\(timeFormatter.string(from: Date()))] \u{001B}[1;31m\(message)\u{001B}[0;0m", terminator: terminator)
    exit(1)
}

private let completed = OSAllocatedUnfairLock(initialState: false);
private let progress = OSAllocatedUnfairLock(initialState: Double(0))
private let eta = OSAllocatedUnfairLock(initialState: Double(0))
var start_time: Date = Date();
private var activeSession: PhotogrammetrySession? = nil

@_cdecl("is_completed")
public func IsCompleted() -> Bool {
    completed.withLock { completed in return completed }
}

@_cdecl("get_progress")
public func GetProgress() -> Double {
    progress.withLock { progress in return progress }
}

@_cdecl("get_eta")
public func GetETA() -> Double {
    eta.withLock { eta in return eta }
}

@_cdecl("stop_photogrammetry_session")
public func StopPhotogrammetrySession() {
    guard let session = activeSession else {
        warn("No active session to stop.")
        return
    }

    Task {
        session.cancel()
        warn("Photogrammetry session was requested to cancel.")
    }
}

@_cdecl("run_photogrammetry_session")
public func RunPhotogrammetrySession(imagesPath: UnsafePointer<CChar>) {
    let imagesPath = String(cString: imagesPath)
    
    guard PhotogrammetrySession.isSupported else {
        err("Photogrammetry not supported on this device")
        exit(1)
    }

    let inputFolderUrl = URL(filePath: imagesPath, directoryHint: URL.DirectoryHint.isDirectory)
    info("Using input images folder: \(String(describing: inputFolderUrl))")

    var configuration = PhotogrammetrySession.Configuration()
    configuration.sampleOrdering = .sequential
    configuration.featureSensitivity = .normal

    info("Using configuration: \(String(describing: configuration))")

    var possibleSession: PhotogrammetrySession? = nil;
    do {
        possibleSession = try PhotogrammetrySession(input: inputFolderUrl, configuration: configuration)
        info("Created Photogrammetry Session")
    } catch {
        err("Error creating session: \(String(describing: error))")
        exit(1)
    }

    guard let session = possibleSession else {
        exit(1)
    }

    activeSession = session

    let logger = Task {
        for try await output in session.outputs {
            switch output {
                case .processingComplete:
                    info("Processing is complete!")
                    info("Time Taken: \(timeIntervalFormatter.string(from: start_time, to: Date()) ?? "")")

                    let modelAsset = MDLAsset(url: URL(filePath: "\(imagesPath)/out.usdz"))
                    modelAsset.loadTextures()

                    try? FileManager.default.createDirectory(at: URL(filePath: "model"), withIntermediateDirectories: true)

                     do {
                         try modelAsset.export(to: URL(filePath: "\(imagesPath)/model/out.obj"))
                     } catch {
                         err("Error while exporting: \(error)")
                     }

                    completed.withLock { completed in completed = true }

                    return
                case .processingCancelled:
                    warn("Processing was cancelled.")
                    progress.withLock { $0 = 0.0 }
                    eta.withLock { $0 = 0.0 }
                    completed.withLock { $0 = false }
                    activeSession = nil
                    return
                case .requestError(let request, let error):
                    info("Request \(String(describing: request)) had an error: \(String(describing: error))")
                case .requestComplete(let request, let result):
                    info("Request complete: \(String(describing: request)) with result...")
                    switch result {
                        case .modelFile(let url):
                            info("\tmodelFile available at url=\(url)")
                        default:
                            warn("\tUnexpected result: \(String(describing: result))")
                    }
                case .requestProgress(_, let fractionComplete):
                    progress.withLock { progress in progress = fractionComplete }
                    info("Progress = \(percentFormatter.string(from: NSNumber(value: fractionComplete)) ?? "")")
                case .inputComplete:
                    info("Data ingestion is complete.  Beginning processing...")
                case .invalidSample(let id, let reason):
                    warn("Invalid Sample! id=\(id)  reason=\"\(reason)\"")
                case .skippedSample(let id):
                    warn("Sample id=\(id) was skipped by processing.")
                case .automaticDownsampling:
                    warn("Automatic downsampling was applied!")
                case .requestProgressInfo(_, let requestInfo):
                    eta.withLock { eta in eta = requestInfo.estimatedRemainingTime ?? -1 }
                    info("Estimated Time Remaining = \(secondsFormatter.string(from: NSNumber(value: requestInfo.estimatedRemainingTime!)) ?? "Unknown")")
                case .stitchingIncomplete:
                    warn("Received stitching incomplete message.")
                @unknown default:
                    err("Output: unhandled message: \(output.localizedDescription)")
            }
        }
    }

    withExtendedLifetime((session, logger)) {
        do {
            try? FileManager.default.removeItem(atPath: "\(imagesPath)/out.usdz")

            let request = PhotogrammetrySession.Request.modelFile(url: URL(filePath: "\(imagesPath)/out.usdz"), detail: .medium)
            info("Using request: \(String(describing: request))")

            start_time = Date()

            try session.process(requests: [ request ])
        } catch {
            print("Error occured: \(error)")
        }
    }
}