// cc-capture: captures system audio (the other side of the call) and the
// microphone (you) as two separate streams with ScreenCaptureKit, converts
// both to 16 kHz mono Float32, and writes framed PCM to stdout.
//
// Frame: 1 byte stream tag ('S' = system, 'M' = microphone),
//        UInt32 little-endian sample count, then that many Float32 LE samples.
// Status and errors go to stderr. Exits when stdin closes or on SIGTERM.
//
// --list-mics          print the microphones as JSON ([{"id", "name", "default"}]) and exit
// --mic <id>           capture this microphone instead of the system default

import AVFoundation
import Foundation
import ScreenCaptureKit

let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!
let writeQueue = DispatchQueue(label: "cc-capture.write")

func log(_ message: String) {
    FileHandle.standardError.write(("cc-capture: " + message + "\n").data(using: .utf8)!)
}

final class Converter {
    private var converter: AVAudioConverter?
    private var sourceFormat: AVAudioFormat?

    func convert(_ buffer: AVAudioPCMBuffer) -> [Float]? {
        if sourceFormat == nil || !(sourceFormat!.isEqual(buffer.format)) {
            sourceFormat = buffer.format
            converter = AVAudioConverter(from: buffer.format, to: targetFormat)
        }
        guard let converter else { return nil }
        let ratio = targetFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 32
        guard let out = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else { return nil }
        var supplied = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if supplied {
                status.pointee = .noDataNow
                return nil
            }
            supplied = true
            status.pointee = .haveData
            return buffer
        }
        if let error {
            log("convert failed: \(error.localizedDescription)")
            return nil
        }
        guard let channel = out.floatChannelData?[0] else { return nil }
        return Array(UnsafeBufferPointer(start: channel, count: Int(out.frameLength)))
    }
}

func emit(tag: UInt8, samples: [Float]) {
    guard !samples.isEmpty else { return }
    var data = Data([tag])
    var count = UInt32(samples.count).littleEndian
    data.append(Data(bytes: &count, count: 4))
    samples.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
    writeQueue.async {
        do {
            try FileHandle.standardOutput.write(contentsOf: data)
        } catch {
            exit(0)
        }
    }
}

final class Capture: NSObject, SCStreamOutput, SCStreamDelegate {
    private var stream: SCStream?
    private let systemConverter = Converter()
    private let micConverter = Converter()
    private let sampleQueue = DispatchQueue(label: "cc-capture.samples")

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        guard let display = content.displays.first else {
            throw NSError(domain: "cc-capture", code: 1, userInfo: [NSLocalizedDescriptionKey: "no display found"])
        }
        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.captureMicrophone = true
        if let mic = micID {
            config.microphoneCaptureDeviceID = mic
        }
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 48_000
        config.channelCount = 1
        // Video is required by the API; keep it as small and slow as possible.
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: sampleQueue)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: sampleQueue)
        try stream.addStreamOutput(self, type: .microphone, sampleHandlerQueue: sampleQueue)
        try await stream.startCapture()
        self.stream = stream
        log("capturing")
    }

    func stop() async {
        try? await stream?.stopCapture()
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio || type == .microphone, sampleBuffer.isValid else { return }
        guard let description = sampleBuffer.formatDescription?.audioStreamBasicDescription else { return }
        var asbd = description
        guard let format = AVAudioFormat(streamDescription: &asbd) else { return }
        try? sampleBuffer.withAudioBufferList { list, _ in
            guard let pcm = AVAudioPCMBuffer(pcmFormat: format, bufferListNoCopy: list.unsafePointer) else { return }
            let converter = type == .audio ? systemConverter : micConverter
            if let samples = converter.convert(pcm) {
                emit(tag: type == .audio ? UInt8(ascii: "S") : UInt8(ascii: "M"), samples: samples)
            }
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        log("stopped: \(error.localizedDescription)")
        exit(1)
    }
}

func microphones() -> [AVCaptureDevice] {
    AVCaptureDevice.DiscoverySession(deviceTypes: [.microphone, .external], mediaType: .audio,
                                     position: .unspecified).devices
}

let arguments = CommandLine.arguments
if arguments.contains("--list-mics") {
    let defaultID = AVCaptureDevice.default(for: .audio)?.uniqueID
    let list = microphones().map { ["id": $0.uniqueID, "name": $0.localizedName, "default": $0.uniqueID == defaultID] as [String: Any] }
    let data = try! JSONSerialization.data(withJSONObject: list)
    FileHandle.standardOutput.write(data)
    exit(0)
}
var micID: String? = nil
if let index = arguments.firstIndex(of: "--mic"), index + 1 < arguments.count {
    micID = arguments[index + 1]
}

let capture = Capture()
signal(SIGPIPE, SIG_IGN)
let term = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
signal(SIGTERM, SIG_IGN)
term.setEventHandler {
    Task {
        await capture.stop()
        exit(0)
    }
}
term.resume()

// Stop when the parent closes our stdin.
DispatchQueue.global().async {
    while !FileHandle.standardInput.availableData.isEmpty {}
    Task {
        await capture.stop()
        exit(0)
    }
}

Task {
    do {
        try await capture.start()
    } catch {
        log("could not start: \(error.localizedDescription)")
        exit(2)
    }
}

dispatchMain()
