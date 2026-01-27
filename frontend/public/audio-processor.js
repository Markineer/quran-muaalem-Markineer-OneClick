/**
 * AudioWorklet Processor for real-time audio streaming
 *
 * This processor captures audio in chunks and sends them to the main thread
 * for WebSocket transmission. AudioWorklet runs on a separate thread and
 * has lower latency than ScriptProcessorNode.
 */

class AudioStreamProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this.bufferSize = 4096
    this.buffer = new Float32Array(this.bufferSize)
    this.bufferIndex = 0
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0]
    if (!input || !input[0]) {
      return true
    }

    const inputChannel = input[0]

    // Accumulate samples into buffer
    for (let i = 0; i < inputChannel.length; i++) {
      this.buffer[this.bufferIndex++] = inputChannel[i]

      // When buffer is full, send to main thread
      if (this.bufferIndex >= this.bufferSize) {
        // Convert float32 to int16
        const pcmData = new Int16Array(this.bufferSize)
        for (let j = 0; j < this.bufferSize; j++) {
          pcmData[j] = Math.max(-32768, Math.min(32767, this.buffer[j] * 32768))
        }

        // Send buffer to main thread
        this.port.postMessage({
          type: 'audio',
          buffer: pcmData.buffer
        }, [pcmData.buffer])

        // Reset buffer
        this.buffer = new Float32Array(this.bufferSize)
        this.bufferIndex = 0
      }
    }

    return true
  }
}

registerProcessor('audio-stream-processor', AudioStreamProcessor)
