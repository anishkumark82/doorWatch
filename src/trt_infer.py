import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTEngine:
    def __init__(self, engine_path):
        # Step 1: load the compiled engine file into CPU memory -- blob of bytes
        # f -- file descriptor
        # trt.Runtime (runtime) -- the deserializer; turns those raw bytes into
        #   a live, GPU-resident engine object via deserialize_cuda_engine()
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # Context: runtime state for one inference session (I/O buffer bindings,
        # workspace memory). Only one inference should be in flight per context.
        self.context = self.engine.create_execution_context()

        # 1. Stream: an ordered queue of async GPU operations.
        # 2. Current use case has only one stream (one frame at a time).
        # 3. synchronize() at the end blocks the CPU until the whole queued
        #    sequence (copy-in -> execute -> copy-out) has actually finished --
        #    without it, infer() could return before the GPU is done and
        #    host_mem would contain stale/garbage data.
        self.stream = cuda.Stream()

        # Storage, keyed by tensor name rather than a single fixed name --
        # this is the fix: SCRFD has multiple output tensors (boxes, scores,
        # landmarks), so a single self.output_name would silently drop all
        # but the last one found in the loop below.
        self.input_names = []
        self.output_names = []
        self.host_mem = {}
        self.dev_mem = {}
        self.shapes = {}

        # Iterate through every named input/output slot the engine exposes
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)

            # Allocate matching CPU (pinned/pagelocked -- faster transfer)
            # and GPU buffers for this tensor
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)

            self.host_mem[name] = host_mem
            self.dev_mem[name] = dev_mem
            self.shapes[name] = shape
            self.context.set_tensor_address(name, int(dev_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        print(f"Loaded engine: inputs={self.input_names}, outputs={self.output_names}")

    def infer(self, input_data):
        # Accepts a single array (models with one input, e.g. ArcFace/CLIP)
        # or a dict {tensor_name: array} for models with multiple inputs
        if isinstance(input_data, dict):
            for name, arr in input_data.items():
                np.copyto(self.host_mem[name], arr.ravel())
        else:
            name = self.input_names[0]
            np.copyto(self.host_mem[name], input_data.ravel())

        # Copy every input CPU -> GPU
        for name in self.input_names:
            cuda.memcpy_htod_async(self.dev_mem[name], self.host_mem[name], self.stream)

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy every output GPU -> CPU
        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.host_mem[name], self.dev_mem[name], self.stream)

        # Block until the whole queued sequence above has actually completed
        self.stream.synchronize()

        # Return all outputs, reshaped to their proper dimensions, keyed by name
        outputs = {name: self.host_mem[name].reshape(self.shapes[name]).copy() for name in self.output_names}
        return outputs