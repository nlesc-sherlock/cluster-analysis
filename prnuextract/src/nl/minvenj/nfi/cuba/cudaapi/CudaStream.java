/*
 * Copyright (c) 2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.minvenj.nfi.cuba.cudaapi;

import static jcuda.driver.JCudaDriver.cuStreamCreate;
import static jcuda.driver.JCudaDriver.cuStreamDestroy;
import static jcuda.driver.JCudaDriver.cuStreamSynchronize;
import static jcuda.driver.JCudaDriver.cuStreamWaitEvent;

import jcuda.driver.CUstream;
import jcuda.driver.CUstream_flags;

public final class CudaStream {
    private final CUstream _stream;

    public CudaStream() {
        _stream = new CUstream();

        cuStreamCreate(_stream, CUstream_flags.CU_STREAM_NON_BLOCKING);
    }

    public CUstream cuStream() {
        return _stream;
    }

    public void synchronize() {
        cuStreamSynchronize(_stream);
    }

    public void waitEvent(CudaEvent event) {
        cuStreamWaitEvent(_stream, event.cuEvent(), 0);
    }

    public void destroy() {
        cuStreamDestroy(_stream);
    }
}
