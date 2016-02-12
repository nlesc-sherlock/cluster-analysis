/*
 * Copyright (c) 2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.minvenj.nfi.cuba.cudaapi;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;

import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;

public final class CudaContext {
    private final CUcontext _context;
    CudaDevice device;

    CudaContext(final CUdevice device, CudaDevice cudaDev) {
        _context = new CUcontext();
        this.device = cudaDev;

        cuCtxCreate(_context, 0, device);
    }

    public CudaStream createStream() {
        return new CudaStream();
    }

    public CudaEvent createEvent() {
        return new CudaEvent();
    }

    public CudaModule loadModule(final String cuSource, final String... nvccOptions) {
        return new CudaModule(cuSource, nvccOptions);
    }

    public CudaDevice getDevice() {
        return this.device;
    }

    public void synchronize() {
        cuCtxSynchronize();
    }

    public void destroy() {
        cuCtxDestroy(_context);
    }

    public CudaMemByte allocBytes(final int elementCount) {
        return new CudaMemByte(elementCount);
    }

    public CudaMemInt allocInts(final int elementCount) {
        return new CudaMemInt(elementCount);
    }

    public CudaMemFloat allocFloats(final int elementCount) {
        return new CudaMemFloat(elementCount);
    }

    public CudaMemFloat allocFloats(final float[] data) {
        final CudaMemFloat mem = new CudaMemFloat(data.length);
        mem.copyHostToDevice(data, data.length);
        return mem;
    }

    public CudaMemDouble allocDoubles(final int elementCount) {
        return new CudaMemDouble(elementCount);
    }

    @Override
    public String toString() {
        return _context.toString();
    }
}
