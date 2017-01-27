/*
 * Copyright (c) 2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.minvenj.nfi.cuba.cudaapi;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoHAsync;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoDAsync;
import static jcuda.driver.JCudaDriver.cuMemsetD32;
import static jcuda.driver.JCudaDriver.cuMemsetD32Async;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;

public abstract class CudaMem {
    private final CUdeviceptr _deviceptr;

    CudaMem(final long bytesize) {
        _deviceptr = new CUdeviceptr();

        cuMemAlloc(_deviceptr, bytesize);
    }

    public Pointer toPointer() {
        return Pointer.to(_deviceptr);
    }

    public CUdeviceptr getDevicePointer() {
        return _deviceptr;
    }

    public void copyHostToDevice(final Pointer srcHost, final long byteCount) {
    	cuCtxSynchronize();
        cuMemcpyHtoD(_deviceptr, srcHost, byteCount);
    	cuCtxSynchronize();
    }

    public void copyDeviceToHost(final Pointer dstHost, final long byteCount) {
    	cuCtxSynchronize();
        cuMemcpyDtoH(dstHost, _deviceptr, byteCount);
    	cuCtxSynchronize();
    }

    public void copyHostToDeviceAsync(final Pointer srcHost, final long byteCount, CudaStream stream) {
    	cuMemcpyHtoDAsync(_deviceptr, srcHost, byteCount, stream.cuStream());
    }

    public void copyDeviceToHostAsync(final Pointer dstHost, final long byteCount, CudaStream stream) {
    	cuMemcpyDtoHAsync(dstHost, _deviceptr, byteCount, stream.cuStream());
    }
    
    protected void memsetD32(final int ui, final int n) {
        cuMemsetD32(_deviceptr, ui, n);
    }

    protected void memsetD32(final int ui, final int n, CudaStream stream) {
        cuMemsetD32Async(_deviceptr, ui, n, stream.cuStream());
    }

    public void free() {
        cuMemFree(_deviceptr);
    }

    @Override
    public String toString() {
        return _deviceptr.toString();
    }
}
