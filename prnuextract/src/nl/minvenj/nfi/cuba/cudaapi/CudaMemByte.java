/*
 * Copyright (c) 2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.minvenj.nfi.cuba.cudaapi;

import jcuda.Pointer;
import jcuda.Sizeof;

public final class CudaMemByte extends CudaMem {
    private static final int ELEMENT_SIZE = Sizeof.BYTE;

    private final int _elementCount;

    CudaMemByte(final int elementCount) {
        super(ELEMENT_SIZE * elementCount);

        _elementCount = elementCount;
    }

    public int elementCount() {
        return _elementCount;
    }

    public void copyHostToDevice(final byte[] data, final int elementCount) {
        super.copyHostToDevice(Pointer.to(data), (ELEMENT_SIZE * elementCount));
    }

    public void copyDeviceToHost(final byte[] dst, final int elementCount) {
        super.copyDeviceToHost(Pointer.to(dst), (ELEMENT_SIZE * elementCount));
    }
    
    public void copyHostToDeviceAsync(final byte[] src, final int elementCount, CudaStream stream) {
        super.copyHostToDeviceAsync(Pointer.to(src), (ELEMENT_SIZE * elementCount), stream);
    }

    public void copyDeviceToHostAsync(final byte[] dst, final int elementCount, CudaStream stream) {
        super.copyDeviceToHostAsync(Pointer.to(dst), (ELEMENT_SIZE * elementCount), stream);
    }
    
    public void copyHostToDeviceAsync(final byte[] src, CudaStream stream) {
        super.copyHostToDeviceAsync(Pointer.to(src), (ELEMENT_SIZE * src.length), stream);
    }

    public void copyDeviceToHostAsync(final byte[] dst, CudaStream stream) {
        super.copyDeviceToHostAsync(Pointer.to(dst), (ELEMENT_SIZE * dst.length), stream);
    }
    
}