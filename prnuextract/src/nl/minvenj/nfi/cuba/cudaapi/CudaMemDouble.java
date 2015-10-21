/*
 * Copyright (c) 2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.minvenj.nfi.cuba.cudaapi;

import jcuda.Pointer;
import jcuda.Sizeof;

public final class CudaMemDouble extends CudaMem {
    private static final long ELEMENT_SIZE = Sizeof.DOUBLE;

    private final int _elementCount;

    CudaMemDouble(final int elementCount) {
        super(ELEMENT_SIZE * elementCount);

        _elementCount = elementCount;
    }

    public int elementCount() {
        return _elementCount;
    }

    public void copyHostToDevice(final double[] src, final int elementCount) {
        super.copyHostToDevice(Pointer.to(src), (ELEMENT_SIZE * elementCount));
    }

    public void copyDeviceToHost(final double[] dst, final int elementCount) {
        super.copyDeviceToHost(Pointer.to(dst), (ELEMENT_SIZE * elementCount));
    }

    public void copyHostToDeviceAsync(final double[] src, final int elementCount, CudaStream stream) {
        super.copyHostToDeviceAsync(Pointer.to(src), (ELEMENT_SIZE * elementCount), stream);
    }

    public void copyDeviceToHostAsync(final double[] dst, final int elementCount, CudaStream stream) {
        super.copyDeviceToHostAsync(Pointer.to(dst), (ELEMENT_SIZE * elementCount), stream);
    }
    
    public void copyHostToDeviceAsync(final double[] src, CudaStream stream) {
        super.copyHostToDeviceAsync(Pointer.to(src), (ELEMENT_SIZE * src.length), stream);
    }

    public void copyDeviceToHostAsync(final double[] dst, CudaStream stream) {
        super.copyDeviceToHostAsync(Pointer.to(dst), (ELEMENT_SIZE * dst.length), stream);
    }
    
    public void memset(final double d, final int elementCount) {
        super.memsetD32((int)d, (elementCount * 2));
    }
}