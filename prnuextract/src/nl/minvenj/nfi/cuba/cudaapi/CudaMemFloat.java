/*
 * Copyright (c) 2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.minvenj.nfi.cuba.cudaapi;

import jcuda.Pointer;
import jcuda.Sizeof;

public final class CudaMemFloat extends CudaMem {
    private static final long ELEMENT_SIZE = Sizeof.FLOAT;

    private final int _elementCount;

    CudaMemFloat(final int elementCount) {
        super(ELEMENT_SIZE * elementCount);

        _elementCount = elementCount;
    }

    public int elementCount() {
        return _elementCount;
    }

    public void copyHostToDevice(final float[] src, final int elementCount) {
        super.copyHostToDevice(Pointer.to(src), (ELEMENT_SIZE * elementCount));
    }

    public void copyDeviceToHost(final float[] dst, final int elementCount) {
        super.copyDeviceToHost(Pointer.to(dst), (ELEMENT_SIZE * elementCount));
    }

    public void copyHostToDeviceAsync(final float[] src, final int elementCount, CudaStream stream) {
        super.copyHostToDeviceAsync(Pointer.to(src), (ELEMENT_SIZE * elementCount), stream);
    }

    public void copyDeviceToHostAsync(final float[] dst, final int elementCount, CudaStream stream) {
        super.copyDeviceToHostAsync(Pointer.to(dst), (ELEMENT_SIZE * elementCount), stream);
    }
    
    public void copyHostToDeviceAsync(final float[] src, CudaStream stream) {
        super.copyHostToDeviceAsync(Pointer.to(src), (ELEMENT_SIZE * src.length), stream);
    }

    public void copyDeviceToHostAsync(final float[] dst, CudaStream stream) {
        super.copyDeviceToHostAsync(Pointer.to(dst), (ELEMENT_SIZE * dst.length), stream);
    }
    
    public void memset(final float val, final int elementCount) {
        super.memsetD32((int)val, elementCount);
    }

    public void memset(final float val, final int elementCount, CudaStream stream) {
        super.memsetD32((int)val, elementCount);
    }


}
