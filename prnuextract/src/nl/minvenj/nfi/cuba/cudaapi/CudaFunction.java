/*
 * Copyright (c) 2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.minvenj.nfi.cuba.cudaapi;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import java.util.HashMap;
import java.util.Map;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;

public final class CudaFunction {
	private final CUfunction _function;
	private final Map<Integer, Pointer> _params;

	int _grid_x;
	int _grid_y;
	int _grid_z;
	int _threads_x;
	int _threads_y;
	int _threads_z;

	CudaFunction(final CUfunction function) {
		_function = function;
		_params = new HashMap<Integer, Pointer>();
	}

	public void setParam(final int i, final CudaMem param) {
		_params.put(i, param.toPointer());
	}

	public void setParam(final int i, final int param) {
		_params.put(i, Pointer.to(new int[]{param}));
	}

	public void launchKernel(final int gridDimX, final int gridDimY, final int gridDimZ, final int blockDimX, final int blockDimY, final int blockDimZ, final int sharedMemBytes, final CudaStream stream) {
		final CUstream cuStream = (stream == null) ? null : stream.cuStream();
		cuLaunchKernel(_function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, cuStream, kernelParams(), null);
	}
	
	public void launchKernel(final int gridDimX, final int gridDimY, final int gridDimZ, final int blockDimX, final int blockDimY, final int blockDimZ, final int sharedMemBytes, final CudaStream stream, Pointer parameters) {
		final CUstream cuStream = (stream == null) ? null : stream.cuStream();
		cuLaunchKernel(_function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, cuStream, parameters, null);
	}

	public void setDim(int gx, int gy, int gz, int tx, int ty, int tz) {
		_grid_x = gx;
		_grid_y = gy;
		_grid_z = gz;
		_threads_x = tx;
		_threads_y = ty;
		_threads_z = tz;
	}

	public void launch(CudaStream stream, Pointer parameters) {
		cuLaunchKernel(_function, _grid_x, _grid_y, _grid_z, _threads_x, _threads_y, _threads_z, 0, stream.cuStream(), parameters, null);
	}

	private Pointer kernelParams() {
		int maxParam = -1;
		for (final int i : _params.keySet()) {
			maxParam = Math.max(maxParam, i);
		}

		final Pointer[] params = new Pointer[maxParam + 1];
		for (final Map.Entry<Integer, Pointer> entry : _params.entrySet()) {
			params[entry.getKey()] = entry.getValue();
		}
		return Pointer.to(params);
	}

	public CUfunction getFunc() {
		return _function;
	}

	@Override
	public String toString() {
		return _function.toString();
	}
}
