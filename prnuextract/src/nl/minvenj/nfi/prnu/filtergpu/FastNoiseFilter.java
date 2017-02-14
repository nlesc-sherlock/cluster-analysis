/*
* Copyright 2015 Netherlands eScience Center, VU University Amsterdam, and Netherlands Forensic Institute
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
package nl.minvenj.nfi.prnu.filtergpu;

import nl.minvenj.nfi.cuba.cudaapi.*;
import jcuda.*;
import jcuda.driver.*;

/**
 * FastNoiseFilter for extraction PRNU pattern from an image
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 *
 */
public final class FastNoiseFilter {

	private static final float EPS = 1.0f;

	protected CudaContext _context;
	protected CudaStream _stream;

	//handles to CUDA kernels
	protected CudaFunction _normalized_gradient;
	protected CudaFunction _gradient;

	//handles to device memory arrays
	protected CudaMemFloat _d_input;
	protected CudaMemFloat _d_temp;

	//threads
	protected int _threads_x = 32;
	protected int _threads_y = 16;
	protected int _threads_z = 1;

	//grid
	protected int _grid_x;
	protected int _grid_y;
	protected int _grid_z;

	//parameterlists for kernel invocations
	protected Pointer normalized_gradient;
	protected Pointer gradient;

	protected int h;
	protected int w;

	/**
	 * Constructor for the FastNoise Filter, used only by the PRNUFilter factory
	 * 
	 * @param h - the image height in pixels
	 * @param w - the image width in pixels
	 * @param context - the CudaContext as created by the factory
	 * @param stream - the CudaStream as created by the factory
	 * @param input - the GPU memory for storing the pattern
	 * @param module - the CudaModule containing the kernels compiled by the factory
	 */
	public FastNoiseFilter (int h, int w, CudaContext context, CudaStream stream, CudaMemFloat input, CudaModule module) {
		_context = context;
		_stream = stream;
		_d_input = input;
		this.h = h;
		this.w = w;
		
		//setup grid dimensions
		_grid_x = (int)Math.ceil((float)w / (float)_threads_x);
		_grid_y = (int)Math.ceil((float)h / (float)_threads_y);
		_grid_z = 1;

		//setup cuda functions
		_normalized_gradient = module.getFunction("normalized_gradient");
		_normalized_gradient.setDim(_grid_x, _grid_y, _grid_z, _threads_x, _threads_y, _threads_z);

		_gradient = module.getFunction("gradient");
		_gradient.setDim(_grid_x, _grid_y, _grid_z, _threads_x, _threads_y, _threads_z);

		// Allocate the CUDA buffers for this kernel
		_d_temp = _context.allocFloats(w*h);

		// Setup the parameter lists for each kernel call 
		normalized_gradient = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_temp.getDevicePointer()),
				Pointer.to(_d_input.getDevicePointer())
				);

		gradient = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_input.getDevicePointer()),
				Pointer.to(_d_temp.getDevicePointer())
				);

	}

	/**
	 * Vertically computes a local gradient for each pixel in an image.
	 * Takes forward differences for first and last row.
	 * Takes centered differences for interior points.
	 * 
	 * @param h - the image height in pixels
	 * @param w - the image width in pixels
	 * @param output - the local gradient values 
	 * @param input - the input image stored as an 1D float array 
	 */
	static void convolveVertically(int h, int w, float[] output, float[] input) {
		for (int j = 0; j < w; j++) {
			output[0*w+j] += input[1*w+j] - input[0*w+j];
			output[(h-1)*w+j] += input[(h-1)*w+j] - input[(h-2)*w+j];

			for (int i = 1; i < h - 1; i++) {
				output[i*w+j] += 0.5f * (input[(i+1)*w+j] - input[(i-1)*w+j]);
			}
		}

	}

	/**
	 * Horizontally computes a local gradient for each pixel in an image.
	 * Takes forward differences for first and last element.
	 * Takes centered differences for interior points.
	 * 
	 * @param h - the image height in pixels
	 * @param w - the image width in pixels
	 * @param output - the local gradient values 
	 * @param input - the input image stored as an 1D float array 
	 */
	static void convolveHorizontally(int h, int w, float[] output, float[] input) {
		for (int i = 0; i < h; i++) {
			output[i*w+0] += input[i*w+1] - input[i*w+0];
			output[i*w+w-1] += input[i*w+w-1] - input[i*w+w-2];

			for (int j = 1; j < w - 1; j++) {
				output[i*w+j] += 0.5f * (input[i*w+j+1] - input[i*w+j-1]);
			}
		}
	}

	/**
	 * Normalizes gradient values in place.
	 * 
	 * @param h - the image height in pixels
	 * @param w - the image width in pixels
	 * @param dxs - an array of gradient values
	 * @param dys - an array of gradient values
	 */
	static void normalize(int h, int w, float[] dxs, float[] dys) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				float dx = dxs[i*w+j];
				float dy = dys[i*w+j];

				float norm = (float) Math.sqrt((dx * dx) + (dy * dy));
				float scale = 1.0f / (EPS + norm);

				dxs[i*w+j] = scale * dx;
				dys[i*w+j] = scale * dy;
			}
		}
	}

	/**
	 * Zeros all values in an 1D array of float values.
	 * 
	 * @param h - the image height in pixels
	 * @param w - the image width in pixels
	 * @param input - the array containing only zero values after this method
	 */
	static void toZero(int h, int w, float[] input) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				input[i*w+j] = 0.0f;
			}
		}
	}

	/**
	 * Applies the FastNoise Filter to extract a PRNU Noise pattern from the input image.
	 * 
	 * @param input - a float array containing a grayscale image or single color channel
	 * @return - a float array containing the extract PRNU Noise pattern
	 */
	public float[] apply1D(float[] input) {
		float[] dxs = new float[h*w];
		float[] dys = new float[h*w];

		convolveHorizontally(h, w, dxs, input);
		convolveVertically(h, w, dys, input);

		normalize(h, w, dxs, dys);

		toZero(h, w, input);

		convolveHorizontally(h, w, input, dxs);
		convolveVertically(h, w, input, dys);

		return input;
	}

	/**
	 * This method applies the FastNoise Filter on the GPU.
	 * The input is already in GPU memory.
	 * The output PRNU Noise pattern is stored in place of the input.
	 */
	public void applyGPU() {

		_normalized_gradient.launch(_stream, normalized_gradient);

		_gradient.launch(_stream, gradient);

	}

	/**
	 * cleans up GPU memory
	 */
	public void cleanup() {
		_d_temp.free();
	}

}
