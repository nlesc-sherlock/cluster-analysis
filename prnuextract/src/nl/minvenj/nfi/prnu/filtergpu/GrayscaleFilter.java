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

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

import nl.minvenj.nfi.cuba.cudaapi.*;
import jcuda.*;
import jcuda.driver.*;

/**
 * Grayscale Filter class that takes an image as a BufferedImage
 * and converts it into a grayscale image stored as a float array
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */
public class GrayscaleFilter {

	private CudaContext _context;
	protected CudaStream _stream;

	//handles to CUDA kernels
	private CudaFunction _grayscale;

	//handles to device memory arrays
	protected CudaMemFloat _d_input;
	private CudaMemByte _d_colors;

	//parameterlist for kernel invocations
	private Pointer grayscale;

	/**
	 * Constructor for the Grayscale Filter, used only by the PRNUFilter factory
	 * 
	 * @param h - the image height in pixels
	 * @param w - the image width in pixels
	 * @param context - the CudaContext as created by the factory
	 * @param stream - the CudaStream as created by the factory
	 * @param input - the GPU memory for storing the grayscale image
	 * @param module - the CudaModule containing the kernels compiled by the factory
	 */
	public GrayscaleFilter (int h, int w, CudaContext context, CudaStream stream, CudaMemFloat input, CudaModule module) {
		_context = context;
		_stream = stream;
		_d_input = input;

		//setup gpu memory
		_d_colors =  _context.allocBytes(w*h*3);

		//setup cuda function
		final int threads_x = 32;
		final int threads_y = 16;
		_grayscale = module.getFunction("grayscale");
		_grayscale.setDim(      (int)Math.ceil((float)w / (float)threads_x), (int)Math.ceil((float)h / (float)threads_y), 1,
				threads_x, threads_y, 1);

		// Setup the parameter lists for each kernel call 
		grayscale = Pointer.to(
				Pointer.to(new int[]{h}),
				Pointer.to(new int[]{w}),
				Pointer.to(_d_input.getDevicePointer()),
				Pointer.to(_d_colors.getDevicePointer())
				);

	}
	
	/**
	 * Convert the image into a grayscaled image stored as an 1D float array on the CPU.
	 * This function is mainly here to compare with the result and performance of the GPU.
	 * 
	 * The conversion used currently is 0.299 r + 0.587 g + 0.114 b
	 * 
	 * @param image - a BufferedImage that needs to be converted into grayscale
	 * @return - an 1D float array that contains the grayscaled image
	 */
	public float[] apply1D(final BufferedImage image) {
		byte[] colors =  ((DataBufferByte) image.getRaster().getDataBuffer()).getData();

		int w = image.getWidth();
		int h = image.getHeight();
		float[] pixelsFloat = new float[w*h];

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				// switch them around, because the byte array is b g r
				float b = (float) (colors[(i*w+j) * 3 + 0] & 0xff);
				float g = (float) (colors[(i*w+j) * 3 + 1] & 0xff);
				float r = (float) (colors[(i*w+j) * 3 + 2] & 0xff);
				pixelsFloat[i*w+j] = 0.299f * r + 0.587f * g + 0.114f * b;
			}
		}

		return pixelsFloat;
	}

	/**
	 * Convert the image into a grayscaled image stored as an 1D float array on the GPU.
	 * The output is left in GPU memory for further processing.
	 * 
	 * The conversion used currently is 0.299 r + 0.587 g + 0.114 b
	 * 
	 * @param image - a BufferedImage that needs to be converted into grayscale
	 */
	public void applyGPU(BufferedImage image) {
		
		//extract the color values of the image into a byte array
		byte[] colors = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();

        //print value of first 5 pixels
        //for (int i=0; i < 15; i+=3) {
        //    System.out.print(colors[i] + "," + colors[i+1] + "," + colors[i+2] + " ");
        //}
        //System.out.println("");		

		//copy the image color values to the GPU
		_d_colors.copyHostToDeviceAsync(colors, _stream);

		//call GPU kernel to convert the color values to grayscaled float values
		_grayscale.launch(_stream, grayscale);

		//for measuring time
		JCudaDriver.cuCtxSynchronize();
	}

	/**
	 * cleans up allocated GPU memory
	 */
	public void cleanup() {
		_d_colors.free();		
	}

}
