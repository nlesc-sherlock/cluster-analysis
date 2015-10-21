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
package nl.minvenj.nfi.prnu;

import java.awt.color.CMMException;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;

import javax.imageio.ImageIO;
import edu.emory.mathcs.jtransforms.fft.FloatFFT_2D;

/**
 * Util class for PRNU extraction applications
 * 
 * @author Pieter Hijma <pieter@cs.vu.nl> and Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */
public class Util {


	public static BufferedImage readImage(final File file) throws IOException {
		final InputStream fileInputStream = new FileInputStream(file);
		try {
			final BufferedImage image = ImageIO.read(new BufferedInputStream(fileInputStream));
			if ((image != null) && (image.getWidth() >= 0) && (image.getHeight() >= 0)) {
				return image;
			}
		}
		catch (final CMMException e) {
			throw new IOException(e.getMessage());
		}
		catch (final RuntimeException e) {
			throw new IOException(e.getMessage());
		}
		finally {
			fileInputStream.close();
		}

		return null;
	}

	public static BufferedImage readImage2(final File file) {
		try {
			final BufferedImage image = ImageIO.read(file);
			if ((image != null) && (image.getWidth() > 0) && (image.getHeight() > 0)) {
				return image;
			}
		}
		catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}

		return null;
	}


	public static Object readJavaObject(final File inputFile) throws IOException {
		final ObjectInputStream inputStream = new ObjectInputStream(new BufferedInputStream(new FileInputStream(inputFile)));
		try {
			return inputStream.readObject();
		}
		catch (final ClassNotFoundException e) {
			throw new IOException("Cannot read pattern: " + inputFile.getAbsolutePath(), e);
		}
		finally {
			inputStream.close();
		}
	}

	public static void writeJavaObject(final Object object, final File outputFile) throws IOException {
		final OutputStream outputStream = new FileOutputStream(outputFile);
		try {
			final ObjectOutputStream objectOutputStream = new ObjectOutputStream(new BufferedOutputStream(outputStream));
			objectOutputStream.writeObject(object);
			objectOutputStream.close();
		}
		finally {
			outputStream.close();
		}
	}

	public static boolean compare2DArray(final float[][] expected, final float[][] actual, final float delta) {
		for (int i = 0; i < expected.length; i++) {
			for (int j = 0; j < expected[i].length; j++) {
				if (Math.abs(actual[i][j] - expected[i][j]) > delta) {
					System.err.println("value at " + i + "," + j + " is " + actual[i][j] + " should have been " + expected[i][j]);
					return false;
				}
			}
		}
		return true;
	}

	public static boolean compareArray(final float[] expected, final float[] actual, final float delta) {
		int zeroActual = 0;
		int zeroExpected = 0;
		int numerr = 0;
		for (int i = 0; i < expected.length && numerr < 20; i++) {
			float diff1 = Math.abs(actual[i] - expected[i])/expected[i];
			float diff2 = Math.abs(actual[i] - expected[i])/actual[i];
			float diff = Math.max(diff1, diff2);
			//if (Math.abs(actual[i] - expected[i]) > delta) {
			if (diff > delta) {
				System.err.println("value at " + i + " is " + actual[i] + " should have been " + expected[i] + " diff is " + diff );
				numerr++;
			}
			if (Float.isNaN(actual[i])) {
				System.err.println("value at " + i + " is " + actual[i]);
				numerr++;
			}
			if (Float.isNaN(expected[i])) {
				System.err.println("value at " + i + " is " + actual[i] + " should have been " + expected[i]);
				numerr++;
			}
			if (Math.abs(actual[i]) < delta) {
				zeroActual++;
			}
			if (Math.abs(expected[i]) < delta) {
				zeroExpected++;
			}
		}
		if (zeroActual != zeroExpected) {
			System.err.println("Number of zeros in array is " + zeroActual + " but expected " + zeroExpected);
		}
		if (zeroActual > 0.95*expected.length) {
			System.err.println("Number of zeros in both arrays is larger than 95% of the values");
			return false;
		}
		if (numerr > 0) {
			return false;
		}

		return true;
	}


	public static void print2DArray(float[][] array) {
		for (int i = 0; i < array.length; i++) {
			for (int j = 0; j < array[0].length; j++) {
				System.out.printf("% 2.1f ", array[i][j]);
			}
			System.out.println();
		}
	}


	public static float[][] transpose(float[][] pixels) {
		float[][] pixelsTransposed = new float[pixels[0].length][pixels.length];

		for (int i = 0; i < pixels.length; i++) {
			for (int j = 0; j < pixels[0].length; j++) {
				pixelsTransposed[j][i] = pixels[i][j];
			}
		}

		return pixelsTransposed;
	}
	public static float[] transpose(int h, int w, float[] pixels) {
		float[] pixelsTransposed = new float[w*h];

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				pixelsTransposed[j*h+i] = pixels[i*w+j];
			}
		}

		return pixelsTransposed;
	}


	// assuming that the layout is r i r i r i
	//                             r i r i r i
	// and it becomes r i r i
	//                r i r i
	//                r i r i
	public static float[][] transposeComplex(float[][] input) {
		float[][] transposed = new float[input[0].length/2][input.length * 2];

		for (int i = 0; i < input.length; i++) {
			for (int j = 0; j < input[0].length/2; j++) {
				transposed[j][i*2] = input[i][j*2];
				transposed[j][i*2+1] = input[i][j*2+1];
			}
		}
		return transposed;
	}


	public static float[][] copy(float[][] input) {
		int h = input.length;
		int w = input[0].length;
		float[][] output = new float[h][w];

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				output[i][j] = input[i][j];
			}
		}
		return output;
	}




	public static float[] toComplex(int h, int w, float[] input) {
		float[] complex = new float[h * w * 2];

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				complex[i * w * 2 + 2 * j] = input[i * w + j];
				complex[i * w * 2 + (2 * j + 1)] = 0.0f;
			}
		}
		return complex;
	}



	public static float[][] toComplex(float[][] input) {
		int h = input.length;
		int w = input[0].length;

		float[][] complex = new float[h][w*2];

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				complex[i][2*j] = input[i][j];
				complex[i][2*j+1] = 0.0f;
			}
		}
		return complex;
	}


	public static float[] toReal(float[] input) {
		float[] real = new float[input.length/2];

		for (int i = 0; i < input.length/2; i++) {
			real[i] = input[i * 2];
		}
		return real;
	}

	public static float[][] toReal(float[][] input) {
		int h = input.length;
		int w = input[0].length / 2;

		float[][] real = new float[h][w];

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				real[i][j] = input[i][2*j];
			}
		}
		return real;
	}

	public static float[][] fft(float[][] input) {
		int h = input.length;
		int w = input[0].length/2; // count in complex numbers
		new FloatFFT_2D(h, w).complexForward(input);
		return input;
	}

	public static float[][] ifft(float[][] input) {
		int h = input.length;
		int w = input[0].length/2; // count in complex numbers
		new FloatFFT_2D(h, w).complexInverse(input, true);
		return input;
	}



	public static float[] fft(int h, int w, float[] input) {
		new FloatFFT_2D(h, w).complexForward(input);
		return input;
	}

	public static float[] ifft(int h, int w, float[] input) {
		new FloatFFT_2D(h, w).complexInverse(input, true);
		return input;
	}



	public static float[][] initializeArray(int h, int w, float value) {
		float[][] result = new float[h][w];

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				result[i][j] = value;
			}
		}

		return result;
	}


	/*** Add a border of borderSize. */
	public static float[][] addBorder(float[][] input, int borderSize) {
		int h = input.length;
		int w = input[0].length;

		float[][] result = new float[h + 2 * borderSize][w + 2 * borderSize];

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				result[i + borderSize][j + borderSize] = input[i][j];
			}
		}

		return result;
	}
	/*** Add a border of borderSize. */
	public static float[] addBorder(int h, int w, float[] input, int borderSize) {

		float[] result = new float[(h + 2 * borderSize) * (w + 2 * borderSize)];

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				result[(i + borderSize)*(w+2*borderSize)+(j + borderSize)] = input[i*w+j];
			}
		}

		return result;
	}

	/*** Sum the values of an array.
	 */
	public static float sum(float[][] input) {
		int h = input.length;
		int w = input[0].length;

		float sum = 0.0f;

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				sum += input[i][j];
			}
		}

		return sum;
	}
	public static float sum(float[] input) {
		int h = input.length;

		double sum = 0.0f;

		for (int i = 0; i < h; i++) {
			sum += input[i];
		}

		return (float)sum;
	}

	/*** Compute the max of an array as a reduction.
	 */
	public static int max(final int[] values) {
		int maxValue = values[0];
		for (int i = 1; i < values.length; i++) {
			maxValue = Math.max(maxValue, values[i]);
		}
		return maxValue;
	}


	/*** Compute the minimum of two arrays.
	 */
	public static float[][] minimum(float[][] a, float[][] b) {
		// a = min(a, b)
		// assuming dimensions of a and b are the same
		int h = a.length;
		int w = a[0].length;

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				a[i][j] = Math.min(a[i][j], b[i][j]);
			}
		}
		return a;
	}
	public static float[] minimum(float[] a, float[] b) {
		// a = min(a, b)
		// assuming dimensions of a and b are the same
		int h = a.length;

		for (int i = 0; i < h; i++) {
			a[i] = Math.min(a[i], b[i]);
		}
		return a;
	}

	/** Assign b to a (a = b).
	 */
	public static float[][] assign(float[][] a, float[][] b) {
		int h = a.length;
		int w = a[0].length;

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				a[i][j] = b[i][j];
			}
		}
		return a;
	}
	/** Assign b to a (a = b).
	 */
	public static float[] assign(float[] a, float[] b) {
		for (int i = 0; i < a.length; i++) {
			a[i] = b[i];
		}
		return a;
	}


	/** Multiply a and b and return a new array.
	 */
	public static float[][] multiply(float[][] a, float[][] b) {
		int h = a.length;
		int w = a[0].length;

		float[][] result = new float[h][w];

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				result[i][j] = a[i][j] * b[i][j];
			}
		}
		return result;
	}
	public static float[] multiply(float[] a, float[] b) {
		int h = a.length;

		float[] result = new float[h];

		for (int i = 0; i < h; i++) {
			result[i] = a[i] * b[i];
		}
		return result;
	}




	// assumes that input is of size float[h + (filterSize/2)*2][w + (filterSize/2)*2]
	// returns output[h][w]
	public static float[][] convolve(int h, int w, int filterSize, float[][] input) {
		float[][] output = new float[h][w];

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				float sum = 0.0f;
				for (int fi = 0; fi < filterSize; fi++) {
					for (int fj = 0; fj < filterSize; fj++) {
						sum += input[i + fi][j + fj];
					}
				}
				output[i][j] = sum / (filterSize * filterSize);
			}
		}
		return output;
	}
	public static float[] convolve(int h, int w, int filterSize, float[] input) {
		float[] output = new float[h*w];

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				float sum = 0.0f;
				for (int fi = 0; fi < filterSize; fi++) {
					for (int fj = 0; fj < filterSize; fj++) {
						sum += input[(i + fi)*(w+(filterSize/2)*2)+(j + fj)];
					}
				}
				output[i*w+j] = sum / (filterSize * filterSize);
			}
		}
		return output;
	}

	public static float[][] from1DTo2D(int h, int w, float[] input) {
		float[][] output = new float[h][w];

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				output[i][j] = input[i * w + j];
			}
		}
		return output;
	}


	public static float[] from2DTo1D(int h, int w, float[][] input) {
		float[] output = new float[h * w];

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				output[i * w + j] = input[i][j];
			}
		}
		return output;
	}

	public static void compare1D2DArray(float[] array, float[][] expected,
			float diff) {
		float[][] array2D = from1DTo2D(expected.length, expected[0].length,
				array);
		compare2DArray(array2D, expected, diff);
	}
}
