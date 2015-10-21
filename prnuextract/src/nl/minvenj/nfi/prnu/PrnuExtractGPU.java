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

import java.io.IOException;
import java.io.File;
import java.awt.image.BufferedImage;
import nl.minvenj.nfi.prnu.filtergpu.*;
import nl.minvenj.nfi.prnu.PeakToCorrelationEnergy;

import nl.minvenj.nfi.prnu.Util;

//stuff for output
import java.io.PrintWriter;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;


/**
 * PrnuExtractGPU is a test application for the PRNU filter pipeline.
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.2a 
 */
public class PrnuExtractGPU {

	FastNoiseFilter fastNoiseFilter;
	ZeroMeanTotalFilter zeroMeanTotalFilter;
	WienerFilter wienerFilter;
	GrayscaleFilter grayscaleFilter;

        PeakToCorrelationEnergy PCE;

	static final File TESTDATA_FOLDER = new File("/var/scratch/bwn200/Dresden/2748x3664");
	static final String TEMP_DIR = "/var/scratch/bwn200/temp/";

	/**
	 * This method performs the PRNU pattern extraction on the CPU, it is only included
	 * here for comparison with the GPU results and performance
	 * 
	 * @param height - image height in pixels
	 * @param width - image width in pixels
	 * @param image - the input image as a BufferedImage
	 * @return pixels - float array for storing the extracted pattern
	 */
	private float[] extractImageCPU(BufferedImage image) {
		long start = System.nanoTime();
		long end = 0;
		
		start = System.nanoTime();
		float[] pixels = grayscaleFilter.apply1D(image);
		end = System.nanoTime();
		System.out.println("grayscale image CPU: " + (end-start)/1e6 + " ms.");

		start = System.nanoTime();
		pixels = fastNoiseFilter.apply1D(pixels);
		end = System.nanoTime();
		System.out.println("Fast Noise Filter: " + (end-start)/1e6 + " ms.");

		start = System.nanoTime();
		pixels = zeroMeanTotalFilter.apply1D(pixels);
		end = System.nanoTime();
		System.out.println("Zero Mean Filter: " + (end-start)/1e6 + " ms.");

		start = System.nanoTime();
		pixels = wienerFilter.apply1D(pixels);
		end = System.nanoTime();
		System.out.println("Wiener Filter: " + (end-start)/1e6 + " ms.");
		
		return pixels;
	}

	/**
	 * This is the main non-static method of this test application.
	 * It shows how the PRNUFilterFactory and PRNUFilter classes should be used.
	 * 
	 * After the pattern is extracted for a test image the pattern is also
	 * extracted on the CPU and the result is compared. The CPU implementation
	 * directly uses the individual filters, which is not necessary normally.
	 * 
	 * @throws IOException - an IOException is thrown when the input image cannot be read
	 */
	public void run() throws IOException {

	        long start = 0;
        	long end = 0;
        	BufferedImage image;
		
		//instantiate the PRNUFilterFactory to compile CUDA source files
		PRNUFilterFactory filterFactory = new PRNUFilterFactory();

		int numfiles = TESTDATA_FOLDER.listFiles().length;
		File INPUT_FILES[] = new File[numfiles];

		int f=0;
		for (File file : TESTDATA_FOLDER.listFiles()) { 
		    INPUT_FILES[f++] = file;
		}
		numfiles = f;
		String[] filenames = new String[numfiles];
		float[][] patternsGPU = new float[numfiles][];
		float[][] patternsGPU_fft = new float[numfiles][];
		double cortable[][] = new double[numfiles][numfiles];

		image = Util.readImage(INPUT_FILES[0]);
		System.out.println("Image size: " + image.getHeight() + "x" + image.getWidth());
		int h = image.getHeight();
		int w = image.getWidth();

		PRNUFilter filter = filterFactory.createPRNUFilter(image.getHeight(), image.getWidth());
		PeakToCorrelationEnergy PCE = new PeakToCorrelationEnergy (image.getHeight(),
								image.getWidth(), 11);

		int patternSize = 2 * 4 * h * w; //complex * sizeof(float) * h * w

		//extract patterns
		System.out.println("Extracting patterns...");
		start = System.nanoTime();
		for (int i=0; i<numfiles; i++) {
		  image = Util.readImage(INPUT_FILES[i]);
		  patternsGPU[i] = filter.apply(image);
		  patternsGPU_fft[i] = Util.toComplex(h, w, patternsGPU[i]);
		  PCE._fft.complexForward(patternsGPU_fft[i]);

		  filenames[i] = INPUT_FILES[i].getName();
		  write_float_array_to_file(patternsGPU_fft[i], filenames[i], patternSize);
		  
		  patternsGPU_fft[i] = null;
		  INPUT_FILES[i] = null;
		  patternsGPU[i] = null;
		  System.gc();

		}
		end = System.nanoTime();
		System.out.println("Read and extracted " + numfiles + " images in " + (end-start)/1e9 + " seconds.");

		//compute PCE scores
		System.out.println("Comparing patterns...");
		PrintWriter edgefile = new PrintWriter("/var/scratch/bwn200/edgelist.txt");

		start = System.nanoTime();
		for (int i=0; i<numfiles; i++) {
		  patternsGPU_fft[i] = read_float_array_from_file(filenames[i], patternSize);
		  for (int j=0; j<i; j++) {
		    patternsGPU_fft[j] = read_float_array_from_file(filenames[j], patternSize);

		    cortable[i][j] = PCE.compare_fft(patternsGPU_fft[i], patternsGPU_fft[j]);
		    edgefile.println(INPUT_FILES[i].getName() + " " + INPUT_FILES[j].getName() + " " + cortable[i][j]);
		    cortable[j][i] = cortable[i][j];

		    patternsGPU_fft[j] = null;
		    System.gc();
		  }
		  patternsGPU_fft[i] = null;
    		  System.gc();
		}
		end = System.nanoTime();
		System.out.println("Computed PCE scores for " + numfiles + " images in " + (end-start)/1e9 + " seconds.");

		edgefile.close();

		//print results
		//System.out.println("PCE Scores:");
		PrintWriter textfile = new PrintWriter("/var/scratch/bwn200/matrix.txt");
		for (int i=0; i<numfiles; i++) {
		  for (int j=0; j<numfiles; j++) {
		    //System.out.format("%.6f, ", cortable[i][j]);
		    textfile.format("%.6f, ", cortable[i][j]);
		  }
		  //System.out.println();
		  textfile.println();
		}
		//System.out.println();
  		textfile.println();

		textfile.close();


		//dump matrix
   try{
        RandomAccessFile aFile = new RandomAccessFile("/var/scratch/bwn200/matrix.dat", "rw");
        FileChannel outChannel = aFile.getChannel();

        //one float 3 bytes
        ByteBuffer buf = ByteBuffer.allocate(8*numfiles*numfiles);
        buf.clear();
	for (double[] tableRow : cortable) {
	        buf.asDoubleBuffer().put(tableRow);
        }

        //while(buf.hasRemaining()) 
        {
            outChannel.write(buf);
        }

        outChannel.close();

    }
    catch (IOException ex) {
        System.err.println(ex.getMessage());
    }



	}


	public static void main(final String[] args) throws IOException {
		System.out.println("FilterGPU");

		new PrnuExtractGPU().run();

		System.out.println("done");
		
		//exit because the JTransforms library used in Wienerfilter takes a minute to time out
		System.exit(0);
	}







  void write_float_array_to_file(float[] array, String filename, int size) {

     try{
        RandomAccessFile aFile = new RandomAccessFile(TEMP_DIR + filename.substring(0, filename.lastIndexOf('.')) +  ".dat", "rw");
        FileChannel outChannel = aFile.getChannel();

        ByteBuffer buf = ByteBuffer.allocate(size);
        buf.clear();
	buf.asFloatBuffer().put(array);

        //while(buf.hasRemaining()) 
        {
            outChannel.write(buf);
        }

        outChannel.close();

      }
      catch (IOException ex) {
        System.err.println(ex.getMessage());
      }

  }


  float[] read_float_array_from_file(String filename, int size) {

     try{
        RandomAccessFile aFile = new RandomAccessFile(TEMP_DIR + filename.substring(0, filename.lastIndexOf('.')) +  ".dat", "rw");
        FileChannel inChannel = aFile.getChannel();

        ByteBuffer buf = ByteBuffer.allocate(size);
        buf.clear();

        //while(buf.hasRemaining()) 
        {
            inChannel.read(buf);
        }

        inChannel.close();
	return buf.asFloatBuffer().array();

      }
      catch (IOException ex) {
        System.err.println(ex.getMessage());
      }


    return null;

  }



}
