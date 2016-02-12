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
import nl.minvenj.nfi.prnu.compare.PeakToCorrelationEnergy;
import nl.minvenj.nfi.prnu.NormalizedCrossCorrelation;

import nl.minvenj.nfi.prnu.Util;

//stuff for output
import java.io.PrintWriter;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.DoubleBuffer;
import java.nio.channels.FileChannel;
import java.util.Comparator;
import java.util.Arrays;

import java.io.*;



/**
 * PrnuExtractGPU is a test application for the PRNU filter pipeline.
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */
public class PrnuExtractGPU {

    FastNoiseFilter fastNoiseFilter;
    ZeroMeanTotalFilter zeroMeanTotalFilter;
    WienerFilter wienerFilter;
    GrayscaleFilter grayscaleFilter;

    PRNUFilterFactory filterFactory;
    PRNUFilter filter;

    PrnuPatternCache cache;

    int width;
    int height;

    String testcase;
    String EDGELIST_FILENAME;
    String MATRIX_BIN_FILENAME;
    String MATRIX_TXT_FILENAME;

//    static final File TESTDATA_FOLDER = new File("/var/scratch/bwn200/Dresden/2748x3664");
    File TESTDATA_FOLDER;
//    static final File TESTDATA_FOLDER = new File("/var/scratch/bwn200/PRNUtestcase");
    static final String TEMP_DIR = "/var/scratch/bwn200/patterns/";
//    static final String TEMP_DIR = "/var/scratch/bwn200/temp/";

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


    void extractPatterns(String[] filenames, float[][] patternsGPU, File[] input_files) {
        int numfiles = filenames.length;
        //extract patterns
        System.out.println("Extracting patterns...");
        long start = System.nanoTime();
        for (int i=0; i<filenames.length; i++) {
            BufferedImage image = null;
            try {
                image = Util.readImage(input_files[i]);
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(1);
            }
            patternsGPU[i] = filter.apply(image);

          //write to cache
          //write_float_array_to_file(patternsGPU[i], filenames[i], patternSize);
          //patternsGPU[i] = null;
          //read from cache 
          //patternsGPU[i] = read_float_array_from_file(filenames[i], patternSize);

          input_files[i] = null;

        }
        long end = System.nanoTime();
        System.out.println("Read and extracted " + numfiles + " images in " + (end-start)/1e9 + " seconds.");
    }

    //compute NCC scores
    double[][] computeNCC(String[] filenames, float[][] patternsGPU) {
        int numfiles = filenames.length;
        double cortable[][] = new double[numfiles][numfiles];

        System.out.println("Comparing patterns...");
        PrintWriter edgefile = null;
        try {
            edgefile = new PrintWriter(EDGELIST_FILENAME);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        double[] sumsquares = new double[numfiles];
        for (int i=0; i<numfiles; i++) {
//            sumsquares[i] = NormalizedCrossCorrelation.sumSquared(patternsGPU[i]);
            sumsquares[i] = NormalizedCrossCorrelation.sumSquared(cache.retrieve(filenames[i]));
        }
        System.out.println("Finished computing sum of squares");

        //compute standard deviation of all
        double stddev = 0.0;
        for (int i=0; i<numfiles; i++) {
            stddev += sumsquares[i];
        }
        stddev = Math.sqrt(stddev / (double)numfiles);
        System.out.println("Standard deviation of all patterns combined is: " + stddev + " 1/sigma= " + 1.0/stddev);

        int total = (numfiles*numfiles)/2 - numfiles/2;
        int c = 0;
        System.out.println("     ");

        //compare patterns and print edgelist
        for (int i=0; i<numfiles; i++) {
          //patternsGPU[i] = read_float_array_from_file(filenames[i], patternSize);
          for (int j=0; j<i; j++) {
            //patternsGPU[j] = read_float_array_from_file(filenames[j], patternSize);
//            cortable[i][j] = NormalizedCrossCorrelation.compare(patternsGPU[i], sumsquares[i], patternsGPU[j], sumsquares[j]);
            cortable[i][j] = NormalizedCrossCorrelation.compare(cache.retrieve(filenames[i]), sumsquares[i], cache.retrieve(filenames[j]), sumsquares[j]);

            edgefile.println(filenames[i] + " " + filenames[j] + " " + cortable[i][j]);
            //System.out.println(filenames[i] + " " + filenames[j] + " " + cortable[i][j]);
            cortable[j][i] = cortable[i][j];

            //patternsGPU[j] = null;
            c++;

            if (c % 50 == 0) {
              System.out.format("\r Progress: %2.2f %%", (((float)c/(float)total)*100.0));
              edgefile.flush();
            }
          }
          //patternsGPU[i] = null;
        }
        edgefile.close();

        return cortable;
    }

        //GPU version
    double[][] computePCE(String[] filenames, float[][] patternsGPU, boolean useRealPeak) {
        int numfiles = filenames.length;
        double cortable[][] = new double[numfiles][numfiles];

        PeakToCorrelationEnergy PCE = new PeakToCorrelationEnergy(
                this.height,
                this.width, filterFactory.getContext(), 
                filterFactory.compile("PeakToCorrelationEnergy.cu"), useRealPeak);

        System.out.println("Comparing patterns...");

        PrintWriter edgefile = null;
        try {
            edgefile = new PrintWriter(EDGELIST_FILENAME);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        int total = (numfiles*numfiles)/2 - numfiles/2;
        int c = 0;
        int cmod = 0;
        System.out.println("     ");

        long start = System.nanoTime();

/* one-for-one comparison

        for (int i=0; i<numfiles; i++) {
          //patternsGPU[i] = filter.apply(Util.readImage(INPUT_FILES[i]));
          //patternsGPU_fft[i] = read_float_array_from_file(filenames[i], patternSize);
          //patternsGPU[i] = read_float_array_from_file(filenames[i], patternSize);

          for (int j=0; j<i; j++) {
            //patternsGPU[j] = filter.apply(Util.readImage(INPUT_FILES[j]));
            //patternsGPU_fft[j] = read_float_array_from_file(filenames[j], patternSize);

            //start = System.nanoTime();            
            //patternsGPU[j] = read_float_array_from_file(filenames[j], patternSize);
            //end = System.nanoTime();
            //double readtime = (end-start)/1e6;
            //System.out.println("reading file" + filenames[i] + "took: " + readtime + " ms.");

            //start = System.nanoTime();            
            //cortable[i][j] = PCE.compare(patternsGPU[i], patternsGPU[j]);
            //end = System.nanoTime();
            //double cputime = (end-start)/1e6;

            //long start_gpu = System.nanoTime();
            cortable[i][j] = PCE.compareGPU(patternsGPU[i], patternsGPU[j]);
            //long end_gpu = System.nanoTime();
            //double gputime = (end_gpu-start_gpu)/1e6;
            //System.out.println("GPU PCE took:" + gputime + " ms.");

            edgefile.println(filenames[i] + " " + filenames[j] + " " + cortable[i][j]);
            //System.out.println(filenames[i] + " " + filenames[j] + " cpu=" + cortable[i][j] + " " + cputime + " ms." +
            //                        " gpu=" + GPU_SCORE + " " + gputime + " ms." );
            cortable[j][i] = cortable[i][j];
            c++;

            if (c > 4) {
                System.exit(1);
            } else {
                double check_score_cpu = PCE.compare(patternsGPU[i], patternsGPU[j]);
                System.out.println("Verify score using one-on-one GPU: " + cortable[i][j] + " CPU:" + check_score_cpu);

            }

            if (c % 50 == 0) {
                //long start_output = System.nanoTime();
                //System.out.println("     ");
                System.out.format("\r Progress: %2.2f %%", (((float)c/(float)total)*100.0));
                   //System.out.print("\n");
                edgefile.flush();
                //System.out.println("Format progress output and edgefile.flush took:" + (System.nanoTime()-start_output)/1e6 + " ms.");

            }
            //patternsGPU[j] = null;
          }
          //patternsGPU[i] = null;
        }
        end = System.nanoTime();
        System.out.println("Computed PCE scores for " + numfiles + " images in " + (end-start)/1e9 + " seconds.");

        edgefile.close();

*/
//blocked comparison

        //PCE constructor determines the amount of GPU memory available and chooses a num_patterns accordingly
        int block_size = PCE.num_patterns;
        float[][] xPatterns = new float[block_size][];
        float[][] yPatterns = new float[block_size][];

        boolean debugPrint = false;

        //small part of the correlation table that will be used to temporarily store the resulting PCE scores of this block
        double[][] result;

        //small predicate table of block_size*block_size used to control which comparisons are made
        //the iteration space of the blocked loop may exceed that of the original domain, therefore some blocks will be partially predicated
        boolean[][] predicate = new boolean[block_size][block_size];

        //number of blocks in the i-direction (basically the number of rows containing blocks)
        int iblocks = (int)Math.ceil(numfiles/(double)block_size);

        for (int i=0; i<iblocks; i++) {
            //the number of blocks on this row, +1 to have at least one block and to include the diagonal
            int jblocks = i+1;

            for (int j=0; j<jblocks; j++) {
                int non_null = 0;

                //set all patterns to null and all predicates to false
                for (int ib=0; ib<block_size; ib++) {
                    xPatterns[ib] = null;
                    yPatterns[ib] = null;
                    for (int jb=0; jb<block_size; jb++) {
                         predicate[ib][jb] = false;
                    }
                }

                //add Patterns to be compared to Pattern arrays and set predicate
                for (int ib=0; ib < block_size; ib++) {
                    int gi = i*block_size+ib;

                    //check if the (globally-indexed) row is part of the to be computed domain
                    if (gi > 0 && gi < numfiles) {
//                        xPatterns[ib] = patternsGPU[gi];
                        xPatterns[ib] = cache.retrieve(filenames[gi]);
                        //if (debugPrint) { System.out.println("xPatterns["+ib+"]=" + filenames[gi]); }
                    } else {
                        //if not predicate entire row
                        for (int jb=0; jb<block_size; jb++) {
                            predicate[ib][jb] = false;
                        }
                    }

                    //for each column (gj) within this block on this row (gi)
                    for (int jb=0; jb<block_size; jb++) {
                        int gj = j*block_size+jb;
                        //check if the (gi,gj) pair needs to be computed
                        if (gi < numfiles && gj < gi) {
                            //yPatterns[jb] = patternsGPU[gj];
                            yPatterns[jb] = cache.retrieve(filenames[gj]);
                            //if (debugPrint) { System.out.println("yPatterns["+jb+"]=" + filenames[gj]); }
                            predicate[ib][jb] = true;
                            non_null++;
                        } else {
                            predicate[ib][jb] = false;
                        }

                    }
                }

                if (debugPrint) {
                    System.out.println("predicate matrix:");
                    for (int ib=0; ib<block_size; ib++) {
                        for (int jb=0; jb<block_size; jb++) {
                            System.out.print(predicate[ib][jb] + " ");
                        }
                        System.out.println();
                    }
                    System.out.println();
                }

                //finally start the computation for this block
                //long start_gpu = System.nanoTime();
                result = PCE.compareGPU(xPatterns, yPatterns, predicate);
                //long end_gpu = System.nanoTime();
                //double gpu_time = (end_gpu - start_gpu) / 1e6;
                //if (debugPrint) { 
                //    System.out.println("Blocked PCE GPU took: " + gpu_time + " ms. on average: " + gpu_time/(non_null) + " ms.");
                //}

                c += non_null;
//                cmod += non_null;
//                if (cmod > 50) {
//                    cmod -= 50;
                    System.out.format("\r Progress: %2.2f %%", (((float)c/(float)total)*100.0));
//                }

                //fill the cortable with the result
                for (int ib=0; ib < block_size; ib++) {
                    for (int jb=0; jb<block_size; jb++) {

                        //obtain global indexes
                        int gi = i*block_size+ib;
                        int gj = j*block_size+jb;
                        if (gi < numfiles && gj < gi && predicate[ib][jb]) {
                            cortable[gi][gj] = result[ib][jb];
                            cortable[gj][gi] = cortable[gi][gj];
                            edgefile.println(filenames[gi] + " " + filenames[gj] + " " + cortable[gi][gj]);
                            //System.out.println(filenames[gi] + " " + filenames[gj] + " " + cortable[gi][gj]); // debug
                            if (debugPrint) {
                                System.out.println(filenames[gi] + " " + filenames[gj] + " " + cortable[gi][gj]);
                                double check_score_gpu = PCE.compareGPU(patternsGPU[gi], patternsGPU[gj]);
                                double check_score_cpu = PCE.compare(patternsGPU[gi], patternsGPU[gj]);
                                System.out.println("Verify score using one-on-one GPU: " + check_score_gpu + " CPU:" + check_score_cpu);
                            }

                        }
                    }
                }                            
                

            }
        }

        long end = System.nanoTime();
        System.out.println("\rComputed PCE scores for " + numfiles + " images in " + (end-start)/1e9 + " seconds.");

        edgefile.close();

        return cortable;
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
    public void run(String testcase, String folderpath, String mode) throws IOException {

        long start = 0;
        long end = 0;
        
        this.testcase = testcase;
        this.EDGELIST_FILENAME = "/var/scratch/bwn200/edgelist-" + testcase + ".txt";
        this.MATRIX_BIN_FILENAME = "/var/scratch/bwn200/matrix-" + testcase + ".dat";
        this.MATRIX_TXT_FILENAME = "/var/scratch/bwn200/matrix-" + testcase + ".txt";
        this.TESTDATA_FOLDER = new File(folderpath);

        //instantiate the PRNUFilterFactory to compile CUDA source files
        this.filterFactory = new PRNUFilterFactory();
        int numfiles = TESTDATA_FOLDER.listFiles().length;
        File INPUT_FILES[] = new File[numfiles];

        File[] files = TESTDATA_FOLDER.listFiles();
        Arrays.sort(files, new Comparator<File>() {
            public int compare(File f1, File f2) {
                return f1.getName().compareTo(f2.getName());
            }
        });

        INPUT_FILES = files;
        numfiles = files.length;

        String[] filenames = new String[numfiles];
        for (int i=0; i<numfiles; i++) {
            filenames[i] = INPUT_FILES[i].getName();
        }

        float[][] patternsGPU = new float[numfiles][];

        BufferedImage image = Util.readImage(INPUT_FILES[0]);
        this.height = image.getHeight();
        this.width = image.getWidth();
        System.out.println("Image size: " + this.height + "x" + this.width);

        this.filter = filterFactory.createPRNUFilter(image.getHeight(), image.getWidth());

        int patternSize = height*width;

        //either extract all patterns here or use cache
        //extractPatterns(filenames, patternsGPU, INPUT_FILES);
        //free resources on the GPU
        //filter.cleanup();


        //clear up some stuff
        image = null;
        for (int i=0; i<numfiles; i++) {
            INPUT_FILES[i] = null;
        }
        System.gc();

        //use cache instead
        cache = new PrnuPatternCache(height, width, filter, folderpath);
        start = System.nanoTime();
        cache.populate(filenames);
        end = System.nanoTime();
        System.out.println("Populating the cache took " + (end-start)/1e9 + " seconds.");


        double cortable[][];

        switch (mode) {
            case "NCC":
                cortable = computeNCC(filenames, patternsGPU);
                break;
            case "PCE":
                cortable = computePCE(filenames, patternsGPU, true);
                break;
            case "PCE0":
                cortable = computePCE(filenames, patternsGPU, false);
                break;
            default:
                throw new IllegalArgumentException("Invalid mode use NCC|PCE|PCE0: " + mode);
        }

        //print results
        write_matrix_text(cortable);
        write_matrix_binary(cortable);

    }

    public static boolean containsIllegals(String toExamine) {
        String[] arr = toExamine.split("[~#@*+%{}<>\\[\\]|\"\\_^]", 2);
        return arr.length > 1;
    }

    public static void printUsage() {
        System.out.println("Usage: <program-name> [testcase] [folderpath] [mode]");
        System.out.println("    testcase is the name you give to this run");
        System.out.println("    folderpath is the path to the folder containing images");
        System.out.println("    mode is any of NCC, PCE, or PCE0");
        System.exit(0);
    }

    public static void main(final String[] args) throws IOException {

        if (args.length != 3) {
            printUsage();
        }
        if (containsIllegals(args[0])) {
            System.out.println("testcase will be used for filenames, please do not use special characters");
            printUsage();
        }
        File f = new File(args[1]);
        if (!f.exists() || !f.isDirectory()) {
            System.out.println("folderpath does not exist or is not a directory");
            printUsage();
        }
        String[] supportedModes = {"NCC", "PCE", "PCE0"};
        if (!Arrays.asList(supportedModes).contains(args[2])) {
            System.out.println("Unknown mode: " + args[2]);
            printUsage();
        }

        new PrnuExtractGPU().run(args[0], args[1], args[2]);

        System.out.println("done");
        
        //exit because the JTransforms library used in Wienerfilter takes a minute to time out
        System.exit(0);
    }


        //dump matrix
void write_matrix_binary(double[][] cortable) {
   int numfiles = cortable[0].length;

   try{
    FileOutputStream fos = new FileOutputStream(MATRIX_BIN_FILENAME); 
    DataOutputStream dos = new DataOutputStream(fos);
    for (int i=0; i<numfiles; i++) {
        for (int j=0; j<numfiles; j++) {
                dos.writeDouble(cortable[i][j]);
            }
        }

    /*
        RandomAccessFile aFile = new RandomAccessFile("/var/scratch/bwn200/matrix.dat", "rw");
        FileChannel outChannel = aFile.getChannel();

        //one float 3 bytes
        ByteBuffer buf = ByteBuffer.allocate(8*numfiles*numfiles);
        buf.clear();
    DoubleBuffer db = buf.asDoubleBuffer();
    for (int i=0; i<numfiles; i++) {
        for (int j=0; j<numfiles; j++) {
                db.put(cortable[j][i]);
            }
        }

    db.rewind();
        outChannel.write(buf);
        outChannel.close();
    */
    }
    catch (Exception ex) {
        System.err.println(ex.getMessage());
    }

}


  void write_matrix_text(double[][] cortable) {
    int numfiles = cortable[0].length;
    try{
    //System.out.println("PCE Scores:");

    PrintWriter textfile = new PrintWriter(MATRIX_TXT_FILENAME);
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
    }
    catch (Exception ex) {
        System.err.println(ex.getMessage());
    }
  }


  void write_float_array_to_file(float[] array, String filename, int size) {
    /*
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
    */
    String file = TEMP_DIR + filename.substring(0, filename.lastIndexOf('.')) + ".dat";

    try {
      DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)));
 
      for (int i=0;i<size;i++) {
           out.writeFloat(array[i]);
      }

      out.close();

    }
    catch (IOException ex) {
        System.err.println(ex.getMessage());
    }

  }


float[] read_float_array_from_file(String filename, int size) {

    /*
        RandomAccessFile aFile = new RandomAccessFile(TEMP_DIR + filename.substring(0, filename.lastIndexOf('.')) +  ".dat", "rw");
        FileChannel inChannel = aFile.getChannel();

        ByteBuffer buf = ByteBuffer.allocate(size);
        buf.clear();

        while(buf.hasRemaining())
        {
            inChannel.read(buf);
        }

        inChannel.close();
        float[] result = new float[size/4];
        buf.rewind();

        FloatBuffer fbuf = buf.asFloatBuffer();

        fbuf.get(result);

        return result;
    */
    String file = TEMP_DIR + filename.substring(0, filename.lastIndexOf('.')) + ".dat";

    try{
        DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));

        float [] result = new float[size];

        for (int i=0;i<size;i++) {
            result[i] = in.readFloat();
        }    

        return result;
    }
    catch (IOException ex) {
        System.err.println(ex.getMessage());
    }

    return null;

}



}
