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
 * PrnuExtractGPU is an application for extracting and comparing PRNU filters.
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.2
 */
public class PrnuExtractGPU {

    PRNUFilterFactory filterFactory;
    PRNUFilter filter;

    PrnuPatternCache cache;

    int width;
    int height;

    String testcase;
    String EDGELIST_FILENAME;
    String MATRIX_BIN_FILENAME;
    String MATRIX_TXT_FILENAME;

    File TESTDATA_FOLDER;

    //this TEMP_DIR is used by the routines for dumping and reading PRNU patterns to disk
    //the methods for dumping and reading are currently not used since it is much faster
    //to read the JPEG image and recompute the PRNU pattern on the GPU
    //the methods are however occasionally used for debugging purposes
    static final String TEMP_DIR = "";

    /**
     * This methods extracts the PRNU pattern of all the images in the dataset
     * It fills the patternsGPU array, which is an array of PRNU patterns stored as float arrays
     * This method has been replaced by using the PRNU pattern cach instead
     *
     * @param filenames     a String array containing all the filenames of the images to be compared
     * @param patternsGPU   an array of PRNU patterns stored as float arrays, the output of this method
     * @param input_files   an array of File objects pointing to the input files for this dataset
     */
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

          input_files[i] = null;
        }
        long end = System.nanoTime();
        System.out.println("Read and extracted " + numfiles + " images in " + (end-start)/1e9 + " seconds.");
    }

    /**
     * This method computes the NCC on the CPU scores for all images listed in filenames
     *
     * @param filenames     a String array containing all the filenames of the images to be compared
     * @param patternsGPU   an array of PRNU patterns stored as float arrays, no longer in use, using cache instead
     * @returns             a matrix containing the NCC scores for all the images in this dataset
     */
    double[][] computeNCC(String[] filenames, float[][] patternsGPU) {
        int numfiles = filenames.length;
        double cortable[][] = new double[numfiles][numfiles];
        long start, end;
        double cpu_time, gpu_time;
    
        NormalizedCrossCorrelation NCC = new NormalizedCrossCorrelation(
                this.height,
                this.width, filterFactory.getContext(),
                filterFactory.compile("NormalizedCrossCorrelation.cu"), 
                filterFactory.compile("PeakToCorrelationEnergy.cu"));

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

            start = System.nanoTime();
            sumsquares[i] = NCC.sumSquaredGPU(cache.retrieve(filenames[i]));
            gpu_time = (System.nanoTime() - start)/1e6;
            start = System.nanoTime();
            double sumsquared = NormalizedCrossCorrelation.sumSquared(cache.retrieve(filenames[i]));
            cpu_time = (System.nanoTime() - start)/1e6;

            System.out.println("sumSquared CPU: " + sumsquared + " took: " + cpu_time + " ms.");
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

            start = System.nanoTime();
            double nccscore = NormalizedCrossCorrelation.compare(cache.retrieve(filenames[i]), sumsquares[i], cache.retrieve(filenames[j]), sumsquares[j]);
            cpu_time = (System.nanoTime() - start)/1e6;

            start = System.nanoTime();
            cortable[i][j] = NCC.compareGPU(cache.retrieve(filenames[i]), sumsquares[i], cache.retrieve(filenames[j]), sumsquares[j]);
            gpu_time = (System.nanoTime() - start)/1e6;

            edgefile.println(filenames[i] + " " + filenames[j] + " " + cortable[i][j]);
            System.out.println("computeNCC CPU: " + nccscore + " took: " + cpu_time + " ms.");
            cortable[j][i] = cortable[i][j];

            //patternsGPU[j] = null;
            c++;

            if (c % 50 == 0) {
              System.out.format("\r Progress: %2.2f %%", (((float)c/(float)total)*100.0));
              System.out.println(); //temporary
              edgefile.flush();
            }
          }
          //patternsGPU[i] = null;
        }
        edgefile.close();

        NCC.printTime();

        return cortable;
    }


    /**
     * This method computes the PCE scores on the GPU for all images in the array filenames
     *
     * This method uses a block-tiled loop to iterate over its iteration domain,
     * this is to enable data reuse on the GPU and also enables data reuse in the PRNU pattern cache.
     *
     * @param filenames     a String array containing all the filenames of the images to be compared
     * @param patternsGPU   an array of PRNU patterns stored as float arrays, no longer in use, using cache instead
     * @param useRealPeak   a boolean specifying the variant of PCE to be computed, true uses real peak, false uses last pixel instead
     * @returns             a matrix containing the PCE scores for all the images in this dataset
     */
    double[][] computePCE(String[] filenames, float[][] patternsGPU, boolean useRealPeak) {
        int numfiles = filenames.length;
        double cortable[][] = new double[numfiles][numfiles];

        //instantiate the PCE object for making comparisons
        //this starts with obtaining the CudaModule object from the
        //filterFactory which is misused as an interface to the compiler
        PeakToCorrelationEnergy PCE = new PeakToCorrelationEnergy(
                this.height,
                this.width, filterFactory.getContext(), 
                filterFactory.compile("PeakToCorrelationEnergy.cu"), useRealPeak);

        System.out.println("Comparing patterns...");

        //open edgefile for writing output as and edgelist
        PrintWriter edgefile = null;
        try {
            edgefile = new PrintWriter(EDGELIST_FILENAME);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        //the total number of comparisons is N over 2, where N is the number of files
        int total = (numfiles*numfiles)/2 - numfiles/2;
        int c = 0;
        //print a newline for the Progress prints
        System.out.println("            "); 

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
                        //xPatterns[ib] = patternsGPU[gi];
                        xPatterns[ib] = cache.retrieve(filenames[gi]);
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
                System.out.format("\r Progress: %2.2f %%", (((float)c/(float)total)*100.0));

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
                                System.out.println("Verify score using one-by-one GPU: " + check_score_gpu + " CPU:" + check_score_cpu);
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
     * This is the main non-static method of this application.
     * It reads the names of all the image files in the target directory
     * 
     * @param testcase      a String containing name for this run, given as commandline argument
     * @param folderpath    a String containing the path to the target folder
     * @param mode          a String specifying which metric to use, NCC, PCE, or PCE0
     * @throws IOException - an IOException is thrown when an input image cannot be read
     */
    public void run(String testcase, String folderpath, String mode) throws IOException {
        long start = 0;
        long end = 0;
        
        this.testcase = testcase;
        this.EDGELIST_FILENAME = "edgelist-" + testcase + ".txt";
        this.MATRIX_BIN_FILENAME = "matrix-" + testcase + ".dat";
        this.MATRIX_TXT_FILENAME = "matrix-" + testcase + ".txt";
        this.TESTDATA_FOLDER = new File(folderpath);

        //instantiate the PRNUFilterFactory to compile CUDA source files
        this.filterFactory = new PRNUFilterFactory();
        int numfiles = TESTDATA_FOLDER.listFiles().length;
        File INPUT_FILES[] = new File[numfiles];

        //obtain the list of files in this folder and sort them on filename
        File[] files = TESTDATA_FOLDER.listFiles();
        Arrays.sort(files, new Comparator<File>() {
            public int compare(File f1, File f2) {
                return f1.getName().compareTo(f2.getName());
            }
        });
        INPUT_FILES = files;
        numfiles = files.length;

        //extract the filenames from all File objects and store in String array filenames
        String[] filenames = new String[numfiles];
        for (int i=0; i<numfiles; i++) {
            filenames[i] = INPUT_FILES[i].getName();
        }

        //array for storing all the patterns, currently using the pattern cache instead
        float[][] patternsGPU = new float[numfiles][];

        //open one image to obtain the width and height of all images in this folder
        BufferedImage image = Util.readImage(INPUT_FILES[0]);
        this.height = image.getHeight();
        this.width = image.getWidth();
        System.out.println("Image size: " + this.height + "x" + this.width);

        //create the filterfactory, this compiles the CUDA modules that are part of PRNUFilter
        this.filter = filterFactory.createPRNUFilter(image.getHeight(), image.getWidth());

        //either extract all patterns here or use cache
        //extractPatterns(filenames, patternsGPU, INPUT_FILES);
        //free resources on the GPU
        //filter.cleanup();

        //clear up some memory of stuff we no longer need
        image = null;
        for (int i=0; i<numfiles; i++) {
            INPUT_FILES[i] = null;
        }
        System.gc();

        //use cache instead of extracting all patterns
        cache = new PrnuPatternCache(height, width, filter, folderpath);

        //populate the cache with the first n patterns where n is the size of the cache
        start = System.nanoTime();
        cache.populate(filenames);
        end = System.nanoTime();
        System.out.println("Populating the cache took " + (end-start)/1e9 + " seconds.");

        double cortable[][];

        //depending on the mode call the appropiate routine for computing the similarity metric
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

        //write the correlation matrix to disk in binary and text form
        write_matrix_text(cortable);
        write_matrix_binary(cortable);

    }


    /**
     * Simple helper function to detect certain characters that would 
     * be illegal to use for filenames in most file systems
     *
     * @param toExamine     the string to examine
     */
    private static boolean containsIllegals(String toExamine) {
        String[] arr = toExamine.split("[~#@*+%{}<>\\[\\]|\"\\_^]", 2);
        return arr.length > 1;
    }

    /**
     * Simple method that prints the expected usage of the program through the commandline and exits
     */ 
    private static void printUsage() {
        System.out.println("Usage: <program-name> [testcase] [folderpath] [mode]");
        System.out.println("    testcase is the name you give to this run");
        System.out.println("    folderpath is the path to the folder containing images");
        System.out.println("    mode is any of NCC, PCE, or PCE0");
        System.exit(0);
    }

    /**
     * The main routine, it checks the commandline arguments and then calls the non-static run()
     */
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


    /**
     * This method writes a correlation matrix to a binary file
     * The location of the text file is determined by the name of the testcase set by the user
     * Note that Java writes its doubles in big endian
     * 
     * @param cortable a double matrix, with equal width and height, storing the results of a computation of a similarity metric or correlation
     */
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
        } catch (Exception ex) {
            System.err.println(ex.getMessage());
        }
    }

    /**
     * This method writes a correlation matrix to a text file
     * The location of the text file is determined by the name of the testcase set by the user
     * 
     * @param cortable a double matrix, with equal width and height, storing the results of a computation of a similarity metric or correlation
     */
    void write_matrix_text(double[][] cortable) {
        int numfiles = cortable[0].length;
        try {
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
        } catch (Exception ex) {
            System.err.println(ex.getMessage());
        }
    }

    /**
     * This method writes a PRNU pattern to a file on disk
     *
     * This method is now only used for debugging purposes because it is much
     * faster to read the JPEG and recompute the PRNU pattern than reading a
     * stored pattern from disk.
     *
     * @param array     a float array containing the PRNU pattern
     * @param filename  a string containing the name of the JPG file, its current extension will be replaced with '.dat'
     * @param size      the size of the PRNU pattern
     */
    void write_float_array_to_file(float[] array, String filename, int size) {
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

    /**
     * This method reads a PRNU pattern from a file on disk
     *
     * This method is now only used for debugging purposes because it is much
     * faster to read the JPEG and recompute the PRNU pattern than reading a
     * stored pattern from disk.
     *
     * @param filename  the name of the JPEG file whose PRNU pattern we are now fetching from disk
     * @param size      the size of the PRNU pattern in the number of floats
     * @returns         a float array containing the PRNU pattern
     */
    float[] read_float_array_from_file(String filename, int size) {
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
