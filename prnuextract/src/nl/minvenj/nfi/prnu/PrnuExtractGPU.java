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
import nl.minvenj.nfi.prnu.compare.NormalizedCrossCorrelation;
import nl.minvenj.nfi.prnu.compare.PatternComparator;

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
    double[][] computeNCC(String[] filenames, float[][] patternsGPU, boolean CPU) {
        int numfiles = filenames.length;
        double cortable[][] = new double[numfiles][numfiles];
        long start, end;

        NormalizedCrossCorrelation NCC = new NormalizedCrossCorrelation(
                this.height,
                this.width, filterFactory.getContext(),
                filterFactory.compile("NormalizedCrossCorrelation.cu"), 
                filterFactory.compile("PeakToCorrelationEnergy.cu"));

        System.out.println("Comparing patterns...");

        double[] sumsquares = new double[numfiles];
        for (int i=0; i<numfiles; i++) {
            if (CPU) { 
                sumsquares[i] = NormalizedCrossCorrelation.sumSquared(cache.retrieve(filenames[i]));
            } else {
                sumsquares[i] = NCC.sumSquaredGPU(cache.retrieve(filenames[i]));
            }
        }
        System.out.println("Finished computing sum of squares");

        //compute standard deviation of all
        double stddev = 0.0;
        for (int i=0; i<numfiles; i++) {
            stddev += sumsquares[i];
        }
        stddev = Math.sqrt(stddev / (double)numfiles);
        System.out.println("Standard deviation of all patterns combined is: " + stddev + " 1/sigma= " + 1.0/stddev);

        if (CPU) {
            //compare all patterns one-by-one on the CPU
            int total = (numfiles*numfiles)/2 - numfiles/2;
            int c = 0;
            System.out.println("     ");

            for (int i=0; i<numfiles; i++) {
              for (int j=0; j<i; j++) {
                cortable[i][j] = NormalizedCrossCorrelation.compare(cache.retrieve(filenames[i]), sumsquares[i], cache.retrieve(filenames[j]), sumsquares[j]);
                cortable[j][i] = cortable[i][j];
                c++;

                if (c % 50 == 0) {
                  System.out.format("\r Progress: %2.2f %%", (((float)c/(float)total)*100.0));
                }
              }
            }
        
        } else {
            //do a blocked comparison on the GPU
            cortable = blockedCompare(filenames, NCC.num_patterns, NCC);

            //do the divison on the CPU
            for (int i=0; i<numfiles; i++) {
                for (int j=0; j<numfiles; j++) {
                    cortable[i][j] = cortable[i][j] / Math.sqrt(sumsquares[i] * sumsquares[j]);
                }
            }


        }
        NCC.printTime();

        return cortable;
    }



    /**
     * This method performs a block-tiled comparison of PRNU patterns
     *
     * @param filenames     a String array of filenames of the JPG files to be compared
     * @param block_size    an int describing the block size of the block-tiled loop
     * @param PSM           a PRNU pattern similarity metric that implements the PatternComparator interface
     */
    public double[][] blockedCompare(String[] filenames, int block_size, PatternComparator PSM) {

        int numfiles = filenames.length;
        double cortable[][] = new double[numfiles][numfiles];
        int c = 0;
        int total = numfiles*numfiles/2 - numfiles/2;

        //stuff needed for loop blocking
        int iblocks = (int)Math.ceil(numfiles/(double)block_size);
        boolean[][] predicate = new boolean[block_size][block_size];
        double[][] result;
        float[][] xPatterns = new float[block_size][];
        float[][] yPatterns = new float[block_size][];

        //compare patterns and print edgelist
        for (int i=0; i<iblocks; i++) {
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

                result = PSM.compareGPU(xPatterns, yPatterns, predicate);

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

                        }
                    }
                }


            } //end of blocked loop
        } 

        System.out.println(); // end the line with progress report
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

        PeakToCorrelationEnergy PCE = new PeakToCorrelationEnergy(
                this.height,
                this.width, filterFactory.getContext(), 
                filterFactory.compile("PeakToCorrelationEnergy.cu"), useRealPeak);

        System.out.println("Comparing patterns...");

        cortable = blockedCompare(filenames, PCE.num_patterns, PCE);

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
        start = System.nanoTime();

        //depending on the mode call the appropiate routine for computing the similarity metric
        switch (mode) {
            case "NCCcpu":
                cortable = computeNCC(filenames, patternsGPU, true);
                break;
            case "NCC":
                cortable = computeNCC(filenames, patternsGPU, false);
                break;
            case "PCE":
                cortable = computePCE(filenames, patternsGPU, true);
                break;
            case "PCE0":
                cortable = computePCE(filenames, patternsGPU, false);
                break;
            default:
                throw new IllegalArgumentException("Invalid mode use NCC(cpu)|PCE|PCE0: " + mode);
        }

        end = System.nanoTime();
        System.out.println("Computing similarity scores took " + (end-start)/1e9 + " seconds.");

        //write edgelist
        write_edgelist(cortable, filenames);

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
        String[] supportedModes = {"NCC", "NCCcpu", "PCE", "PCE0"};
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
     * This method writes a correlation matrix to an edgelist text file
     * The location of the text file is determined by the name of the testcase set by the user
     * 
     * @param cortable      a double matrix, with equal width and height, storing the results of a computation of a similarity metric or correlation
     * @param filenames     a String array containing the filenames that were compared in the correlation
     */
    void write_edgelist(double[][] cortable, String[] filenames) {
        PrintWriter edgefile = null;
        try {
            edgefile = new PrintWriter(EDGELIST_FILENAME);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        int n = cortable[0].length;
        for (int i=0; i<n; i++) {
            for (int j=0; j<i; j++) {
                 edgefile.println(filenames[i] + " " + filenames[j] + " " + cortable[i][j]);
            }
        }
        edgefile.close();        
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
