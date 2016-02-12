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
package nl.minvenj.nfi.prnu.compare;

import static org.junit.Assert.*;

import java.io.File;
import java.awt.image.BufferedImage;

import jcuda.runtime.JCuda;
import jcuda.jcufft.JCufft;
import nl.minvenj.nfi.prnu.filtergpu.PRNUFilter;
import nl.minvenj.nfi.prnu.filtergpu.PRNUFilterFactory;
import nl.minvenj.nfi.prnu.Util;
import nl.minvenj.nfi.cuba.cudaapi.*;
import nl.minvenj.nfi.prnu.compare.PeakToCorrelationEnergy;

import jcuda.driver.*;
import jcuda.jcufft.*;
import jcuda.runtime.cudaError;

import edu.emory.mathcs.jtransforms.fft.FloatFFT_2D;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;

/**
 * Class for applying a series of Wiener Filters to a PRNU pattern 
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */
public class PeakToCorrelationEnergyTest {

    static PRNUFilterFactory filterFactory;
    static PRNUFilter filter;
    static PeakToCorrelationEnergy PCE;
    static float[] x;
    static float[] y;

//    static File testImage1 = new File("/var/scratch/bwn200/PRNUtestcase/Agfa_Sensor505-x_0_1890.JPG");
//    static File testImage2 = new File("/var/scratch/bwn200/PRNUtestcase/Agfa_Sensor505-x_0_1900.JPG");
    static File testImage1 = new File("/var/scratch/bwn200/Dresden/2748x3664/Kodak_M1063_4_12664.JPG");
    static File testImage2 = new File("/var/scratch/bwn200/Dresden/2748x3664/Kodak_M1063_4_12665.JPG");

    /**
     * @throws java.lang.Exception
     */
    @BeforeClass
    public static void setUpBeforeClass() throws Exception {
        //instantiate the PRNUFilterFactory to compile CUDA source files
        filterFactory = new PRNUFilterFactory(false);

        //load test image and measure time
        long start = System.nanoTime();
        long end = 0;
        BufferedImage image = Util.readImage(testImage1);
        BufferedImage image2 = Util.readImage(testImage2);
        end = System.nanoTime();
        System.out.println("Loading images: " + (end-start)/1e6 + " ms. size: " + image.getHeight()+"x"+image.getWidth());

        //construct a PRNUFilter for this image size
        filter = filterFactory.createPRNUFilter(image.getHeight(), image.getWidth());

        //extract PRNU patterns from images
        x = filter.apply(image);

        //boolean testNaN = Util.compareArray(x, x, 10f);
        //if (!testNaN) { System.err.println("constructor input x contains NaN before start"); }
        //assertTrue(testNaN);

        y = filter.apply(image2);

        //free GPU resources
        filter.cleanup();

        //construct PCE instance
        PCE = new PeakToCorrelationEnergy(
                image.getHeight(),
                image.getWidth(), filterFactory.getContext(),
                filterFactory.compile("PeakToCorrelationEnergy.cu"), false);

    }


    public void assertNoCudaError() {
		JCudaDriver.cuCtxSynchronize();
        int err = JCuda.cudaGetLastError();
        if (err != cudaError.cudaSuccess) {
            System.out.println("CUDA Error: " + JCuda.cudaGetErrorString(err));
        } else {
            System.out.println("CUDA: No Errors!");
        }
        assertTrue(err == cudaError.cudaSuccess);
    }

    @Test
    public void toComplexTest() {
        boolean testNaN = Util.compareArray(x, x, 10f);
        if (!testNaN) { System.err.println("toComplexTest input x contains NaN before start"); }
        assertTrue(testNaN);

        PCE._d_inputx.copyHostToDeviceAsync(x, PCE._stream1);
        PCE._tocomplex.launch(PCE._stream1, PCE.toComplex);

        PCE._stream1.synchronize();

        PCE.toComplexAndFlip(x,y);
        //check if PCE._x matches _d_x
        float x_GPU[] = new float[PCE._x.length];
        PCE._d_x.copyDeviceToHost(x_GPU, PCE._x.length);

        boolean result = Util.compareArray(PCE._x, x_GPU, 1f/256f);
        assertTrue(result);

        assertNoCudaError();
    } 

    @Test
    public void toComplexAndFlipTest() {
        PCE._d_inputy.copyHostToDeviceAsync(y, PCE._stream1);
        PCE._tocomplexandflip.launch(PCE._stream1, PCE.toComplexAndFlip);

        PCE._stream1.synchronize();

        PCE.toComplexAndFlip(x,y);
        //check if PCE._y matches _d_y
        float y_GPU[] = new float[PCE._y.length];
        PCE._d_y.copyDeviceToHost(y_GPU, PCE._y.length);

        boolean result = Util.compareArray(PCE._y, y_GPU, 1f/256f);
        assertTrue(result);

        assertNoCudaError();
    } 

    @Test
    public void computeCrossCorrTest() {
        boolean testNaN = Util.compareArray(x, x, 10f);
        if (!testNaN) { System.err.println("computeCrossCorrTest input x contains NaN before start"); }
        assertTrue(testNaN);

        testNaN = Util.compareArray(y, y, 10f);
        if (!testNaN) { System.err.println("computeCrossCorrTest input y contains NaN before start"); }
        assertTrue(testNaN);

        PCE._d_inputx.copyHostToDeviceAsync(x, PCE._stream1);
        PCE._d_inputy.copyHostToDeviceAsync(y, PCE._stream1);

        JCufft.cufftExecC2C(PCE._plan1, PCE._d_x.getDevicePointer(), PCE._d_x.getDevicePointer(), JCufft.CUFFT_FORWARD);
        JCufft.cufftExecC2C(PCE._plan1, PCE._d_y.getDevicePointer(), PCE._d_y.getDevicePointer(), JCufft.CUFFT_FORWARD);

        PCE._computeCrossCorr.launch(PCE._stream1, PCE.computeCrossCorr);
        PCE._stream1.synchronize();

        //transfer _d_x to _x and _d_y to _y
        PCE._d_x.copyDeviceToHost(PCE._x, PCE._x.length);
        PCE._d_y.copyDeviceToHost(PCE._y, PCE._y.length);
        PCE.compute_crosscorr();

        //check if PCE._c matches _d_c
        float c_GPU[] = new float[PCE._c.length];
        PCE._d_c.copyDeviceToHost(c_GPU, PCE._c.length);

        boolean result = Util.compareArray(PCE._c, c_GPU, 1e-5f);
        assertTrue(result);

        assertNoCudaError();
    } 

    @Test
    public void findPeakTest() {
        boolean testNaN = Util.compareArray(x, x, 10f);
        if (!testNaN) { System.err.println("findPeakTest input x contains NaN before start"); }
        assertTrue(testNaN);

        testNaN = Util.compareArray(y, y, 10f);
        if (!testNaN) { System.err.println("findPeakTest input y contains NaN before start"); }
        assertTrue(testNaN);

        PCE._d_inputx.copyHostToDeviceAsync(x, PCE._stream1);
        PCE._d_inputy.copyHostToDeviceAsync(y, PCE._stream1);

        JCufft.cufftExecC2C(PCE._plan1, PCE._d_x.getDevicePointer(), PCE._d_x.getDevicePointer(), JCufft.CUFFT_FORWARD);
        JCufft.cufftExecC2C(PCE._plan1, PCE._d_y.getDevicePointer(), PCE._d_y.getDevicePointer(), JCufft.CUFFT_FORWARD);

        PCE._computeCrossCorr.launch(PCE._stream1, PCE.computeCrossCorr);

        PCE._findPeak.launch(PCE._stream1, PCE.findPeak);
        PCE._maxlocFloats.launch(PCE._stream1, PCE.maxlocFloats);

        PCE._stream1.synchronize();

        float peak[] = new float[1];
        int peakIndex[] = new int[1];
        PCE._d_peakValue.copyDeviceToHostAsync(peak, 1, PCE._stream1);
        PCE._d_peakIndex.copyDeviceToHostAsync(peakIndex, 1, PCE._stream1);

        PCE._stream1.synchronize();

        //transfer _d_c to PCE._c
        PCE._d_c.copyDeviceToHost(PCE._c, PCE._c.length);

        int peakIndexCPU = PCE.findPeak();
        double peakCPU = PCE._c[((PCE._rows * PCE._columns) - 1) << 1];

        double diff = Math.abs(peak[0] - peakCPU);
        System.err.println("peakGPU: " + peak[0] + " peakCPU: " + peakCPU);
        assertTrue(diff < 1e15); //should be the same value
        System.err.println("indexGPU: " + peakIndex[0] + " indexCPU: " + peakIndexCPU);
        assertTrue(peakIndex[0] == peakIndexCPU); //should match exactly

        assertNoCudaError();
    }

    @Test
    public void energyTest() {
        boolean testNaN = Util.compareArray(x, x, 10f);
        if (!testNaN) { System.err.println("energyTest input x contains NaN before start"); }
        assertTrue(testNaN);

        testNaN = Util.compareArray(y, y, 10f);
        if (!testNaN) { System.err.println("energyTest input y contains NaN before start"); }
        assertTrue(testNaN);

        PCE._d_inputx.copyHostToDeviceAsync(x, PCE._stream1);
        PCE._d_inputy.copyHostToDeviceAsync(y, PCE._stream1);

        JCufft.cufftExecC2C(PCE._plan1, PCE._d_x.getDevicePointer(), PCE._d_x.getDevicePointer(), JCufft.CUFFT_FORWARD);
        JCufft.cufftExecC2C(PCE._plan1, PCE._d_y.getDevicePointer(), PCE._d_y.getDevicePointer(), JCufft.CUFFT_FORWARD);
		JCudaDriver.cuCtxSynchronize();

        PCE._computeCrossCorr.launch(PCE._stream1, PCE.computeCrossCorr);

        PCE._findPeak.launch(PCE._stream1, PCE.findPeak);
        PCE._maxlocFloats.launch(PCE._stream1, PCE.maxlocFloats);

        PCE._computeEnergy.launch(PCE._stream1, PCE.computeEnergy);
        PCE._sumDoubles.launch(PCE._stream1, PCE.sumDoubles);

        int peakIndex[] = new int[1];
        double energyGPU[] = new double[1];
        PCE._d_peakIndex.copyDeviceToHostAsync(peakIndex, 1, PCE._stream1);
        PCE._d_energy.copyDeviceToHostAsync(energyGPU, 1, PCE._stream1);
        PCE._stream1.synchronize();

        //transfer _d_c to PCE._c
        PCE._d_c.copyDeviceToHost(PCE._c, PCE._c.length);
        int indexY = peakIndex[0] / PCE._columns;
        int indexX = peakIndex[0] - (indexY * PCE._columns);
        double energyCPU = PCE.energyFixed(PCE._squareSize, indexX, indexY);

        double err = Math.abs(energyGPU[0] - energyCPU);
        if (energyCPU != 0.0) {
            err /= energyCPU;
        } else if (energyGPU[0] != 0.0) {
            err /= energyGPU[0];
        } else {
            System.err.println("both energyCPU and energyGPU are 0.0");
            assertTrue(false);
        }
        System.err.println("energyGPU: " + energyGPU[0] + " energyCPU: " + energyCPU);
        assertTrue(err < 1e5);

        assertNoCudaError();
    }


    @AfterClass
    public static void tearDownAfterClass() throws Exception {
        PCE.cleanup();
    }



}
