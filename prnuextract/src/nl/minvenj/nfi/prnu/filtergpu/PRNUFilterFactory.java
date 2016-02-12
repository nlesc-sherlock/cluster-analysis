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

import nl.minvenj.nfi.cuba.cudaapi.CudaContext;
import nl.minvenj.nfi.cuba.cudaapi.CudaDevice;
import nl.minvenj.nfi.cuba.cudaapi.CudaModule;

import org.apache.commons.io.IOUtils;

/**
 * PRNUFilterFactory should be instantiated once to compile the CUDA source files.
 * The PRNUFilterFactory is independent of image size and prevents that
 * the CUDA source files are compiled for each image size among files.
 * 
 * After creation the Factory can be used to construct multiple PRNUFilter
 * objects. These objects are created for a specific image size, because
 * they allocate GPU memory and create plans for CUFFT, etc. If there is not
 * enough GPU memory creation of the PRNUFilter obje
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */
public class PRNUFilterFactory {

	private static final String[] filenames = { "GrayscaleFilter.cu", "FastNoiseFilter.cu", "ZeroMeanTotalFilter.cu", "WienerFilter.cu"};
	private static String architecture = "compute_20";
	private static String capability = "sm_20";
	
	private CudaContext _context;
	private CudaModule[] modules;

    private boolean _fmad = true;	

	/**
	 * Instantiates the factory to compile the CUDA source files, do this
	 * only once per application run
	 */
	public PRNUFilterFactory() {
		this(true);	
	}
	
	/**
	 * Alternative constructor that can explicitly enable or disable fused
	 * multiply add, mainly intended for comparing CPU and GPU results. 
	 * 
	 * Instantiates the factory to compile the CUDA source files, do this
	 * only once per application run.
	 * 
	 * @param fmad - boolean TRUE is fused multiply add should be enabled, FALSE is fused multiply add should be disabled
	 */
	public PRNUFilterFactory(boolean fmad) {
		
		final CudaDevice device = CudaDevice.getBestDevice();
		System.out.println("Using GPU: " + device);
		_context = device.createContext();
        _fmad = fmad;

        int cc[] = device.getMajorMinor();
        this.architecture = "compute_" + cc[0] + "" + cc[1];
        this.capability = "sm_" + cc[0] + "" + cc[1];

		modules = new CudaModule[filenames.length];
		
		for (int i = 0; i < filenames.length; i++) {
            modules[i] = compile(filenames[i]);
		}
		
	}

    /**
     * Helper function that compiles a CUDA file and returns a CudaModule instance
     */
    public CudaModule compile(String filename) {
        //read the CUDA source file into a string
		String source = "";
		try {
			source = IOUtils.toString(FastNoiseFilter.class.getResource(filename));
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
			
		//compile the CUDA code to run on the GPU
        CudaModule module;
		if (_fmad) {
			module = _context.loadModule(source, "-gencode=arch=" + architecture + ",code=" +
					capability, "-Xptxas=-v");
		} else {
			module = _context.loadModule(source, "-gencode=arch=" + architecture + ",code=" +
					capability, "-Xptxas=-v", "--fmad=false");
		}
        return module;
    }


	/**
	 * This method creates a PRNUFilter for a specified image size,
	 * using the source files compiled by the Factory.
	 * If the application is to extract PRNU patterns from images
	 * of different sizes, create only one PRNUFilterFactory but use 
	 * it to create a PRNUFilter object for each image size.
	 * 
	 * @param height - image height in pixels
	 * @param width - image width in pixels
	 * @return a PRNUFilter object ready to be used on images
	 */
    public PRNUFilter createPRNUFilter(int height, int width) {
    	return new PRNUFilter(height, width, _context, modules);
    }

    /**
     * Return CUDA context
     */
    public CudaContext getContext() {
        return _context;
    }
    
    /**
     * cleanup GPU resources
     */
    public void cleanup() {
    	//unload modules
    	for (CudaModule module : modules) {
    		module.cleanup();
    	}
    	//destroy context
    	_context.destroy();
    }

}
