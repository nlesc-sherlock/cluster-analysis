/*
 * Copyright (c) 2012-2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.minvenj.nfi.prnu;


public final class NormalizedCrossCorrelation {

    public static double sumSquared(final float[] pattern) {
	double sumsq = 0.0;
	for (int i=0; i<pattern.length; i++) {
	  sumsq += pattern[i] * pattern[i];
	}
	return sumsq;
    }

    public static double compare(final float[] x, double sumsq_x, final float[] y, double sumsq_y) {
	double sum_xy = 0.0;

        for (int i=0; i<x.length; i++) {
          sum_xy += x[i] * y[i];
       	}

	return (sum_xy / Math.sqrt(sumsq_x * sumsq_y));
    }

}
