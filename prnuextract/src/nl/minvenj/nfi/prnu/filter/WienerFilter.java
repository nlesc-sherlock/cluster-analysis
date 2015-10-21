/*
 * Copyright (c) 2012-2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.minvenj.nfi.prnu.filter;

import java.util.Arrays;

import nl.minvenj.nfi.prnu.Util;
import edu.emory.mathcs.jtransforms.fft.FloatFFT_1D;
import edu.emory.mathcs.jtransforms.fft.FloatFFT_2D;

public final class WienerFilter implements ImageFilter {
    private static final int[] FILTER_SIZES = {3, 5, 7, 9};
    /** The amount of padding (border size) of the source data. */
    private static final int BORDER_SIZE = max(FILTER_SIZES) >> 1;

    private final int _width;
    private final int _height;
    private final int _paddedWidth;
    private final FloatFFT_1D _fftColumnTransform;
    private final FloatFFT_1D _fftRowTransform;
    private final float[][] _fft;
    private final float[] _fftColumnBuffer;
    private final float[][] _squaredMagnitudes;
    private final float[] _sumSquareBuffer;

    public WienerFilter(final int width, final int height) {
        if (Math.min(width, height) < BORDER_SIZE) {
            throw new IllegalArgumentException(String.format("Wiener filter requires images of at least %dx%d", BORDER_SIZE, BORDER_SIZE));
        }

        _height = height;
        _width = width;
        _paddedWidth = (_width + (2 * BORDER_SIZE));

        _fftRowTransform = new FloatFFT_1D(width);
        _fftColumnTransform = new FloatFFT_1D(height);
        _fft = new float[_height][_width * 2];
        _fftColumnBuffer = new float[2 * _height];
        _squaredMagnitudes = new float[_height + (2 * BORDER_SIZE) + 1][_paddedWidth];
        _sumSquareBuffer = new float[_paddedWidth + 1];
    }
    
    @Override
    public void apply(final float[][] pixels) {
        final float[][] varianceEstimates = pixels; // Reuse array to minimize memory usage

        double sumSquares = 0.0;
        
        float[][] pixelsTransposed = Util.transpose(pixels);

        
        float[][] pixelsComplexTransposed = new float[_width][_height * 2];
        for (int i = 0; i < _width; i++) {
        	for (int j = 0; j < _height; j++) {
        		pixelsComplexTransposed[i][2*j] = pixelsTransposed[i][j];
        		pixelsComplexTransposed[i][2*j+1] = 0.0f;
        	}
        }
        
        FloatFFT_2D bla = new FloatFFT_2D(_width, _height);
        bla.complexForward(pixelsComplexTransposed);
        
        /*
        float[][] pixelsComplex = new float[_height][_width * 2];
        float[][] pixelsComplex2 = new float[_height][_width * 2];
        
        for (int i = 0; i < _height; i++) {
        	for (int j = 0; j < _width; j++) {
        		pixelsComplex[i][2*j] = pixels[i][j];
        		pixelsComplex[i][2*j+1] = 0.0f;
        		
        		pixelsComplex2[i][2*j] = pixels[i][j];
        		pixelsComplex2[i][2*j + 1] = 0.0f;
        	}
        }
        
        FloatFFT_2D bla = new FloatFFT_2D(_height, _width);
        bla.complexForward(pixelsComplex);
        
        FloatFFT_1D row = new FloatFFT_1D(_width);
        FloatFFT_1D col = new FloatFFT_1D(_height);
        
        for (int i = 0; i < _height; i++) {
        	row.complexForward(pixelsComplex2[i]);
        }
        
        for (int i = 0; i < _width; i++) {
        	float[] temp = new float[_height * 2];
        	
        	for (int j = 0; j < _height; j++) {
        		temp[2*j] = pixelsComplex2[j][i*2];
        		temp[2*j + 1] = pixelsComplex2[j][i * 2 + 1];
        	}
        	col.complexForward(temp);
        	
        	for (int j = 0; j < _height; j++) {
        		pixelsComplex2[j][i*2] = temp[2*j];
        		pixelsComplex2[j][i*2+1] = temp[2*j+1];
        	}
        }
        
        	
        System.out.println("2d 1d compare");
        Util.compare2DArray(pixelsComplex2, pixelsComplex, 0.000001f);
        
        */
        

        // Compute variance of input, perform FFT and initialize variance estimates
        // Note #1: We assume that the mean of the input data is zero (after 'zero mean').
        // Note #2: Both variance and variance estimates are scaled by 'n' to avoid divisions.
        for (int x = 0; x < _width; x++) {
            sumSquares += realColumnToComplex(pixels, x);

            _fftColumnTransform.complexForward(_fftColumnBuffer);

            storeComplexColumn(_fftColumnBuffer, x);
        }
        for (int y = 0; y < _height; y++) {
            _fftRowTransform.complexForward(_fft[y]);
        }
        
        float[][] pixelsComplex = Util.transposeComplex(pixelsComplexTransposed);
        
        //System.out.printf("comparing");
        Util.compare2DArray(_fft, pixelsComplex, 0.000001f);
        //System.out.printf("end comparing");
        
        for (int y = 0; y < _height; y++) {
            Arrays.fill(varianceEstimates[y], Float.MAX_VALUE);

            computeComplexMagnitudes(_fft[y], _squaredMagnitudes[y + BORDER_SIZE]);
        }

        // Estimate the minimum variance for each filter at each position
        for (final int filterSize : FILTER_SIZES) {
            updateVarianceEstimates(varianceEstimates, filterSize);
        }

        // 'Clean' the input using the minimum variance estimates and perform IFFT
        final int n = _width * _height;
        final float variance = (float) ((sumSquares * n) / (n - 1));
        for (int x = 0; x < _width; x++) {
            cleanColumn(varianceEstimates, variance, x, _fftColumnBuffer);

            _fftColumnTransform.complexInverse(_fftColumnBuffer, true);

            storeComplexColumn(_fftColumnBuffer, x);
        }
        for (int y = 0; y < _height; y++) {
            _fftRowTransform.complexInverse(_fft[y], true);

            complexToReal(_fft[y], pixels[y], _width);
        }
    }

    private double realColumnToComplex(final float[][] pixels, final int x) {
        double sumSquares = 0.0;
        for (int y = 0; y < _height; y++) {
            final int idx2 = 2 * y;
            final float f = pixels[y][x];
            _fftColumnBuffer[idx2] = f; // re
            _fftColumnBuffer[idx2 + 1] = 0.0f; // im
            sumSquares += (f * f);
        }
        return sumSquares;
    }

    private void computeComplexMagnitudes(final float[] src, final float[] dest) {
        for (int x = 0; x < _width; x++) {
            final float re = src[x + x];
            final float im = src[x + x + 1];
            dest[BORDER_SIZE + x] = (re * re) + (im * im);
        }
    }

    private void cleanColumn(final float[][] varianceEstimates, final float variance, final int x, final float[] dest) {
        final int idx1 = 2 * x;
        for (int y = 0; y < _height; y++) {
            // Note: 'magScale' are the elements of 'Fmag1./Fmag' in the Matlab source!
            final float magScale = variance / Math.max(variance, varianceEstimates[y][x]);
            final int idx2 = 2 * y;
            dest[idx2] = _fft[y][idx1] * magScale;
            dest[idx2 + 1] = _fft[y][idx1 + 1] * magScale;
        }
    }

    private void storeComplexColumn(final float[] src, final int x) {
        final int idx1 = 2 * x;
        for (int y = 0; y < _height; y++) {
            final int idx2 = 2 * y;
            _fft[y][idx1] = src[idx2];
            _fft[y][idx1 + 1] = src[idx2 + 1];
        }
    }

    private static void complexToReal(final float[] complexSrc, final float[] realDest, final int size) {
        for (int i = 0; i < size; i++) {
            realDest[i] = complexSrc[i + i];
        }
    }

    private static int max(final int[] values) {
        int maxValue = values[0];
        for (int i = 1; i < values.length; i++) {
            maxValue = Math.max(maxValue, values[i]);
        }
        return maxValue;
    }

    /**
     * Updates the variance estimates.
     * <p>
     * This computes the variance of each sample and the neighboring pixels
     * that surround it and, if the result is lower than the variance estimate
     * for that pixel, updates the variance estimate.
     *
     * @param varianceEstimates the variance estimates to update 
     * @param filterSize the size of the square-shaped neighborhood window
     */
    private void updateVarianceEstimates(final float[][] varianceEstimates, final int filterSize) {
        final int borderOffset = BORDER_SIZE - ((filterSize - 1) / 2);
        final int paddedColumns = _paddedWidth - (2 * borderOffset);
        final float fScale = 1.0f / (filterSize * filterSize);

        Arrays.fill(_sumSquareBuffer, 0.0f);

        // Compute sum of squares for each column for row (0)
        for (int y = 0; y < filterSize; y++) {
            final float[] rowMag = _squaredMagnitudes[y + borderOffset];
            for (int x = 0; x < paddedColumns; x++) {
                _sumSquareBuffer[x + 1] += rowMag[borderOffset + x];
            }
        }
        for (int x = 1; x <= paddedColumns; x++) {
            _sumSquareBuffer[x] *= fScale;
        }

        // Process rows
        for (int y = 0; y < _height; y++) {
            // Update minimum variance at each column
            float sumSquare = 0.0f;
            for (int x = 1; x < filterSize; x++) {
                sumSquare += _sumSquareBuffer[x];
            }
            final float[] rowEstimates = varianceEstimates[y];
            for (int x = 0; x < _width; x++) {
                sumSquare += (_sumSquareBuffer[x + filterSize] - _sumSquareBuffer[x]);

                // Note: If we assume that the mean of the samples is '0', we can use the sum of squares
                //       as estimate for the variance.
                rowEstimates[x] = Math.min(rowEstimates[x], sumSquare);
            }

            // Update sum of squares at each column for next row
            final float[] topSqMag = _squaredMagnitudes[y + borderOffset];
            final float[] bottomSqMag = _squaredMagnitudes[y + borderOffset + filterSize];
            for (int x = 0; x < paddedColumns; x++) {
                _sumSquareBuffer[x + 1] += fScale * (bottomSqMag[borderOffset + x] - topSqMag[borderOffset + x]);
            }
        }
    }
}
