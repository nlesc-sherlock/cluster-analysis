/*
 * Copyright (c) 2012-2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.minvenj.nfi.prnu;

import edu.emory.mathcs.jtransforms.fft.FloatFFT_2D;

public final class PeakToCorrelationEnergy {
    private final int _rows;
    private final int _columns;
    private final int _squareSize;
    private final float[] _rowBuffer1;
    private final float[] _rowBuffer2;
    private float[] _c;
    private float[] _x;
    private float[] _y;

    public final FloatFFT_2D _fft;

   /**
     * Creates a new PCE correlation strategy.
     *
     * @param rows the number of rows (height in pixels) of the patterns to compare
     * @param columns the number of columns (width in pixels) of the patterns to compare
     * @param squareSize the size of the area around the peak (A)
     */
    public PeakToCorrelationEnergy(final int rows, final int columns, final int squareSize) {
        _rows = rows;
        _columns = columns;
        _squareSize = squareSize;
        
        /*
        if (_squareSize == 0) {
            throw new IllegalArgumentException("squareSize cannot be 0");
        }
        if ((_squareSize > Math.min(_rows, _columns)) || ((_squareSize == _rows) && (_squareSize == _columns))) {
            throw new IllegalArgumentException(String.format("squareSize is invalid for float[] size (%d)", squareSize));
        }
        */

        _fft = new FloatFFT_2D(rows, columns);
        _rowBuffer1 = new float[columns];
        _rowBuffer2 = new float[columns];
        _c = new float[_rows * _columns * 2];
        _x = new float[_rows * _columns * 2];
        _y = new float[_rows * _columns * 2];
    }

    /*
     * This method does the same as compare but assumes the input in FFT space
     */
    public double compare_fft(final float[] fx, final float[] fy) {
        crosscorr_fft(fx, fy);
	int peakIndex = findPeak();
	double peak = _c[peakIndex];
	int indexY = peakIndex / _columns;
	int indexX = peakIndex - indexY;
        double absPce = (peak * peak) / energyFixed(_squareSize, indexX, indexY);
        return (peak < 0.0) ? -absPce : absPce;
    }
 
    public double compare(final float[] x, final float[] y) {
        //argPattern("x", x);
        //argPattern("y", y);

	long start, end;
        start = System.nanoTime();
        crosscorr(x, y); // result is stored in '_c' !!
	end = System.nanoTime();
        //System.out.println("crosscor took: " + (end-start)/1e6 + " ms.");

	/*
        double peak = _x[((_rows * _columns) - 1) * 2]; //how do we know the peak is here?

        start = System.nanoTime();
        double absPce = (peak * peak) / energy(_squareSize);
	end = System.nanoTime();
        //System.out.println("energy took: " + (end-start)/1e6 + " ms. result= " + absPce);
	*/

        start = System.nanoTime();
	int peakIndex = findPeak();
	end = System.nanoTime();
        //System.out.println("findPeak took: " + (end-start)/1e6 + " ms.");

	double peak = _c[peakIndex];
	int indexY = peakIndex / _columns;
	int indexX = peakIndex - indexY;

        start = System.nanoTime();
        double absPce = (peak * peak) / energyFixed(_squareSize, indexX, indexY);
	end = System.nanoTime();
        //System.out.println("energyFixed took: " + (end-start)/1e6 + " ms. result= " + absPce);

        return (peak < 0.0) ? -absPce : absPce;
    }

    public int findPeak() { 
	float max = 0.0f;
	int res = 0;
	for (int i=0; i<_c.length; i+=2) { //only look at real values in complex array x
	  if (_c[i] > max) {
	    max = _c[i];
	    res = i;
	  }
	}
	return res;
    }

    private double energy(final int squareSize) {
        final int radius = (squareSize - 1) / 2;
        final int n = (_rows * _columns) - (squareSize * squareSize);

        // Determine the energy, i.e. the sample variance of circular cross-correlations, minus an area around the peak
        double energy = 0.0f;
        for (int row = 0; row <= radius; row++) {
            final int offset = row * _columns*2;
            for (int col = (radius + 1); col < (_columns - radius); col++) {
                final float f = _c[offset + (col + col)];
                energy += (f * f);
            }
        }
        for (int row = (radius + 1); row < (_rows - radius); row++) {
            final int offset = row * _columns*2;
            for (int col = 0; col < _columns; col++) {
                final float f = _c[offset + (col + col)];
                energy += (f * f);
            }
        }
        for (int row = (_rows - radius); row < _rows; row++) {
            final int offset = row * _columns*2;
            for (int col = (radius + 1); col < (_columns - radius); col++) {
                final float f = _c[offset + (col + col)];
                energy += (f * f);
            }
        }
        return (energy / n);
    }


    private double energyFixed(final int squareSize, int peakIndexX, int peakIndexY) {
        final int radius = (squareSize - 1) / 2;
        final int n = (_rows * _columns) - (squareSize * squareSize);

        // Determine the energy, i.e. the sample variance of circular cross-correlations, minus an area around the peak
        double energy = 0.0f;
        for (int row = 0; row < _rows; row++) {
	    boolean peakRow = row > peakIndexY - radius && row < peakIndexY + radius;
            for (int col = 0; col < _columns; col++) {
		if (peakRow && col > peakIndexX - radius && col < peakIndexX + radius) {
		  continue;
		}
		else {
		  float f = _c[row*_columns*2 + col*2];
                  energy += (f * f);
		}
            }
        }
        return (energy / n);
    }



    /*
    private void argPattern(final String name, final float[] arg) {
        argNotNull(name, arg);

        if ((arg.getRowCount() != _rows) || (arg.getColumnCount() != _columns)) {
            throw new IllegalArgumentException(String.format("%s has invalid size (%dx%d)", name, arg.getRowCount(), arg.getColumnCount()));
        }
    }
    */


    /*
     * This is the same crosscorr method as below, but uses input already in frequency space
     */
    public void crosscorr_fft(float[] fx, float[] fy) {
	this._x = fx;
	this._y = fy;

	compute_crosscorr();

        _fft.complexInverse(_c, true);
    }



    private void crosscorr(final float[] x, final float[] y) {
        for (int row = 0; row < _rows; row++) {
            //x.selectRow(row, _rowBuffer1);
            //y.selectRow(row, _rowBuffer2);
            for (int i=0; i<_columns; i++) {
            	_rowBuffer1[i] = x[row * _columns + i];
            	_rowBuffer2[i] = y[row * _columns + i];
            }            

            final int xOffset = (row * _columns) * 2;
            final int yOffset = (((_rows - row) * _columns) - 1) * 2;

            //this is a toComplex operation that copies the contents of rowbuffer 1 and 2 into _x and _y
            for (int col = 0; col < _columns; col++) {
                _x[xOffset + (col + col)] = _rowBuffer1[col];
                _x[xOffset + (col + col) + 1] = 0.0f;
                _y[yOffset - (col + col)] = _rowBuffer2[col];
                _y[xOffset + (col + col) + 1] = 0.0f;
            }
        }

        //forward transform of both inputs
        _fft.complexForward(_x);
        _fft.complexForward(_y);

	compute_crosscorr();

        _fft.complexInverse(_c, true);
    }



    void compute_crosscorr() {

        for (int i = 0; i < _x.length; i += 2) {
            final float xRe = _x[i];
            final float xIm = _x[i + 1];
            final float yRe = _y[i];
            final float yIm = _y[i + 1];
            _c[i] = (xRe * yRe) - (xIm * yIm);
            _c[i + 1] = (xRe * yIm) + (xIm * yRe);
        }

    }



}
