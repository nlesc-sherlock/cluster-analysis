/*
 * Copyright (c) 2011-2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.minvenj.nfi.prnu.filter;

public final class FastNoiseFilter implements ImageFilter {
    private static final float EPS = 1.0f;

    private final int _width;
    private final int _height;
    private final float[][] _dx;
    private final float[][] _dy;

    public FastNoiseFilter(final int width, final int height) {
        if (Math.min(width, height) < 2) {
            throw new IllegalArgumentException("FastNoise requires images of at least 2x2");
        }

        _width = width;
        _height = height;
        _dx = new float[height][width];
        _dy = new float[height][width];
    }

    @Override
    public void apply(final float[][] pixels) {
        for (int y = 0; y < _height; y++) {
            computeHorizontalGradient(pixels[y], _dx[y]);
        }
        for (int x = 0; x < _width; x++) {
            computeVerticalGradient(pixels, x);
        }
        for (int y = 0; y < _height; y++) {
            normalizeGradients(_dx[y], _dy[y]);
        }
        for (int y = 0; y < _height; y++) {
            storeHorizontalGradient(_dx[y], pixels[y]);
        }
        for (int x = 0; x < _width; x++) {
            addVerticalGradient(pixels, x);
        }
    }

    private void computeHorizontalGradient(final float[] src, final float[] dest) {
        // Take forward differences on first and last element
        dest[0] = (src[1] - src[0]);
        dest[_width - 1] = (src[_width - 1] - src[_width - 2]);

        // Take centered differences on interior points
        for (int i = 1; i < (_width - 1); i++) {
            dest[i] = 0.5f * (src[i + 1] - src[i - 1]);
        }
    }

    private void computeVerticalGradient(final float[][] pixels, final int x) {
        // Take forward differences on first and last element
        _dy[0][x] = (pixels[1][x] - pixels[0][x]);
        _dy[_height - 1][x] = (pixels[_height - 1][x] - pixels[_height - 2][x]);

        // Take centered differences on interior points
        for (int i = 1; i < (_height - 1); i++) {
            _dy[i][x] = 0.5f * (pixels[i + 1][x] - pixels[i - 1][x]);
        }
    }

    private void normalizeGradients(final float[] rowDx, final float[] rowDy) {
        for (int i = 0; i < _width; i++) {
            final float dx = rowDx[i];
            final float dy = rowDy[i];
            final float norm = (float) Math.sqrt((dx * dx) + (dy * dy));
            final float scale = 1.0f / (EPS + norm);
            rowDx[i] = (dx * scale);
            rowDy[i] = (dy * scale);
        }
    }
    
    private void storeHorizontalGradient(final float[] src, final float[] dest) {
        // Take forward differences on first and last element
        dest[0] = (src[1] - src[0]);
        dest[_width - 1] = (src[_width - 1] - src[_width - 2]);

        // Take centered differences on interior points
        for (int i = 1; i < (_width - 1); i++) {
            dest[i] = 0.5f * (src[i + 1] - src[i - 1]);
        }
    }

    private void addVerticalGradient(final float[][] dest, final int x) {
        // Take forward differences on first and last element
        dest[0][x] += (_dy[1][x] - _dy[0][x]);
        dest[_height - 1][x] += (_dy[_height - 1][x] - _dy[_height - 2][x]);

        // Take centered differences on interior points
        for (int i = 1; i < (_height - 1); i++) {
            dest[i][x] += 0.5f * (_dy[i + 1][x] - _dy[i - 1][x]);
        }
    }

}