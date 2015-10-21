/*
 * Copyright (c) 2011-2013, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.minvenj.nfi.prnu.filter;

public final class ZeroMeanTotalFilter implements ImageFilter {
    private final int _width;
    private final int _height;

    public ZeroMeanTotalFilter(final int width, final int height) {
        if (Math.min(width, height) < 2) {
            throw new IllegalArgumentException("ZeroMeanTotal requires images of at least 2x2");
        }

        _width = width;
        _height = height;
    }

    @Override
    public void apply(final float[][] pixels) {
        for (int y = 0; y < _height; y++) {
            filterRow(pixels[y]);
        }
        for (int x = 0; x < _width; x++) {
            filterColumn(pixels, x);
        }
    }

    private void filterRow(final float[] rowData) {
        float sumEven = 0.0f;
        float sumOdd = 0.0f;
        for (int i = 0; i < (_width - 1); i += 2) {
            sumEven += rowData[i];
            sumOdd += rowData[i + 1];
        }
        if (!isDivisibleByTwo(_width)) {
            sumEven += rowData[_width - 1];
        }

        final float meanEven = sumEven / ((_width + 1) >> 1);
        final float meanOdd = sumOdd / (_width >> 1);
        for (int i = 0; i < (_width - 1); i += 2) {
            rowData[i] -= meanEven;
            rowData[i + 1] -= meanOdd;
        }
        if (!isDivisibleByTwo(_width)) {
            rowData[_width - 1] -= meanEven;
        }
    }

    private void filterColumn(final float[][] pixels, final int x) {
        float sumEven = 0.0f;
        float sumOdd = 0.0f;
        for (int i = 0; i < (_height - 1); i += 2) {
            sumEven += pixels[i][x];
            sumOdd += pixels[i + 1][x];
        }
        if (!isDivisibleByTwo(_height)) {
            sumEven += pixels[_height - 1][x];
        }

        final float meanEven = sumEven / ((_height + 1) >> 1);
        final float meanOdd = sumOdd / (_height >> 1);
        for (int i = 0; i < (_height - 1); i += 2) {
            pixels[i][x] -= meanEven;
            pixels[i + 1][x] -= meanOdd;
        }
        if (!isDivisibleByTwo(_height)) {
            pixels[_height - 1][x] -= meanEven;
        }
    }

    private static boolean isDivisibleByTwo(final int value) {
        return (value & 1) == 0;
    }
}
