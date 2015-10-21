/*
 * Copyright (c) 2014, Netherlands Forensic Institute
 * All rights reserved.
 */
package nl.minvenj.nfi.prnu;

import java.awt.color.CMMException;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;

import javax.imageio.ImageIO;

import nl.minvenj.nfi.prnu.filter.FastNoiseFilter;
import nl.minvenj.nfi.prnu.filter.ImageFilter;
import nl.minvenj.nfi.prnu.filter.WienerFilter;
import nl.minvenj.nfi.prnu.filter.ZeroMeanTotalFilter;

public final class PrnuExtract {
    static final File TESTDATA_FOLDER = new File("testdata");

    static final File INPUT_FOLDER = new File(TESTDATA_FOLDER, "input");
    public static final File INPUT_FILE = new File(INPUT_FOLDER, "test.jpg");
    static final File EXPECTED_PATTERN_FILE = new File(INPUT_FOLDER, "expected.pat");

    static final File OUTPUT_FOLDER = new File(TESTDATA_FOLDER, "output");
    static final File OUTPUT_FILE = new File(OUTPUT_FOLDER, "test.pat");

    public static void main(final String[] args) throws IOException {
        long start = System.currentTimeMillis();
    	long end = 0;
    	
    	// Laad de input file in
        final BufferedImage image = readImage(INPUT_FILE);
        end = System.currentTimeMillis();
        System.out.println("Load image: " + (end-start) + " ms.");

        // Zet de input file om in 3 matrices (rood, groen, blauw)
        start = System.currentTimeMillis();
        final float[][][] rgbArrays = convertImageToFloatArrays(image);
        end = System.currentTimeMillis();
        System.out.println("Convert image:" + (end-start) + " ms.");

        // Bereken van elke matrix het PRNU patroon (extractie stap)
        start = System.currentTimeMillis();
        for (int i = 0; i < 3; i++) {
            extractImage(rgbArrays[i]);
        }
        end = System.currentTimeMillis();
        System.out.println("PRNU extracted: " + (end-start) + " ms.");

        // Schrijf het patroon weg als een Java object
        writeJavaObject(rgbArrays, OUTPUT_FILE);

        System.out.println("Pattern written");

        // Controleer nu het gemaakte bestand
        final float[][][] expectedPattern = (float[][][]) readJavaObject(EXPECTED_PATTERN_FILE);
        final float[][][] actualPattern = (float[][][]) readJavaObject(OUTPUT_FILE);
        for (int i = 0; i < 3; i++) {
            // Het patroon zoals dat uit PRNU Compare komt, bevat een extra matrix voor transparantie. Deze moeten we overslaan (+1)!
            compare2DArray(expectedPattern[i + 1], actualPattern[i], 0.0001f);
        }

        System.out.println("Validation completed");
        
        //This exit is inserted because the program will otherwise hang for a about a minute
        //most likely explanation for this is the fact that the FFT library spawns a couple
        //of threads which cannot be properly destroyed
        System.exit(0);
    }

    private static BufferedImage readImage(final File file) throws IOException {
        final InputStream fileInputStream = new FileInputStream(file);
        try {
            final BufferedImage image = ImageIO.read(new BufferedInputStream(fileInputStream));
            if ((image != null) && (image.getWidth() >= 0) && (image.getHeight() >= 0)) {
                return image;
            }
        }
        catch (final CMMException e) {
            // Image file is unsupported or corrupt
        }
        catch (final RuntimeException e) {
            // Internal error processing image file
        }
        catch (final IOException e) {
            // Error reading image from disk
        }
        finally {
            fileInputStream.close();
        }

        // Image unreadable or too smalld array
        return null;
    }

    private static float[][][] convertImageToFloatArrays(final BufferedImage image) {
        final int width = image.getWidth();
        final int height = image.getHeight();
        final float[][][] pixels = new float[3][height][width];

        final ColorModel colorModel = ColorModel.getRGBdefault();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                final int pixel = image.getRGB(x, y); // aa bb gg rr
                pixels[0][y][x] = colorModel.getRed(pixel);
                pixels[1][y][x] = colorModel.getGreen(pixel);
                pixels[2][y][x] = colorModel.getBlue(pixel);
            }
        }
        return pixels;
    }

    private static void extractImage(final float[][] pixels) {
        final int width = pixels[0].length;
        final int height = pixels.length;

        long start = System.currentTimeMillis();
        long end = 0;

        final ImageFilter fastNoiseFilter = new FastNoiseFilter(width, height);
        fastNoiseFilter.apply(pixels);

        end = System.currentTimeMillis();
        System.out.println("Fast Noise Filter: " + (end-start) + " ms.");

        start = System.currentTimeMillis();
        final ImageFilter zeroMeanTotalFilter = new ZeroMeanTotalFilter(width, height);
        zeroMeanTotalFilter.apply(pixels);

        end = System.currentTimeMillis();
        System.out.println("Zero Mean Filter: " + (end-start) + " ms.");

        start = System.currentTimeMillis();
        final ImageFilter wienerFilter = new WienerFilter(width, height);
        wienerFilter.apply(pixels);

        end = System.currentTimeMillis();
        System.out.println("Wiener Filter: " + (end-start) + " ms.");
    }

    public static Object readJavaObject(final File inputFile) throws IOException {
        final ObjectInputStream inputStream = new ObjectInputStream(new BufferedInputStream(new FileInputStream(inputFile)));
        try {
            return inputStream.readObject();
        }
        catch (final ClassNotFoundException e) {
            throw new IOException("Cannot read pattern: " + inputFile.getAbsolutePath(), e);
        }
        finally {
            inputStream.close();
        }
    }

    private static void writeJavaObject(final Object object, final File outputFile) throws IOException {
        final OutputStream outputStream = new FileOutputStream(outputFile);
        try {
            final ObjectOutputStream objectOutputStream = new ObjectOutputStream(new BufferedOutputStream(outputStream));
            objectOutputStream.writeObject(object);
            objectOutputStream.close();
        }
        finally {
            outputStream.close();
        }
    }

    private static boolean compare2DArray(final float[][] expected, final float[][] actual, final float delta) {
        for (int i = 0; i < expected.length; i++) {
            for (int j = 0; j < expected[i].length; j++) {
                if (Math.abs(actual[i][j] - expected[i][j]) > delta) {
                    System.err.println("de waarde op " + i + "," + j + " is " + actual[i][j] + " maar had moeten zijn " + expected[i][j]);
                    return false;
                }
            }
        }
        return true;
    }

}
