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

import java.io.File;
import java.util.Map;
import java.util.HashMap;
import java.awt.image.BufferedImage;

import nl.minvenj.nfi.prnu.filtergpu.*;
import nl.minvenj.nfi.prnu.Util;



/**
 * This class implements a simple cache for PRNU patterns
 * It currently uses a least recently used scheme for eviction
 *
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 */
public class PrnuPatternCache {

    //number of patterns stored in the cache
    int numPatterns = 0;

    //counter that is increased with each retrieval of a pattern
    long counter = 0;

    //map that stores the cacheitems based on their associated image filename
    HashMap<String, cacheItem> cache;

    //a reference to the PRNUfilter class for retrieving patterns currently not in the cache
    PRNUFilter filter;

    //path to the folder containing the images
    String path;

    /**
     * Constructor of the PRNU pattern cache
     * 
     * This method also determines the number of patterns that can be stored in the cache based
     * on the amount of memory currently available to the JVM. It will attempt to eat up all the
     * available memory except for the last 5GB, to leave some memory to work with to the rest
     * of the application.
     *
     * @param h         height of the images
     * @param w         width of the images
     * @param filter    reference to PRNUFilter object that is used to extract PRNU patterns from images
     * @param path      String containing the path to the folder containing the images
     */
    public PrnuPatternCache(int h, int w, PRNUFilter filter, String path) {
        this.filter = filter;
        this.path = path;

        int patternSizeBytes = h * w * 4;
        System.out.println("pattern size = " + patternSizeBytes + " in MB: " + (patternSizeBytes/1024/1024));

        //obtain info about the amount of memory available
        Runtime runtime = Runtime.getRuntime();
        long totalSpace = runtime.totalMemory();
        long maxSpace = runtime.maxMemory();
        long freeSpace = runtime.freeMemory();
        System.out.println("max mem=" + maxSpace + " total mem=" + totalSpace + " free mem=" + freeSpace);

        long inuse = totalSpace - freeSpace;
        long actualFreeSpace = maxSpace - inuse;

        System.out.println("free space = " + actualFreeSpace + " in MB: " + (actualFreeSpace/1024/1024));

        //it is probably smart to not use everything, but leave say 5 gigabyte free for other stuff
        double free = (double)actualFreeSpace;
        double patternsize = (double)patternSizeBytes;
        int fitPatterns = (int)Math.floor( (actualFreeSpace - 5*1e9) / patternsize );

        System.out.println("Number of patterns that the cache will hold: " + fitPatterns);
        if (fitPatterns < 10) {
            System.err.println("Number of patterns that cache will hold is too small: " + fitPatterns);
            System.exit(1);
        }

        this.numPatterns = fitPatterns;

        //create a map for storing the cacheItem objects
        //initial capacity is the number of patterns in the cache
        //the load factor is set so that the HashMap won't grow automatically
        cache = new HashMap<String, cacheItem>(numPatterns, 1.1f);
    }

    /**
     * Retrieve the pattern belonging to the image filename from the cache
     * If the pattern is not yet in the cache, the pattern will be recomputed and stored in the cache
     * 
     * @param filename  a String containing the name of the image to which the pattern we want to retrieve belongs
     */
    float[] retrieve(String filename) {
        cacheItem item = cache.get(filename);
        //if item not in cache yet
        if (item == null) {

            //make sure there is room to store at least one new pattern
            while (cache.size() >= numPatterns) {
                evict();
            }

            //create the item and store in cache
            item = new cacheItem(filename);
            cache.put(filename, item);
        }
            
        //return the requested pattern
        return item.getPattern();
    }

    /**
     * Populate the cache with up to numPatterns patterns from the filenames
     *
     * @param filenames     a String array containing the filenames of the images whose PRNU patterns should be in the cache
     */
    void populate(String[] filenames) {
        populate(filenames, numPatterns);
    }

    /**
     * Populate the cache with up to n patterns
     *
     * This method stops when n patterns are include, the filenames to include run out, or when
     * the cache reaches its limit, whichever comes first.
     *
     * @param filenames     a String array containing the filenames of the images whose PRNU patterns should be in the cache
     * @param n             the number of PRNU patterns to populate the cache with      
     */
    void populate(String[] filenames, int n) {
        for (int i = 0; i < n && i < filenames.length && cache.size() < numPatterns; i++) {
            if (cache.containsKey(filenames[i])) {
                //do nothing
            } else {
                cache.put(filenames[i], new cacheItem(filenames[i]));
            }

            //This System.gc() is necessary unfortunately
            if (i % 50 == 0) {
                System.gc();
            }
        }
    }

    /**
     * Evict the least recently used pattern from the cache using a least recently used policy
     */
    void evict() {
        long lru = Long.MAX_VALUE;
        String evict = "";

        for (Map.Entry<String, cacheItem> entry : cache.entrySet()) {
            String key = entry.getKey();
            cacheItem value = entry.getValue();
            if (value.used < lru) {
                lru = value.used;
                evict = key;
            }
        }

        cache.remove(evict);
    }



    /**
     * Simple class to hold the float array containing the pattern and the last use counter
     */
    private class cacheItem {

        public long used;
        public float[] pattern;

        /**
         * Constructs a cacheItem based on a filename of an image
         * The PRNU pattern of the image is extracted and stored in this cacheItem
         *
         * @param filename      a String containing the filename of the image
         */
        cacheItem(String filename) {

            File f = new File(path + "/" + filename);

            BufferedImage image = null;
            try {
                image = Util.readImage(f);
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(1);
            }

            pattern = filter.apply(image);
        }

        /**
         * Getter for the PRNU pattern
         * also increases the least recently used counter
         *
         * @returns     a float array containing the PRNU pattern
         */
        float[] getPattern() {
            used = counter++;
            return pattern;
        }

    }


}
