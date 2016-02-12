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


public class PrnuPatternCache {

    int numPatterns = 0;

    long counter = 0;
    HashMap<String, cacheItem> cache;

    PRNUFilter filter;
    String path;

    public PrnuPatternCache(int h, int w, PRNUFilter filter, String path) {
        this.filter = filter;
        this.path = path;

        int patternSizeBytes = h * w * 4;
        System.out.println("pattern size = " + patternSizeBytes + " in MB: " + (patternSizeBytes/1024/1024));

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

        cache = new HashMap<String, cacheItem>(numPatterns, 1.1f);
    }

    /*
     * Retrieve the pattern belonging to filename from the cache
     * If the pattern is not yet in the cache, the pattern will be retrieved and stored in the cache
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

    /*
     * Populate the cache with up to numPatterns patterns from the filenames
     */
    void populate(String[] filenames) {
        populate(filenames, numPatterns);
    }
    /*
     * Populate the cache with up to n patterns
     * This method stops when the filenames to include run out or when
     * the cache reaches its limit, whichever comes first.
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

    /*
     * Evict the least recently used pattern from the cache
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



    /*
     * Simple class to hold the float array containing the pattern and the last use counter
     */
    private class cacheItem {

        public long used;
        public float[] pattern;

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

        float[] getPattern() {
            used = counter++;
            return pattern;
        }

    }


}
