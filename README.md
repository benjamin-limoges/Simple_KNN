Author: Benjamin Limoges <br>
Date: 10/15/2014


How to Run
------------

**python knn.py *filename.csv* neighbor_num partitions**

Where:

*filename.csv* is a well specified CSV file with the last 

neighbor_num is the maximum value of k to be run. neighbor_num should be an odd positive integer.

partition is the number of partitions - the algorithm will split the dataset into partitions.


Logic
-----------

Reads in the data.  Raises exception of neighbor_num is not an odd positive integer or if partitions is negative or 0.

CSV file should be in the same folder as knn.py - functionality to be build to allow it to be placed in a different folder.

CSV file must have labels at the end and not the beginning of each line.  If the file has them at the beginning, run "flip_csv.py" first.

Values should be continous and not discrete - my choice of distance metrics does not permit classifications as features.

This implements the kth nearest neighbor algorithm, using 3 different weighting schemes: uniform, inverse distance and log of distance.


1. Reads in command line arguments

2. Sets up KNN.  It does this by reading in the data, extracting the first line as a header, and randomly assigning each line a partition.  Returns a tuple containing the dataframe, the header row (minus the label's name), the label (or classification), and the randomly assigned partition numbers.

3. Builds KNN.  It does this without normalizing the data.  For each k from 1 to num_neighbors (inclusive and only the odd integers) it builds holding one partition out as test data.  It tests the accuracy using the three different distance metrics.  It then iterates through each of the held out partitions, the remainder being the training set.  The average for each of the runs is saved, and plotted.

4. After running KNN, the dataframe has its feature space converted to z-scores.  This is then returned to the main function.

5. Same as step 3 except implements with features in z-scores.