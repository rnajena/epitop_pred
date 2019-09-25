"""
Positive Selection Detection Using Amino Acid Variance Analysis

This script aims to detect positive selection sites on a given
Amino Acid MSA by computing a Proper Orthogonal Decomposition
of each column of the MSA. Here the amino acids are interpreted
as a pair of two numbers [polarity, molecular volume] as suggested
by R. Grantham in 1974 and used by K. Katouh et al. in 2002.
The biological assumption behind this approach is, that amino acids on
homolog positive selection sides differ more from their aligned partners, than
amino acids on a non-positive selection side.
The variances along the two primary components of each MSA column
are then used to compute an index, that measures the diversity of the
amino acids within the respective column.
Also the scripts performs a simple significance test, assuming a normal
distribution of the non-positive selection sites with the variance and the
mean of the computed index over the whole MSA.
"""

# To run this program you will need the following packages.
import sys
from Bio import SeqIO
import numpy as np
from scipy.stats import norm
import os

"""
This method transforms the amino acid sequences within a MSA into sequences of
two dimensional vectors ( [polarity, molecular volume] ). The components of these vectors
are saved in two seperate files (pol.csv and mVol.csv).
"""


def transformAAtoVectors(inputFile, outputDirectory):
    with open(os.path.join(outputDirectory,"pol.csv"), "w") as outputP, open(os.path.join(outputDirectory, "mVol.csv"), "w") as outputV:
        # This step determines the length of the longest Sequence within the input file.
        # If the input file always is a MSA (in which all Sequences have the same length) then this step is unnecessary.
        length = sys.maxsize
        aaAln = SeqIO.parse(open(inputFile, "r"), "fasta")
        for species in aaAln:
            if length > len(str(species.seq)):
                length = len(str(species.seq))
                # Now the one character amino acid code is translated into pairs of polarity and molecular volume.
                # This for loop iterates over each column of the input MSA.
                # The Amino Acid properties determined by Grantham (1974)
                p = {"R": 10.5,
                     "L": 4.9,
                     "P": 8,
                     "T": 8.6,
                     "S": 9.2,
                     "A": 8.1,
                     "V": 5.9,
                     "G": 9,
                     "I": 5.2,
                     "F": 5.2,
                     "Y": 6.2,
                     "C": 5.5,
                     "H": 10.4,
                     "Q": 10.5,
                     "N": 11.6,
                     "K": 11.3,
                     "D": 13,
                     "E": 12.3,
                     "M": 5.7,
                     "W": 5.4}

                v = {"R": 124,
                     "L": 111,
                     "P": 32.5,
                     "T": 61,
                     "S": 32,
                     "A": 31,
                     "V": 84,
                     "G": 3,
                     "I": 111,
                     "F": 132,
                     "Y": 136,
                     "C": 55,
                     "H": 96,
                     "Q": 85,
                     "N": 56,
                     "K": 119,
                     "D": 54,
                     "E": 83,
                     "M": 105,
                     "W": 170}
        gapValues = []
        for i in range(length):
            # pValues and vValues will contain the respective value for each amino acid in each MSA column
            pValues = []
            vValues = []
            gapValues_pos_i = []
            aaAln = SeqIO.parse(open(inputFile, "r"), "fasta")
            for species in aaAln:
                sequenceTmp = str(species.seq)
                aa = sequenceTmp[i]

                # if a sequence contains a gap the vector [0, 0] will be assigned to it since "no amino acid" has no polarity or volume.
                if p.get(str(aa),False):
                    pValues.append(p[str(aa)])
                else:
                    pValues.append("-")

                if v.get(str(aa),False):
                    vValues.append(v[str(aa)])
                else:
                    vValues.append("-")

                # gaps are counted so later the bias of them can be removed
                if aa not in p.keys():
                    gapValues_pos_i.append(0)
                else:
                    gapValues_pos_i.append(1)
            gapValues.append(gapValues_pos_i)

            """
            In this step the [p, v] vectors are centerized (so that their mean will be [0, 0]).
            This is necessary for the POD, since this program uses a Singular Value Decomposition and
            the singular values are the square roots of the eigenvalues of adj(A)*A and adj(A)*A is 
            proportional to Cov(A) only if the columns of A are centerized.
            """
            # remove sequences with gap at position i and center them
            useable = np.array(gapValues[i],dtype=bool)
            pValues_useable = np.array(np.array(pValues)[useable],dtype=float)
            pValues_useable = pValues_useable / 12.3
            vValues_useable = np.array(np.array(vValues)[useable],dtype=float)
            vValues_useable = vValues_useable / 170

            #
            xMean = np.mean(pValues_useable)
            yMean = np.mean(vValues_useable)

            pValues_useable = pValues_useable - xMean
            vValues_useable = vValues_useable - yMean

            # Now the p and v values of this column will be written in the respective files.
            pLine = ""
            vLine = ""
            j = 0

            # save all sequences also the ones with gaps
            # but the centered values for the non gap ones
            for xI in range(len(pValues)):
                if pValues[xI] == "-":
                    pLine += str(pValues[xI]) + " "
                else:
                    pLine += str(pValues_useable[j]) + " "
                    j += 1

            pLine.strip(" ")
            j = 0
            for xI in range(len(vValues)):
                if vValues[xI] == "-":
                    vLine += str(vValues[xI]) + " "
                else:
                    vLine += str(vValues_useable[j]) + " "
                    j += 1
            vLine.strip(" ")
            outputP.write(pLine + "\n")
            outputV.write(vLine + "\n")
    return np.array(gapValues)


"""
This method computes the variance index. It uses a Singular Value Decomposition
to determine the variance along the primary components of each set of vectors representing
a column of the input MSA.
The sum of those two variances is multiplied by the "diversity weight" to generate the
index. The "diversity weight" is the number of different amino acids within each column divided
by the number of sequences in the alignment. This aims to reduce the index of columns with only 
few different amino acids.
"""


def computeVarianceIndex(pFile, vFile, outputDir):
    pData = np.genfromtxt(pFile, delimiter=' ')
    vData = np.genfromtxt(vFile, delimiter=' ')

    with open(os.path.join(outputDir,"indexRaw.csv"), "w") as output:
        # This for loop iterates over all columns of the input MSA
        for i in range(len(pData)):
            # rowVectors contains the actual column
            rowVectors = []
            # pList and vList will contain each amino acid value of the column exactly one (to count the number of different amino acids)
            pData_i = pData[i][np.logical_not(np.isnan(pData[i]))]
            vData_i = vData[i][np.logical_not(np.isnan(vData[i]))]
            pList = np.unique(pData_i)
            vList = np.unique(vData_i)
            for j in range(len(pData_i)):
                rowVectors.append([pData_i[j], vData_i[j]])

            # compute the "diversity weight" as descibed before
            numberOfDifferentAA = max(len(pList), len(vList))
            diversityWeight = numberOfDifferentAA / len(pData_i)

            # use numpy for the actual SVD
            tmpArray = np.array(rowVectors)
            tmpArray = np.transpose(tmpArray)
            # s contains the singular values
            U, s, Vh = np.linalg.svd(tmpArray)
            # the singular values are the square root of the eigenvalues of adj(A)*A
            # they need to be squared
            for k in range(len(s)):
                s[k] = s[k] * s[k]
            index = sum(s) * diversityWeight
            output.write(str(index) + "\n")


"""
This method computes the q-values for each MSA column. Please note, that this is a very simple
significance test. It assumes, that the index of MSA columns on non-positive selection sites is normally
distributed with the same variance and mean as the index on the whole input MSA (which might also contain
positive selection sites).
However, this method generates a value within the interval [0, 1], which indicates, if the index of a certain
column is extraordinarily high.
"""


def computeQvalue(indexFile, outputDirectory, gapValues):
    with open(os.path.join(outputDirectory,"qValues.csv"), "w") as output:
        # Numpy and Scipy is used in this method
        rawData = np.genfromtxt(indexFile)
        mean = np.mean(rawData)
        stdDev = np.std(rawData)
        if stdDev == 0:
            for i in range(len(rawData)):
                qValue = 0
                qValue_norm = 0
                output.write(str(qValue) + "\t" + str(qValue_norm) + "\n")
        else:
            norm_to_zero = norm.cdf(0,mean,stdDev)
            for i in range(len(rawData)):
                # The q-value is the comulated density function at the index value
                qValue = norm.cdf(rawData[i], mean, stdDev)
                qValue_norm = qValue - ((1-qValue)/(1-norm_to_zero))*norm_to_zero
                output.write(str(qValue) + "\t" + str(qValue_norm) + "\n")


# This method just identifies the colums with a q-value higher or equal to (1 - pValueThreshold)
def determinePositiveSelectionSites(qValueFile, outputDirectory):
    with open(os.path.join(outputDirectory,"detectedPositions.csv"), "w") as output:
        finalIndex = np.genfromtxt(qValueFile)
        finalIndex = np.array(finalIndex)
        for i in range(len(finalIndex)):
            if finalIndex[i][1] >= (1 - pValueThreshold):
                # the first column will be named "1", not "0" (and so on)
                output.write(str(i + 1) + "\n")


"""
The program needs 2 input arguments:

The AA-MSA that is to be analyzed
The output directory (there will be more than one output file) 

You can also specify an p-Value threshold as a third
argument. If it is not specified the default p-Value
threshold is 0.1 (please note that the significance test used
in this program is not in any way sophisticated. This is not a "p-value"
in the common sense.)
"""

inputAAaln = sys.argv[1]
outputDir = sys.argv[2]
if len(sys.argv) == 4:
    pValueThreshold = float(sys.argv[3])
elif len(sys.argv) == 3:
    pValueThreshold = 0.1
else:
    print("UNSUPPORTED!")

# Now the defined methods are called.
# The first two steps are essential (they compute the actual index.

if not os.path.isdir(outputDir):
    os.makedirs(outputDir)
gapValues = transformAAtoVectors(inputAAaln, outputDir)
computeVarianceIndex(os.path.join(outputDir,"pol.csv"), os.path.join(outputDir,"mVol.csv"), outputDir)
# The following steps can be skipped, if they are not needed. They perform the significance test.
computeQvalue(os.path.join(outputDir,"indexRaw.csv"), outputDir, gapValues)
determinePositiveSelectionSites(os.path.join(outputDir,"qValues.csv"), outputDir)
