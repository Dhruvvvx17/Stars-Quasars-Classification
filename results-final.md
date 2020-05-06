# All Features (Not Stratified)

## Accuracy
	File    KNN      Cross    KNN      Cross    KNN      Cross    KNN      Cross    KNN      Cross
	k       3        3        5        5        7        7        9        9        11       11
	cat1    96.92    96.41    97.44    96.92    97.44    96.92    97.44    96.92    96.92    96.41
	cat2    96.34    96.25    96.44    96.34    96.07    95.98    95.89    95.80    95.98    95.89
	cat3    96.12    96.20    96.82    96.74    96.74    96.51    96.51    96.28    96.28    96.20
	cat4    86.02    85.97    86.10    86.04    85.83    85.75    85.73    85.67    85.69    85.65

## F1
	File      KNN     Cross   KNN     Cross   KNN     Cross   KNN     Cross   KNN     Cross
	k         3       3       5       5       7       7       9       9       11      11
	cat1      97      96      97      97      97      97      97      97      97      96
	cat2      96      96      96      96      96      96      96      96      96      96
	cat3      96      96      97      97      97      97      97      96      96      96
	cat4      86      86      86      86      86      86      86      86      86      86


# All Features (Stratified)

## Accuracy
	File    KNN      Cross    KNN      Cross    KNN      Cross    KNN      Cross    KNN      Cross
	k       3        3        5        5        7        7        9        9        11       11
	cat1    97.44    97.44    97.44    97.44    96.92    96.92    96.92    96.92    96.92    96.92
	cat2    94.61    94.61    95.16    94.97    94.88    94.70    94.70    94.52    94.42    94.33
	cat3    95.58    95.50    95.50    95.11    95.42    95.19    95.58    95.35    95.35    95.11
	cat4    85.35    85.31    85.57    85.39    85.45    85.31    85.73    85.53    85.44    85.24

	F1        KNN     Cross   KNN     Cross   KNN     Cross   KNN     Cross   KNN     Cross
	k         3       3       5       5       7       7       9       9       11      11
	cat1      97      97      97      97      97      97      97      97      97      97
	cat2      95      95      95      95      95      95      95      95      94      94
	cat3      96      96      96      95      95      95      96      95      95      95
	cat4      85      85      86      85      85      85      86      86      85      85


# Without Extinction Features (Not Stratified) (k = 5)

## Accuracy
	File    KNN      Cross
	cat1    97.44    96.92
	cat2    96.25    96.16
	cat3    96.28    96.28
	cat4    86.03    85.95

## F1
	File    KNN   Cross
	cat1    97    97
	cat2    96    96
	cat3    96    96
	cat4    86    86


# Without Pairwise Difference Features (Not Stratified) (k = 5)

## Accuracy
	File    KNN      Cross
	cat1    97.44    96.92
	cat2    96.25    96.16
	cat3    96.28    96.28
	cat4    86.03    85.95

## F1
	File    KNN   Cross
	cat1    97    97
	cat2    96    96
	cat3    96    96
	cat4    86    86


# Time
	File    Time       Train Size    Test Size
	cat1    0:00:09    454           195
	cat2    0:03:43    2552          1094
	cat3    0:04:52    3006          1289
	cat4    1:43:57    23424         10039

# K(10) Fold Cross Validation
	file   knn spect
	cat1 96.92 96.77
	cat2 95.72 95.69
	cat3 96.16 96.06
