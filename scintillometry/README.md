# toeplitz_decomposition

### Extracting Data from your binned file ###

To extract your binned data, use extract_realData2.py

The format is

$ python extract_realData2.py binnedDataFile numofrows numofcolms offsetn offsetm n m

where n is the number of blocks (or the number of frequency bins) and m is the size of each block (or the number of time bins). numofrows 
and numofcolms are the total number of frequency and time bins of your dynamic spectrum.

So for example. if I wanted numofrows= 2048, numofcols=330, offsetn= 0, offsetm = 140, n=4, m=8, I would call

python extract_realData2.py gb057_1.input_baseline258_freq_03_pol_all.rebint.1.rebined 2048 330 0 140 4 8
This will create a folder at ./processedData/gate0_numblock_4_meff_16_offsetn_0_offsetm_140

The name of the folder it's create is usually

gate0\_numblock\_(n)\_meff\_(mx2)\_offsetn\_(offsetn)\_offsetm\_(offsetm)

### Plotting results ###

The factorized toeplitz matrix is located in the ./results directory under a npy file with the \_uc.npy at the end. 

The format is 

$ python plot_simulated.py bnumofrows numofcolms offsetn offsetm
