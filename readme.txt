The project consists of 3 excecutable python files:
	1. main_naive.py:
		It consists of a basic implementation in which 702-dimensional training set was given to our GMM model				
		and results were reported accordingly.
	2. main_unoptimised.py:
		It consists of a implementation in which 703-dimensional training set was given to our GMM model.
		The one extra added feature was class of the user: authentic(1) and imposter(0).
		Results were poor than the previous implementation.
	3. main_optimised.py:
		It consists of a basic implementation in which 702-dimensional training set was given to our GMM model		
		and the number of clusters were set to number of users whose keystroke dynamics were provided. We then 
		checked for each training example if it was classified accordingly and hence reported the results
		accordingly.
