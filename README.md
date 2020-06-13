# Synthetic X-Ray Project

Please refer to these:

A. Moturu and A. Chang, “Creation of synthetic x-rays to train a neural network to detect lung cancer.” http://www.cs.toronto.edu/pub/reports/na/Project_Report_Moturu_Chang_1.pdf, 2018.

A. Chang and A. Moturu, "Detecting Early Stage Lung Cancer using a Neural Network Trained with Patches from Synthetically Generated X-Rays." http://www.cs.toronto.edu/pub/reports/na/Project_Report_Moturu_Chang_2.pdf, 2019. 

Creation of Synthetic X-Rays to Train a Neural Network to Detect Lung Cancer

*Edit: chestCT0/ is no longer on the repository; you may use your own chest CT scan (folder containing DICOM files of the slices).*

Directory:  
>   chestCT0/  
	Chunk.hpp  
	Constants.h  
	Coordinate.cpp  
	Coordinate.hpp  
	CTtoTrainingDataParallel.m  
	CTtoTrainingDataPointSource.m  
	dicomHandler.m  
	Lung Segmentation/
	main.cpp  
	Makefile  
	methods.cpp  
	methods.hpp  
	NoduleSpecs.hpp  
	Pixel.hpp  
	positions_0.txt  
	ProjectionPlane.hpp  
	random_shape_generator.py  
	readNPY.m  
	readNPYheader.m  
	SimulatedRay.cpp  
	SimulatedRay.hpp  
	textCTs/  
	textNodules/  
	textXRays/  
	Voxel.cpp  
	Voxel.hpp  

The Lung Segmentation folder contains the segment_lungs.py file which segments lungs for randomized nodule placement,  
but positions_0.txt contains manually selected points as of now.  

A CT scan (chestCT0/) is needed as input to create X-rays. 
Running this program will create X-rays that are placed in the chestXRays0 folder.  
The textCTs, textNodules, and textXRays folders are used in the process of making point-source X-rays.  

**CTtoTrainingDataParallel.m** is the main program that makes parallel-ray X-rays (makes ~400 X-rays every 10 minutes).  
Dependencies:  
>	random_shape_generator.py  
	--  
	readNPY.m  
	readNPYheader.m  
	dicomHandler.m  
	--  
	positions_0.txt (etc.)  
	chestCT0/ (etc.)  

To run in MatLab2018a, type into the console: CTtoTrainingDataParallel(CTFolderName, specificationsFileName);  
>	CTFolderName is the folder containing the CT slices files  
	specificationsFileName is the file that contains nodule positions  

Example run: **CTtoTrainingDataParallel('chestCT0/I/NVFRWCBT/5O4VNQBN/', 'positions_0.txt');**  

**CTtoTrainingDataPointSource.m** is the main program that makes point-source X-rays (currently makes 7 X-rays from 7 different point sources).  
Dependencies:  
>	random_shape_generator.py  
	--  
	readNPY.m  
	readNPYheader.m  
	dicomHandler.m  
	--  
	positions_0.txt (etc.)  
	chestCT0/ (etc.)  
	--  
	Constants.h  
	Chunk.hpp  
	Coordinate.hpp  
	methods.hpp  
	SimulatedRay.hpp  
	ProjectionPlane.hpp  
	Voxel.hpp  
	NoduleSpecs.hpp  
	Pixel.hpp  
	--  
	main.cpp  
	methods.cpp  
	Coordinate.cpp  
	SimulatedRay.cpp  
	Voxel.cpp  
	--  
	Makefile  

To run in MatLab2018a, type into the console: CTtoTrainingDataPointSource(CTFolderName, specificationsFileName);  
>	CTFolderName is the folder containing the CT slices files  
	specificationsFileName is the file that contains nodule positions  

Example run: **CTtoTrainingDataPointSource('chestCT0/I/NVFRWCBT/5O4VNQBN/', 'positions_0.txt');**  
