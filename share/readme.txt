eos: A lightweight header-only 3D Morphable Model fitting library in modern C++11/14
=========

Files in this directory:

- ibug2did.txt:
	Mappings from the popular IBUG 68-point 2D facial landmarks markup to
	Surrey Face Model indices.

- sfm_shape_3448.bin:
	The public shape-only Surrey 3D Morphable Face Model.
	To obtain a full 3DMM and higher resolution levels, follow the
	instructions at <todo: add link to the page of the Uni>.
	Details about the different models can be found in:
	<todo: add publication>.

- reference.obj:
	The reference 3D shape used to built the model. We make it available so 
	that new user-defined landmark points can be marked in this lowest-resolution
	model, if the points exist here.
