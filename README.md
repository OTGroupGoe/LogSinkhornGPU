# LogSinkhornGPU

LogSinkhorn routines using the GPU. 

This package is greatly inspired in `geomloss`, but attempts to provide more robust stopping criteria for the LogSinkhorn algorithm. It also provides a basic class from which it is easy to build custom LogSinkhorn's (for example for unbalanced transport) by changing a couple of methods.

Our objective is to provide batched LogSinkhorn for the following matrix of backends and measure formats:

		Torch		KeOps		CUDA (lots of small problems)
Point clouds
Grid


