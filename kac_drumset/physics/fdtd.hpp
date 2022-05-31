#pragma once

// core
#include <array>
#include <math.h>
#include <stdexcept>
#include <vector>

std::vector<double> FDTDWaveform2D(
	std::vector<std::vector<double>> u_0,
	std::vector<std::vector<double>> u_1,
	const std::vector<std::vector<int>>& B,
	const double& c_0,
	const double& c_1,
	const double& d,
	const int& T,
	const std::array<int, 2> x_range,
	const std::array<int, 2> y_range,
	const std::array<double, 2> w
) {
	/*
	Generates a waveform using a 2 dimensional FDTD scheme.
	input:
		u_0 = initial fdtd grid at t = 0
		u_1 = initial fdtd grid at t = 1
		B = boundary conditions
		c_0 = first fdtd coefficient related to the courant number
		c_1 = second fdtd coefficient related to the courant number
		d = decay coefficient
		T = length of simulation in samples.
		x_range = tuple representing the range across the x axis to update
		y_range = tuple representing the range across the y axis to update
		w = the coordinate at which the waveform is sampled.
	output:
		waveform = W[n] ∈  (λ ** 2)(
								u_n_x+1_y + u_n_x-1_y + u_n_x_y+1 + u_n_x_y-1
							) + 2(1 - 2(λ ** 2))u_n_x_y - d(u_n-1_x_y) ∀ u ∈ R^2
	*/

	// handle errors
	if (u_0.size() != u_1.size() || u_0[0].size() != u_1[0].size()) {
		throw std::invalid_argument("u_0 and u_1 differ in size.");
	}
	if (u_0.size() != B.size() || u_0[0].size() != B[0].size()) {
		throw std::invalid_argument("u_0 and B differ in size.");
	}
	// initialise variables
	std::vector<double> waveform(T);	   // output waveform
	std::vector<std::vector<double>> u(	   // the fdtd grid
		u_0.size(),
		std::vector<double>(u_0[0].size(), 0.)
	);
	// handle initial events
	waveform[0] = u_0[w[0]][w[1]];
	waveform[1] = u_1[w[0]][w[1]];
	// main loop
	for (unsigned int t = 2; t < T; t++) {
		if ((t % 2) == 0) {
			for (unsigned int x = x_range[0]; x < x_range[1]; x++) {
				for (unsigned int y = y_range[0]; y < y_range[1]; y++) {
					// dirichlet boundary conditions
					if (B[x][y] != 0) {
						u[x][y] = (u_1[x][y + 1] + u_1[x + 1][y] + u_1[x][y - 1]
								   + u_1[x - 1][y])
								* c_0
							+ c_1 * u_1[x][y] - d * u_0[x][y];
					}
				};
			};
			u_0 = u;
		} else {
			for (unsigned int x = x_range[0]; x < x_range[1]; x++) {
				for (unsigned int y = y_range[0]; y < y_range[1]; y++) {
					// dirichlet boundary conditions
					if (B[x][y] != 0) {
						u[x][y] = (u_0[x][y + 1] + u_0[x + 1][y] + u_0[x][y - 1]
								   + u_0[x - 1][y])
								* c_0
							+ c_1 * u_0[x][y] - d * u_1[x][y];
					}
				};
			};
			u_1 = u;
		}
		waveform[t] = u[w[0]][w[1]];
	}
	return waveform;
}