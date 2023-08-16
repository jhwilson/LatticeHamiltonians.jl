# LatticeHamiltonians.jl

[![Build Status](https://github.com/jhwilson/LatticeHamiltonians.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jhwilson/LatticeHamiltonians.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](http://jhwilson.com/LatticeHamiltonians.jl/dev/)


`LatticeHamiltonians` provides a set of tools for constructing and applying Hamiltonians on lattice systems.
Using Julia's metaprogramming capabilities, this module allows Hamiltonian to be constructed for various lattice systems and potentials by specifying only a few parameters and using efficient looping constructs to iterate over the lattice sites.

The functions in the module generate expressions that construct the Hamiltonian, apply it to input vectors, and handle system parameters and dimensions. These are all encapsulated in a LatticeHamiltonian structure for ease of use.

For more detailed usage information, see the [documentation](http://jhwilson.com/LatticeHamiltonians.jl/dev/).

# Contributing

Please see [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for information on contributing to this project.
