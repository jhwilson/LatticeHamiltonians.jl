# LatticeHamiltonians.jl

[![Build Status](https://github.com/jhwilson/LatticeHamiltonians.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jhwilson/LatticeHamiltonians.jl/actions/workflows/CI.yml?query=branch%3Amain)

The `LatticeHamiltonians` module is designed to construct and manipulate a specific mathematical model called a Hamiltonian. A Hamiltonian is a type of operator used in physics, especially in quantum mechanics, often represented as a matrix.

The Hamiltonian describes the total energy of a system, which can be broken down into kinetic and potential energy components. In this module, these components are modeled as the diagonal (potential energy) and off-diagonal (kinetic energy) elements of the Hamiltonian matrix.

This module uses a technique called tight-binding approximation, often used in solid-state physics, where the system (usually a crystal lattice structure) is divided into discrete sites, each contributing an atomic orbital. The Hamiltonian is then constructed as a matrix that describes the energy and interactions of these orbitals.

The unique aspect of this module is the construction of these Hamiltonian matrices not in the traditional way but via a set of Julia functions that generate Julia expressions (Expr objects), which, when evaluated, perform the necessary calculations for the system.

This is done by using Julia's metaprogramming features, which allow the generation, manipulation, and evaluation of Julia code within Julia itself. This approach provides flexibility and efficiency, particularly when dealing with large systems, as the Hamiltonian is not explicitly stored as a matrix but represented as a function that can perform operations on input vectors, which saves memory and computational effort.

This approach allows a Hamiltonian to be constructed for various lattice systems and potentials by specifying only a few parameters and using efficient looping constructs to iterate over the lattice sites.

The functions in the module generate expressions that construct the Hamiltonian, apply it to input vectors (which can be thought of as the state of the system), and handle system parameters and dimensions. These are all encapsulated in a LatticeHamiltonian structure for ease of use.

In short, this module provides a flexible, efficient, and elegant toolset for constructing and applying Hamiltonians on lattice systems, taking full advantage of Julia's metaprogramming capabilities.

## Package development

We will partly be following [this guide](https://julialang.org/contribute/developing_package/).

## Documenting

Documenting will be done with [Documenter.jl](https://documenter.juliadocs.org/stable/) which can create the documention from comments within files.

## Example usage

In the current version, this is how we build the simplest of tight-binding models

### 1D Nearest neighbor

In this example, we're setting up a one-dimensional tight-binding model using the LatticeHamiltonians package. This model is a fundamental model used to describe electronic behavior in solid state physics.

Here's the commented code:

```julia
using .LatticeHamiltonians

# We use the @lattice_hamiltonian macro to build our model.
LatticeHamiltonians.@lattice_hamiltonian begin 
    # V represents the on-site energies for each lattice site.
    # Here, it's set to 1.0 for all sites in our 1D lattice.
    # This can be thought of as the energy cost for an electron to exist at a site.
    V = ComplexF64[1.0]  # All sites have potential energy 1.0
    
    # T represents the "hopping" terms, i.e., the probability of an electron "hopping" from one site to a neighboring site.
    # The key is the "hopping vector" - here [1] and [-1] signify right and left neighbors respectively.
    # The value is a tuple, where the first two entries are indices for the input and output states.
    # The third entry is the hopping strength.
    T = Dict(
        [1] => ([1], [1], [:(1.0 * t)]),  # Hopping to the right neighbor with strength t
        [-1] => ([1], [1], [:(1.0 * t)])  # Hopping to the left neighbor with strength t
    )
    
    # params is a dictionary where we can store any parameters our model depends on. 
    # Here, we are storing 't', the strength of our hopping term.
    params = Dict{Symbol,ComplexF64}(:t => 1.0)  # Parameter for hopping strength

    # Ls is a list of the number of sites in each dimension.
    # Here, it's [100], meaning we have a 1D lattice (a line) with 100 sites.
    Ls = [100]  # 1D lattice with 100 sites
end
```

The resulting Hamiltonian will describe a system where an electron can hop from one site to its neighbors with a hopping strength determined by `t`, and where being at a site costs an energy given by the site's potential `V`. This Hamiltonian will be applied to wave functions describing the state of electrons in the system, and will be used in subsequent computations to analyze the system's behavior.

#### Breakdown of each part of the above Julia code

1. `using .LatticeHamiltonians`

    - This line is importing the `LatticeHamiltonians` module, which provides the functions and data types needed to work with lattice Hamiltonians. The dot before the module name means that the module is in the current working directory or is a part of the current package.

2. `LatticeHamiltonians.@lattice_hamiltonian begin`

    - `@lattice_hamiltonian` is a macro provided by the `LatticeHamiltonians` module. In Julia, a macro is a special kind of function that generates and manipulates code. Here, `@lattice_hamiltonian` is being used to define a lattice Hamiltonian in a convenient way, with all the details enclosed in the `begin...end` block.

3. `V = ComplexF64[1.0]  # All sites have potential energy 1.0`

    - This line is defining an array named `V` with a single value `1.0` of type `ComplexF64`, which is a complex number with a 64-bit floating-point representation. `V` represents the on-site energy for each site on the lattice. In this case, all sites have the same energy, 1.0.

4. `T = Dict(
        [1] => ([1], [1], [:(1.0 * t)]),  # Hopping to the right neighbor with strength t
        [-1] => ([1], [1], [:(1.0 * t)])  # Hopping to the left neighbor with strength t
    )`

    - This is creating a dictionary named `T`. Dictionaries in Julia are used to store key-value pairs. Here, the keys are `[1]` and `[-1]`, representing the relative positions of neighboring sites, and the values are tuples describing the details of hopping from one site to another. The tuple `([1], [1], [:(1.0 * t)])` represents the hopping interaction between the sites: `[1]` and `[1]` are the indices for the input and output states, and `:(1.0 * t)` (an expression object in Julia representing `1.0 * t`) is the hopping strength.

5. `params = Dict{Symbol,ComplexF64}(:t => 1.0)  # Parameter for hopping strength`

    - Here, we're creating another dictionary called `params` to store parameters that our model depends on. In this case, the only parameter we have is `:t`, which represents the strength of the hopping term. `:t` is a Symbol in Julia, a type used for variables, function names, etc. in code. The `=>` operator is used to associate `:t` with its value `1.0`.

6. `Ls = [100]  # 1D lattice with 100 sites`

    - This line defines `Ls` as an array containing a single integer `100`. `Ls` specifies the number of sites in each dimension of the lattice. Here, we have a 1D lattice with 100 sites.

7. `end`

    - This `end` keyword signals the end of the `@lattice_hamiltonian` block started after `begin`.

In summary, this code defines a one-dimensional lattice Hamiltonian model with 100 sites, where each site has an energy of 1.0 and electrons can hop to adjacent sites with a strength of 1.0.


### 2D Tight-binding model with two orbitals

This is a more complicated Hamiltonian in two-dimensions

```julia
using .LatticeHamiltonians

# We use the @lattice_hamiltonian macro to build our model.
LatticeHamiltonians.@lattice_hamiltonian begin 
    # V represents the on-site energies for each lattice site. Here we have two sites with energies 1 and -1.
    V = ComplexF64[1, -1.0]

    # T represents the "hopping" terms, i.e., the probability of an electron "hopping" from one site to a neighboring site.
    # The keys in the dictionary represent the relative location of the site to which the electron is hopping.
    # The value of each key is a tuple. The first two entries in the tuple are indices for the input and output states.
    # The third entry in the tuple is the hopping strength.
    T = Dict(
        [0,0] => ([1,2],[2,1],ComplexF64[1.0, 1.0]),  # Hopping within a "unit cell" between two orbitals
        [1,0] => ([2], [1], [:(1.0 * t)]),  # Hopping to the right neighbor along x-axis with strength t
        [-1,0] => ([1], [2], [:(1.0 * t)]),  # Hopping to the left neighbor along x-axis with strength t
        [0,1] => ([2], [1], [:(1.0 * t)]),  # Hopping to the top neighbor along y-axis with strength t
        [0,-1] => ([1], [2], [:(1.0 * t)]),  # Hopping to the bottom neighbor along y-axis with strength t
    )

    # params is a dictionary where we can store any parameters our model depends on.
    # Here, we are storing 't', the strength of our hopping term.
    params = Dict{Symbol,ComplexF64}(:t => 1.0)

    # Ls is a list of the number of sites in each dimension.
    # Here, it's [50,50], meaning we have a 2D square lattice (a grid) with 50 sites in each dimension.
    Ls = [50,50]
end
```

This example creates a lattice Hamiltonian using the `@lattice_hamiltonian` macro from the LatticeHamiltonians package. It models a system on a 2D lattice (a grid of points) with a specific form of potential energy and interactions.

1. `V = ComplexF64[1, -1.0]`: This line defines the potential energy on each site in the lattice. V is an array of complex numbers, and it corresponds to the diagonal elements of the Hamiltonian. In this case, there are two types of sites in the lattice, one with a potential of 1 and the other with a potential of -1.

2. `T` is a dictionary that specifies the kinetic energy, or hopping terms, which represent the interaction between adjacent sites. The keys of the dictionary are the relative coordinates between two sites, and the values are tuples containing three elements: the indices of the starting sites, the indices of the ending sites, and the hopping strength between these sites. For example, `[1,0] => ([2], [1], [:(1.0 * t)])` represents an interaction (hopping) from the second type of site to the first type of site in the positive x direction with a strength of `t`.

3. `params = Dict{Symbol,ComplexF64}(:t => 1.0)`: This line defines a dictionary of parameters that can be used in the expressions for the hopping terms. In this case, we have one parameter `t` with a value of `1.0`.

4. `Ls = [50,50]`: This specifies the dimensions of the lattice, in this case, a 50x50 square lattice.

The `@lattice_hamiltonian` macro then constructs a function that can apply the resulting Hamiltonian to any input state (a vector). The Hamiltonian is not explicitly stored as a matrix, but instead is represented as a function that performs certain operations on the input state, saving memory and allowing for more efficient computations. This approach is particularly beneficial for large-scale systems and high-dimensional lattices.

# Developing the front-end

Taking the above, we want to modify how Julia interprets code in order to make the above easier for the user.
The end result should be a call that creates an object of type `LatticeHamiltonian`

```julia
using LatticeHamiltonians

H = @lattice_hamiltonian begin
    orbitals [1.0, 1.0]
    (0,0) -> [0 1 ; 1 0]
    (1,0) -> [0 0 ; t 0]
    (-1,0) -> [0 t ; 0 0]
    (0,1) -> [0 0 ; t 0]
    (0,-1) -> [0 t ; 0 0]
    t = 1.0
    Ls = [50, 50]
end
```

The major task is going to be getting the `macro` `@lattice_hamiltonian` to interpret this new syntax that we make up and then build the Hamiltonian like normal.

With this all set up, we can write a little note on how to use the package and launch it!

Key things to learn:
- Metaprogramming in Julia.
- Working within VSCode with Julia
- Using Git and Github: **Always work on your own branch and then merge it into the main branch when everything works!**
- Making basic `@macros`.
- Working with `Expr` which are Julia expressions that we can manipulate.
- Difference between `Symbol` and `Expr`
- Compile-time and run-time (These are different in Julia!).