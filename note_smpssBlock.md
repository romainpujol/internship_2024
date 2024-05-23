# Introduction

The following note is a collection of remarks on Block-s in SMS++ based on its implementation in MCFBlock.

1. [Introduction](#introduction)
2. [`Block.h` general comment]({blockh-general-comment})
3. [`MCFBlock.h` overview](#mcfblockh-overview)
4. [Comments from `MCFBlock.cpp`](#comments-from-mcfblockcpp)

# `Block.h` general comment
For ease of reading here is a copy of the general comment found in Block.h.

---

Header file for the abstract class Block, which represents the basic concept of a "block" in a block-structured mathematical model.

A Block contains some Variable [see Variable.h], some Constraint [see Constraint.h], one Objective [see Objective.h], and some sub-Block; furthermore, it can be contained into a father Block. Variable and Constraint can either be static (i.e., they are guaranteed to be always there throughout the life of the model, although of course the value that the Variable attain is, well, variable) or dynamic (i.e., that may appear and disappear during the life of the model). Conversely, the sub-Block are static, i.e., they cannot individually appear or disappear. Dynamic Variable and Constraint allow to cope with "very large models" by means of column and row generation techniques.

A Block can be attached to any number of Solver [see Solver.h], that can then be used to solve the corresponding mathematical model.

Variable and Constraint in a Block can be arranged in any number of "sets", or "groups", each of which can be a multi-dimensional array with (in principle) an arbitrary number of dimensions. The idea is that a model with a specific structure (say, a Knapsack, a Traveling Salesman Problem, a SemiDefinite program, ...) be represented as a specific derived class of Block. Hence, its Variable and Constraint will be organized into appropriate, "natural" (multi-dimensional) vectors, and will be accessed as such by specialized Solver that can exploit the specific structure of the Block. Actually, the Variable and Constraint can be represented implicitly by just providing the data that characterizes them (the weights and costs of the item in a knapsack, an annotated graph, the size of a square semidefinite matrix, ...), and a specialized Solver will only need access to that data (characterizing the instance of the problem) to be able to solve the Block. We call this the "physical representation" of the Block. This means that the Constraint may not even need to be explicitly constructed, as specialized Solver already "know" of them. Conversely, the Variable will always have to be constructed, as they are the place where Solver will write the solution of the Block once they have found it.

However, a Block can also be attached to general-purpose solvers that only need the Variable and Constraint to be of some specific type (say, single real numbers and linear functions ...). Hence, the base Block class provides a mechanism whereby, upon request, the Block "exhibits" its Variable and Constraint as "unstructured" lists of (multi-dimensional arrays of) Constraint and Variable; we call this the "abstract representation" of the Block.

A Block supports "lazy" modifications of all its components: each time a modification occurs, an appropriate Modification object [see Modification.h] is dispatched to all Solver "interested" in the Block, i.e., either directly attached to the Block or attached to one of its ancestors. Hence, the next time they are required to solve the Block they will know which modifications have occurred since the last time they have solved it (if any) and be able to react accordingly, hopefully re-optimizing to improve the efficiency of the solution process. Each Solver is only interested in the Modification that occurred after it was (indirectly) attached to the Block and since the last time it solved the Block (if any), but it has the responsibility of cleaning up its list of Modification. The specific classes BlockMod, BlockModAdd and BlockModRmv are also defined in this file to contain all Block-specific Modification.

Block can "save" the current status of its Variable into a Solution object [see Solution.h], and read it back from a Solution object. If Constraint have dual information attached to them, this can similarly be saved.

Block explicitly supports the notion that a model may need to be modified for algorithmic purposes, i.e., by producing either a Reformulation (a different Block that encodes a problem whose optimal solutions are optimal also for the original Block), a Relaxation (a different Block whose optimal value provides valid global lower/upper bounds for the optimal value of the original Block, assuming that was a minimization/maximization problem, while hopefully being easier to solve), or a Restriction (a different Block that encodes a problem whose feasible region is a strict subset of that of the original Block, which hopefully makes it easier to solve). These are called "R3 Block" of the original Block. The set of R3 Block of a given Block is defined by the Block itself; the base class provides no general R3 Block. However, since one of the basic design decisions of SMS++ is that "names" of Variable (and Constraint) are their memory address, it is not in general possible to "copy Variable" (a new Variable will always be a different Variable for any existing one). Therefore, the Block class provides support from the fact that an original Block can map back solution information produced by one of its R3 Blocks. This operation is, again, specific for each Block and R3 Block of its, and the base class provides no general mechanism for it (besides the interface).

# `MCFBlock.h` overview

Below is a breakdown of MCFBlock.h which is a Block concept for solving (linear) Min-Cost Flow (MCF) problems. 

## General Structure

### Includes
- Includes headers for related classes: `Block.h`, `LinearFunction.h`, `FRealObjective.h`, `FRowConstraint.h`, `OneVarConstraint.h`, and `Solution.h`.

### Namespace
- The classes are defined within the `SMSpp_di_unipi_it` namespace.

## Class and Type Definitions

### Forward Declarations
- `MCFBlock` and `MCFSolution` classes are forward-declared.

### Type Definitions
- Various types related to `MCFBlock` are defined, including pointers and iterators for vectors of `MCFBlock` pointers.

## Class MCFBlock

### Class Purpose
- Implements the Block concept for the linear Min-Cost Flow problem, involving a directed graph with nodes and arcs, where each node has a flow deficit, and each arc has a capacity and cost.

### Mathematical Formulation
- The class comments provide a detailed mathematical formulation of the Min-Cost Flow problem, including the primal and dual formulations, constraints, and complementary slackness conditions.

### Class Definition
- The `MCFBlock` class inherits from the `Block` class and contains public and protected members.

## Public Members

### Public Types
- Types for representing flow variables, costs, objective function values, and their respective vectors and iterators.

### Constructor and Destructor
- The constructor initializes the `MCFBlock` object with a pointer to a parent Block.
- The destructor ensures proper cleanup.

### Methods for Initialization
- `load()`: Loads the MCF instance from memory or a stream, and sets up the problem's nodes, arcs, capacities, costs, and deficits.
- `deserialize()`: Extends deserialization to include specific MCFBlock data.
- `generate_abstract_variables()`, `generate_abstract_constraints()`, and `generate_objective()`: Methods for generating the abstract representation of the MCF problem's variables, constraints, and objective.

### Data Access Methods
- Methods to get the number of nodes, arcs, static nodes, and arcs, and to check if certain constraints or variables exist.
- Methods to retrieve specific data such as starting and ending nodes of arcs, arc costs and capacities, node deficits, and the current solution.

### Feasibility and Optimality Checks
- Methods to check if the current solution is feasible (flow, bound, dual feasibility) and if it satisfies complementary slackness conditions.
- Methods to determine if the current solution is approximately feasible or optimal.

### R3 Block Handling
- Methods for managing R3 Blocks, which are presumably related to some advanced block management or transformation functionality.
- `get_R3_Block()`, `map_back_solution()`, and `map_forward_solution()`: Methods to handle R3 Blocks, which seem to involve creating, mapping solutions, and modifications.

### Solution Handling
- Methods to get, set, and manipulate the current solution, including primal flow, dual potentials, and reduced costs.
- `get_Solution()`, `get_objective_value()`, `get_x()`, `get_pi()`, `get_rc()`, `set_x()`, `set_pi()`, and `set_rc()`.

## Protected Members

### Protected Fields
- Fields for storing the number of nodes, arcs, static nodes, static arcs, and other parameters related to the MCF problem.

## Private Members

### Private Methods
- Methods for internal operations such as initialization and cleanup.
- Static initialization method to register MCFBlock methods into method factories.
- Utility methods for handling various internal operations and state checks.

## Additional Classes

### MCFBlockMod
- A class for handling modifications to an `MCFBlock`.
- Types of modifications include changing costs, capacities, deficits, opening/closing arcs, adding, and removing arcs.

### MCFBlockRngdMod
- Derived from `MCFBlockMod`, handles ranged modifications.

### MCFBlockSbstMod
- Derived from `MCFBlockMod`, handles subset modifications.

### MCFSolution
- A class representing a solution to the MCF problem, including flow values and node potentials.

## Summary

The `MCFBlock` class and its related classes provide a comprehensive framework for modeling and solving linear Min-Cost Flow problems using a variety of methods and structures. It includes detailed mechanisms for handling problem data, generating and managing solutions, and ensuring feasibility and optimality. The design also incorporates advanced features such as abstract and physical modifications, and solution management, making it a robust tool for solving complex flow problems in networks.

# Comments from `MCFBlock.cpp`

Below is a collection of remarks on the implementation of MCFBlock.

## General Structure

### Macros

- `#ifndef NDEBUG` and `#define CHECK_DS 0`: A conditional macro for performing long and costly checks on the data structures representing the abstract and physical representations. This is controlled by the `NDEBUG` macro, which is used for debugging purposes.

### Namespace and Using

- `using namespace SMSpp_di_unipi_it;`: Uses the `SMSpp_di_unipi_it` namespace.
- Several `using` declarations for types from the `Block` class, such as `Index`, `c_Index`, `Range`, `c_Range`, `Subset`, and `c_Subset`.

## Implementation Details

### How to create an instance of a Block

R3 Block seems to be used, it comes with functions to edit a Block. Seems to have built-in methods to handle regularization. 

### How to define (abstract) variables

Done through  `void generate_abstract_variables( Configuration *stvv = nullptr ) override;`. There are two kinds of abstract variables: static variables and dynamic variables. If I understood correctly, the second ones offer the flexibility to being turned on/off without being deleted from memory. This seems to be extremely relevant for modifications of a given problem in the context of an iterative algorithm involving said problem as a subroutine. We should aim to have a similar feature with OTBlock.

### How to define constraints

TODO

### How to define objective function

TODO

# Questions related to OTBlock

Here are points to work on, last update 23th May.

- See more precisely the differences between Block and R3 Block. How does one define a regularized variant? It should be with one of the factories...
- What are the variables that should be static and the ones that should be dynamic for OTBlock? (See MCFBlock for an example of static/dynamic variables.)
- How do the Configuration-s work? There seems to be a simple default factory to generate a Configuration and then more involved methods.
