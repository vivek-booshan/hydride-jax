.. image:: doc/static/assets/hydride_logo.svg
   :width: 400
   :align: center

Hydride-jax - Adding hydrogen atoms to molecular models
=======================================================

NOTE: the only difference is relax_hydrogen now uses jax to allow for jax.grad in larger pipelines additionally, the transition from cython to jax fundamentally changes the optimization from a discrete, heuristic-based search to a continuous, gradient-based refinement.
While the physics (the UFF force field and electrostatics) remains the same, the "path" the computer takes to find the optimal hydrogen positions is entirely different.

Legacy Cython: Used a Grid Search. It would discretize the 360 rotation into small steps (e.g., 36 steps of 10∘). It physically moved the atoms to each step, calculated the energy, and picked the one with the lowest value.
Legacy Cython: Inherently Global. Because it checks the entire 360 circle in steps, it can bypass large energy barriers (like an oxygen atom blocking the way) to find a better spot on the other side.
Legacy Cython: Sequential. It optimizes one rotatable bond at a time.
Legacy Cython: Not-Differentiable. There was no need for it to be, so not really a con. 

New JAX Logic: Uses Gradient Descent (Adam). It treats the rotation angle θ as a continuous differentiable variable. JAX calculates the exact derivative (gradient) of the energy with respect to θ and "rolls" the atom down the energy hill toward the nearest minimum.
New JAX Logic: Inherently Local, but supplemented. A single gradient descent run can get stuck in a "local valley." To fix this, the new JAX algorithm uses a multi-start approach (running parallel optimizations from 0, 120, and 240) to mimic the global coverage of the old grid search.
New JAX Logic: Parallel. Uses jit and vectorization. Scales to GPU and handles bigger complexes.
New JAX Logic: Differentiable. Because the entire solver is written in JAX, you can take the gradient of the output with respect to the input. This allows Hydride to be used as a layer inside a models (like AF3) or a any refinement pipeline that uses relax_hydrogen via relax_hydrogen_jit.

Many tasks in structural biology ranging from simulations and hydrogen
bond detection to mere visual analysis, require complete molecular
models.
However, most experimentally determined structures do not include
the position of hydrogen atoms, due to their small size and electron
density.

*Hydride* is an easy-to-use program and library written in Python that
adds missing hydrogen atoms to molecular models based on known bond
lengths and angles.
Since it does not require force-field parameters for the specific
molecule(s), it can be used for adding hydrogen atoms to almost any
organic molecule - from small ligands to large protein complexes.

.. image:: doc/images/cover_structure.svg
   :width: 400
   :align: center

|

Installation
------------

In order to use *Hydride-jax* you need to have Python (at least 3.10) installed.

You can install *Hydride-jax* via

.. code-block:: console

   $ uv add git+https://github.com/vivek-booshan/hydride-jax


Usage
-----

In its basic invocation *Hydride* reads an input structure file, adds hydrogen
atoms to the molecular model and writes the resulting model into an output
structure file.

.. code-block:: console

   $ hydride -i input_structure.pdb -o output_structure.pdb

Python API
----------

*Hydride* also provides a Python API to add hydrogen atoms to ``AtomArray``
objects from `Biotite <https://www.biotite-python.org/>`_.

.. code-block:: python

   # matches legacy code
   atom_array, _ = hydride.add_hydrogen(atom_array)
   atom_array.coord = hydride.relax_hydrogen(atom_array)

   # if you need the explicit jax version to utilize jax.grad
   # use hydride.relax_hydrogen_jit()

   # gets all the dirty params needed for relax_hydrogen that can't be jitted
   params = hydride.get_relaxation_params(atom_array)

   @jax.jit
   def loss_fn(coords, params):
      final_coords, trajectories, energies = hydride.relax_hydrogen_jit(coords, *params)
      return jnp.sum(final_coords**2)

   # obviously can be any coord, not just the one by relax_hydrogen
   coord = jnp.array(atom_array.coord)
   grads = jax.grad(loss_fn)(coord, params)
