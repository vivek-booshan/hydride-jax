.. image:: doc/static/assets/hydride_logo.svg
   :width: 400
   :align: center

Hydride-jax - Adding hydrogen atoms to molecular models
=======================================================

NOTE: the only difference is relax_hydrogen now uses jax to allow for jax.grad in larger pipelines additionally, the transition from cython to jax fundamentally changes the optimization from a discrete, heuristic-based search to a continuous, gradient-based refinement.
While the physics (the UFF force field and electrostatics) remains the same, the "path" the computer takes to find the optimal hydrogen positions is entirely different.
.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Feature
     - Legacy Cython Backend
     - New JAX Backend (rejax)
   * - **Optimization Algorithm**
     - **Grid Search**: Discretizes the 360° rotation into fixed steps (e.g., 36 steps of 10°). It picks the step with the absolute lowest energy.
     - **Gradient Descent (Adam)**: Treats the rotation angle :math:`\theta` as a continuous variable. It "rolls" down the energy hill toward the nearest minimum.
   * - **Search Space**
     - **Global**: By checking the entire circle, it naturally bypasses large energy barriers (e.g., steric clashes) to find the global minimum.
     - **Local (Multi-start)**: Inherently local, but supplemented by running parallel optimizations from 0°, 120°, and 240° to ensure global coverage.
   * - **Execution Model**
     - **Sequential**: Optimizes one rotatable bond at a time on the CPU.
     - **Parallel**: Uses JIT-compilation and vectorization to optimize all bonds simultaneously. Highly scalable on GPU/TPU hardware.
   * - **Differentiability**
     - **Non-Differentiable**: self-explanatory
     - **Fully Differentiable**: ``relax_hydrogen_jit`` is differentiable with respect to the input, enabling use as a layer in ML models like AlphaFold 3.

Key Advantages of the JAX Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Continuous Precision**: Unlike the grid search, which is limited by its step size, gradient descent can settle on the mathematically exact energy minimum.
* **Deep Learning Integration**: By using ``relax_hydrogen_jit()``, *Hydride* can be embedded directly into neural network architectures or any refinement pipeline requiring gradients.
* **High-Throughput Scale**: The JAX version is designed to handle massive molecular complexes and large batches of structures with minimal performance degradation.

.. note::
   While the underlying algorithm has changed, the high-level ``relax_hydrogen()`` wrapper remains a drop-in replacement for legacy code, ensuring existing scripts continue to work without modification.

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
