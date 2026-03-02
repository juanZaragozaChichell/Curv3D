print('Initializing')
import numpy as np
import trimesh
import open3d as o3d
import MeshSlicer
import cma
import math
from time import perf_counter as clock
import sys
from tqdm import tqdm
import os
from joblib import Parallel, delayed

print('Imports done')

def vector_to_segment(vector, segment):
    r"""
    Linearly map an element of the *unit interval* onto an arbitrary segment.

    Given :math:`x\\in[0,1]` return :math:`(1-x)a + xb`.

    Parameters
    ----------
    vector : float | np.ndarray
        Value(s) in the closed interval ``[0, 1]``.
    segment : tuple[float, float]
        Two‑element tuple ``(a, b)`` with target bounds.

    Returns
    -------
    float | np.ndarray
        Mapped value(s) in the interval ``[a, b]``.
    """
    a, b = segment
    return(1-vector)*a + vector*b

def segment_to_vector(segment_values, bounds):
    r"""
    Inverse mapping of :func:`vector_to_segment`.

    Parameters
    ----------
    segment_values : float | np.ndarray
        Value(s) in the interval specified by *bounds*.
    bounds : tuple[float, float]
        Original bounds ``(a, b)``.

    Returns
    -------
    float | np.ndarray
        Normalised value(s) in ``[0, 1]``.
    """
    min_bound, max_bound = bounds
    vector =(segment_values - min_bound)/(max_bound - min_bound)
    return vector

def collision_score(list_of_levelsets ):
    r"""
    Penalise non‑printable level‑sets.

    Parameters
    ----------
    list_of_levelsets : list[trimesh.Trimesh]
        Consecutive level‑set meshes.

    Returns
    -------
    float
        Collision penalty.  Each non‑printable level‑set adds *1 × 10⁶*.
    """
    list_of_bool = mesh_obj.are_levelsets_printable(list_of_levelsets )
    array_of_bool = ~np.array(list_of_bool)
    return 1e9*array_of_bool.sum()

def _distance_score(list_of_levelsets ):
    r"""
    Penalise layer‑thickness violations.

    Returns
    -------
    tuple
        ``(score, smallest_distance, biggest_distance)`` with the summed
        error multiplied by a factor.
    """
    is_surface_at_good_distsance, biggest_distance, smallest_distance = mesh_obj.check_distance_validity_between_levelsets(list_of_levelsets )
    min_criteria = smallest_distance - mesh_obj.min_thickness
    max_criteria = mesh_obj.max_thickness - biggest_distance
    # if this two are positive, then we are within the criteria
    # make them negative so that in case of failing we can feed it to score directly, hence the max
    error_min = max(0, -min_criteria)
    error_max = max(0, -max_criteria)
    score = error_min + error_max
    return 1e9 * score, smallest_distance, biggest_distance

def objective_function_information(solution):
    r"""
    Compute objective score **and** auxiliary diagnostics.

    Returns
    -------
    tuple
        ``(score, col_score, distance_score, min_error, max_error,
        n_levelsets, smallest_distance, biggest_distance)``.
    """
    H, n_levelsets = solution
    n_levelsets = np.round(n_levelsets).astype(int)
    # H, n_levelsets = self.transform(solution)
    mesh_obj.H = H # actualize the H map in the mesh object structure
    various_levelsets = np.linspace(H.min() + 1e-3, H.max()-1e-3, n_levelsets)
    various_levelsets = various_levelsets[1:]
    list_of_levelsets   = mesh_obj.get_multiple_levelsets(levelset_values=various_levelsets)
    array_of_errors = mesh_obj.surface_error_by_canal_surfaces(
                                                    list_of_levelsets  = list_of_levelsets )
    min_error, max_error = array_of_errors.min(), array_of_errors.max()
    accuracy_error = np.sum((100*array_of_errors)**2) # multiplied by 100 such that the small errors are still relevant for further time
    absolute_max_error = 100*np.abs([min_error, max_error]).max()
    col_score = collision_score(list_of_levelsets  = list_of_levelsets )
    distance_score, smallest_distance, biggest_distance = _distance_score(list_of_levelsets  = list_of_levelsets )
    score_value = col_score + distance_score + K*(absolute_max_error + accuracy_error) +(max_K - K)*n_levelsets
    return score_value, col_score, distance_score, min_error, max_error, n_levelsets, smallest_distance, biggest_distance

def objective_function(solution):
    r"""
    Objective passed to CMA‑ES.

    Thin wrapper around :meth:`objective_function_information` returning
    only the first element.
    """
    return objective_function_information(solution=solution)[0]

def transform(scaled_solution):
    r"""
    Rescale CMA‑ES decision vector back to physical units.

    Parameters
    ----------
    scaled_solution : np.ndarray
        Decision vector in ``[0, 1]`` produced by CMA‑ES.

    Returns
    -------
    list
        ``[H, n_levelsets]`` where *H* is a distance array and
        *n_levelsets* an integer.
    """
    # a solution is formed by the H array and the number of levelsets
    # recall the number of levelsets is the last element and shall be treated as an integer
    # scaled_solution is the solution from the optimizer, so all the values are between 0 and 1, hence this functions rescales everything to the adequate size

    H = vector_to_segment(scaled_solution[: -1], bounds_H)
    n_levelsets = vector_to_segment(scaled_solution[-1], bounds_n_levelsets)
    return [H, n_levelsets]

def save_solution(solution, folder_to_store, is_good):
    r"""
    Persist a candidate solution to disk.

    Parameters
    ----------
    solution : list
        ``[H, n_levelsets]`` pair(already de‑normalised).
    folder_to_store : str
        Output directory.
    is_good : bool
        ``True`` for feasible solutions, ``False`` for failed ones.

    Notes
    -----
    The file name suffix distinguishes valid ``.npy`` solutions from
    ``_fail.npy`` artefacts.
    """
    if not os.path.exists(folder_to_store):
        os.makedirs(folder_to_store)

    H, n_levelsets = solution
    if is_good:
        final_part = f'K{K}.npy'
    else:
        final_part = f'_failK{K}.npy'
    things_to_store = [H, n_levelsets]
    their_names = ['H','n_levelsets' ]
    for matriz, nombre in zip(things_to_store, their_names):
        file_path = folder_to_store + surface_to_print + '_' + nombre + '_from_top' + final_part
        np.save(file_path, matriz)

def print_information(best_score, col_score, distance_score, smallest_distance,
                      biggest_distance, min_error, max_error, number_of_layers,
                      computing_time):
    r"""
    Pretty‑print optimisation diagnostics as a simple list.
    """
    print(f"Best Score: {best_score:.3e}")
    print(f"Col Score: {col_score:.3e}")
    print(f"Dist Score: {distance_score:.3e}")
    print(f"Smallest Distance: {smallest_distance}")
    print(f"Biggest Distance: {biggest_distance}")
    print(f"Min Error: {min_error}")
    print(f"Max Error: {max_error}")
    print(f"Number of Layers: {number_of_layers}")
    print(f"Computing Time(s): {computing_time}")

def save_logfile(folder_to_store, surface_to_print, K, hyperparams, 
                 initial_measures, final_measures, total_time, avg_iteration_time, 
                 total_iterations):
    r"""
    Save a comprehensive logfile with all optimization information.
    
    Parameters
    ----------
    folder_to_store : str
        Output directory.
    surface_to_print : str
        Name of the surface being optimized.
    K : float
        Weight parameter used in optimization.
    hyperparams : dict
        Dictionary containing all hyperparameters.
    initial_measures : dict
        Dictionary containing initial objective measures.
    final_measures : dict
        Dictionary containing final optimized measures.
    total_time : float
        Total optimization time in seconds.
    avg_iteration_time : float
        Average time per iteration in seconds.
    total_iterations : int
        Total number of iterations performed.
    """
    if not os.path.exists(folder_to_store):
        os.makedirs(folder_to_store)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logfile_path = os.path.join(folder_to_store, f'{surface_to_print}_optimization_logK{K}.txt')
    
    with open(logfile_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"OPTIMIZATION LOG - {surface_to_print}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("="*80 + "\n\n")
        
        # Hyperparameters section
        f.write("HYPERPARAMETERS:\n")
        f.write("-"*80 + "\n")
        for key, value in hyperparams.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.6e}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Initial measures section
        f.write("INITIAL MEASURES:\n")
        f.write("-"*80 + "\n")
        for key, value in initial_measures.items():
            if isinstance(value, (int, np.integer)):
                f.write(f"  {key}: {value}\n")
            elif isinstance(value, (float, np.floating)):
                f.write(f"  {key}: {value:.6e}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Final optimized measures section
        f.write("FINAL OPTIMIZED MEASURES:\n")
        f.write("-"*80 + "\n")
        for key, value in final_measures.items():
            if isinstance(value, (int, np.integer)):
                f.write(f"  {key}: {value}\n")
            elif isinstance(value, (float, np.floating)):
                f.write(f"  {key}: {value:.6e}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Performance metrics section
        f.write("PERFORMANCE METRICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"  Total Iterations: {total_iterations}\n")
        f.write(f"  Total Optimization Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
        f.write(f"  Average Time per Iteration: {avg_iteration_time:.2f} seconds\n")
        f.write("\n")
        
        # Improvement summary
        f.write("IMPROVEMENT SUMMARY:\n")
        f.write("-"*80 + "\n")
        if 'score' in initial_measures and 'score' in final_measures:
            score_improvement = initial_measures['score'] - final_measures['score']
            score_improvement_pct = (score_improvement / initial_measures['score'] * 100) if initial_measures['score'] != 0 else 0
            f.write(f"  Score Improvement: {score_improvement:.6e} ({score_improvement_pct:.2f}%)\n")
        
        if 'min_error' in initial_measures and 'min_error' in final_measures:
            f.write(f"  Min Error Change: {initial_measures['min_error']:.6e} -> {final_measures['min_error']:.6e}\n")
        
        if 'max_error' in initial_measures and 'max_error' in final_measures:
            f.write(f"  Max Error Change: {initial_measures['max_error']:.6e} -> {final_measures['max_error']:.6e}\n")
        
        if 'n_levelsets' in initial_measures and 'n_levelsets' in final_measures:
            f.write(f"  Number of Layers Change: {initial_measures['n_levelsets']} -> {final_measures['n_levelsets']}\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("END OF LOG\n")
        f.write("="*80 + "\n")
    
    print(f'Logfile saved to: {logfile_path}')

print('Functions defined')
#------------------------------------------------------------------------------------------------------------------

# define weights for the score
max_K = 100
K = 90

# load surfaces
root_to_surfaces_folder = './geometries/'
surfaces_to_print = ['reparation', 'deformed_torus', 'dinosaur', 'pumpkin']


# loop over surfaces
for surface_to_print in surfaces_to_print:
    # Set the folder where results will be stored (same as geometry folder)
    folder_to_store = f'{root_to_surfaces_folder}{surface_to_print}/'

    # surface_mesh = trimesh.load_mesh(f'{root_to_surfaces_folder}{surface_to_print}/{surface_to_print}.stl')
    surface_mesh = trimesh.load_mesh(f'{root_to_surfaces_folder}{surface_to_print}/{surface_to_print}.stl')
    reference_surface = trimesh.load_mesh(f'{root_to_surfaces_folder}{surface_to_print}/reference_surface.stl')

    print('Surfaces Loaded')
    print('Begins Mesh Object Creation')
    # create object
    H_init_method = 'default'
    mesh_obj = MeshSlicer.Mesh(surface_mesh, reference_surface, nozzle_half_angle = np.pi * 0.25, H_init_method=H_init_method)
    mesh_obj.save(filename=f'{folder_to_store}{surface_to_print}_initial_mesh_objK{K}.npz')
    print('Mesh Object Created')

    # define the bounds
    bounds_H = [0, mesh_obj.H.max()]
    bounds_n_levelsets = [mesh_obj.min_NLEVELSETS, mesh_obj.max_NLEVELSETS]

    scaled_H = segment_to_vector(mesh_obj.H, bounds_H)
    # alfa = 0.5
    alfa = 0.2
    scaled_n = segment_to_vector((alfa*mesh_obj.max_NLEVELSETS + (1 - alfa)*mesh_obj.min_NLEVELSETS) // 1, bounds_n_levelsets)

    initial_vector = np.concatenate([scaled_H, [scaled_n]])

    seed = 4
    std_dev = 0.001 # initial standard deviation for the CMA-ES
    popsize = 48 # population size of each generation
    xdim = len(initial_vector)
    maxfeval = int(200*xdim) # maximum number of evaluations
    #! the actual final number of total evaluations is maxfeval/popsize
    print(f'Expected number of generations: {maxfeval/popsize}\n')

    # Compute initial measures before optimization
    print('Computing initial measures...')
    initial_solution = transform(initial_vector)
    initial_score, initial_col_score, initial_distance_score, initial_min_error, initial_max_error, initial_n_levelsets, initial_smallest_distance, initial_biggest_distance = objective_function_information(solution=initial_solution)
    print(f'Initial score: {initial_score:.3e}\n')

    # Store hyperparameters for logging
    hyperparams = {
        'K': K,
        'max_K': max_K,
        'seed': seed,
        'popsize': popsize,
        'xdim': xdim,
        'maxfeval': maxfeval,
        'expected_generations': maxfeval/popsize,
        'bounds_H_min': bounds_H[0],
        'bounds_H_max': bounds_H[1],
        'bounds_n_levelsets_min': bounds_n_levelsets[0],
        'bounds_n_levelsets_max': bounds_n_levelsets[1],
        'alfa': alfa,
        'nozzle_half_angle': mesh_obj.nozzle_half_angle,
        'min_thickness': mesh_obj.min_thickness,
        'max_thickness': mesh_obj.max_thickness,
        'surface_to_print': surface_to_print,
        'H_init_method': H_init_method,
        'sigma0': std_dev
    }

    # Store initial measures for logging
    initial_measures = {
        'score': initial_score,
        'collision_score': initial_col_score,
        'distance_score': initial_distance_score,
        'smallest_distance': initial_smallest_distance,
        'biggest_distance': initial_biggest_distance,
        'min_error': initial_min_error,
        'max_error': initial_max_error,
        'n_levelsets': initial_n_levelsets
    }

    # Start total optimization timer
    start_time = clock()
    print(f'Optimization process starts\n')
    # Initializations.
    best_solution = None # best solution found
    best_score = math.inf # score of the best solution
    iteration_times = []  # track time for each iteration
    # Apply CMA-ES algorithm for solution search.

    es = cma.CMAEvolutionStrategy(initial_vector, std_dev,
                                inopts={'bounds': [0, 1],
                                        'seed':seed,
                                        'maxiter':1e9,
                                        'maxfevals':maxfeval,
                                        'popsize':popsize})
    iteracion = -1
    max_error = 1
    while not es.stop():

        # Build population.
        scaled_solutions = es.ask()
        iteracion+=1
        print(f'iteracion = {iteracion},  best_score =  {best_score} \n')
        tic = clock()
        # Transform the scaled values of the variables to the real values.
        real_solutions=[transform(i) for i in scaled_solutions]
        try:
            print('Inside the parallel')
            list_scores = Parallel(n_jobs=-1, timeout = 100)(delayed(objective_function)(sol) for sol in real_solutions)
        except:
            list_scores = []
            print('\t dentro del primer except\n')
            for j, sol in enumerate(real_solutions):
                try:
                    score = objective_function(sol)
                    list_scores.append(score)
                    print(f'\t len of list_scores = {len(list_scores)}\n')
                except:
                    print('\t dentro del segundo except')
                    # this saves the best one so far
                    save_solution(solution = best_solution, folder_to_store = folder_to_store, is_good = True)
                    # now let's save the one that fails for retrieving purposes
                    save_solution(solution = sol, folder_to_store = folder_to_store, is_good = False)
                    print('Something went wrong\n')
                    sys.exit(0) # exiting the program to make sure that we are not producing any more matrices than needed.

        es.tell(scaled_solutions, list_scores)
        # Update best solution found.
        solution = transform(es.result.xbest)
        score_value, col_score, distance_score, min_error, max_error, n_levelsets, smallest_distance, biggest_distance = objective_function_information(solution=solution)

        if score_value < best_score:
            best_score=score_value
            best_solution=solution
            tac = clock()
            iteration_time = tac - tic
            iteration_times.append(iteration_time)
            print_information(best_score, col_score, distance_score, smallest_distance,
                            biggest_distance, min_error, max_error, n_levelsets,
                            iteration_time)

    save_solution(solution = best_solution,
                folder_to_store = folder_to_store,
                is_good = True)

    print(f'Process finished!!\n')
    # Report total elapsed time
    total_time = clock() - start_time
    print(f'Total optimization time: {total_time:.2f} seconds')

    # Compute final measures
    final_score, final_col_score, final_distance_score, final_min_error, final_max_error, final_n_levelsets, final_smallest_distance, final_biggest_distance = objective_function_information(solution=best_solution)

    # Store final measures for logging
    final_measures = {
        'score': final_score,
        'collision_score': final_col_score,
        'distance_score': final_distance_score,
        'smallest_distance': final_smallest_distance,
        'biggest_distance': final_biggest_distance,
        'min_error': final_min_error,
        'max_error': final_max_error,
        'n_levelsets': final_n_levelsets
    }

    # Calculate average iteration time
    avg_iteration_time = sum(iteration_times) / len(iteration_times) if iteration_times else 0
    total_iterations = iteracion + 1

    # Save comprehensive logfile
    save_logfile(
        folder_to_store=folder_to_store,
        surface_to_print=surface_to_print,
        K=K,
        hyperparams=hyperparams,
        initial_measures=initial_measures,
        final_measures=final_measures,
        total_time=total_time,
        avg_iteration_time=avg_iteration_time,
        total_iterations=total_iterations
    )
