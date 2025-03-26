
import copy
import numpy
import numpy as np
import random


def branching_dp_dynamics(walkers, weights, min_weight=0.2, max_weight=2.0):
    """
    Perform pair branching population control with dynamic walker count
    and target population

    Args:
        walkers (list): List of walker configurations
        weights (np.ndarray): Corresponding walker weights
        target_population (int, optional): Desired number of walkers after branching
        min_weight (float): Minimum weight threshold
        max_weight (float): Maximum weight threshold

    Returns:
        tuple: Updated walkers and weights after pair branching
    """
    # Use default target population if not specified
    target_population = len(walkers)

    # Create lists to store new walkers and weights
    new_walkers = []
    new_weights = []

    # Create a copy of walkers and weights to modify
    mod_walkers = walkers.copy()
    mod_weights = weights.copy()

    # Sort indices by absolute weight
    sort_indices = np.argsort(np.abs(mod_weights))
    indices_upper = numpy.where(mod_weights > max_weight)[0]
    indices_lower = numpy.where(mod_weights < min_weight)[0]

    s, e = 0, len(mod_walkers) - 1
    while s <= e and len(new_walkers) < target_population:
        # Check if pair needs branching
        if (np.abs(mod_weights[sort_indices[s]]) < min_weight or
            np.abs(mod_weights[sort_indices[e]]) > max_weight):

            # Compute total weight of the pair
            wab = (np.abs(mod_weights[sort_indices[s]]) +
                   np.abs(mod_weights[sort_indices[e]]))

            # Randomly decide which walker to clone
            r = np.random.rand()
            if r < np.abs(mod_weights[sort_indices[e]]) / wab:
                # Clone large weight walker
                new_walkers.append(mod_walkers[sort_indices[s]])
                new_weights.append(0.5 * wab)

                # Modify original walker
                mod_weights[sort_indices[e]] = 0.5 * wab
            else:
                # Clone small weight walker
                new_walkers.append(mod_walkers[sort_indices[e]])
                new_weights.append(0.5 * wab)

                # Modify original walker
                mod_weights[sort_indices[s]] = 0.5 * wab

            s += 1
            e -= 1
        else:
            # Add remaining walkers if not reached target population
            if len(new_walkers) < target_population:
                new_walkers.append(mod_walkers[sort_indices[e]])
                new_weights.append(mod_weights[sort_indices[e]])
                e -= 1

    # If still not reached target population, pad with existing walkers
    while len(new_walkers) < target_population:
        # Randomly select from original walkers
        idx = np.random.randint(0, len(mod_walkers))
        new_walkers.append(mod_walkers[idx])
        new_weights.append(mod_weights[idx])

    return new_walkers, np.array(new_weights)


def branching_dp_constant(walkers, weights, min_weight=0.2, max_weight=4.0):
    """
    Perform pair branching population control while maintaining constant walker count.

    Args:
        walkers (list): List of walker configurations
        weights (np.ndarray): Corresponding walker weights
        min_weight (float): Minimum weight threshold
        max_weight (float): Maximum weight threshold

    Returns:
        tuple: Updated walkers and weights after pair branching
    """
    # Total number of walkers
    nwalkers = len(walkers)

    # Create a copy of walkers and weights to modify
    new_walkers = walkers.copy()
    new_weights = weights.copy()

    # Sort indices by absolute weight
    sort_indices = np.argsort(np.abs(weights))

    # Pair branching algorithm
    s, e = 0, nwalkers - 1
    while s < e:
        # Check if pair needs branching
        if (np.abs(new_weights[sort_indices[s]]) < min_weight or
            np.abs(new_weights[sort_indices[e]]) > max_weight):

            # Compute total weight of the pair
            wab = (np.abs(new_weights[sort_indices[s]]) +
                   np.abs(new_weights[sort_indices[e]]))

            # Randomly decide which walker to clone
            r = np.random.rand()
            if r < np.abs(new_weights[sort_indices[e]]) / wab:
                # Clone large weight walker
                new_weights[sort_indices[e]] = 0.5 * wab
                new_walkers[sort_indices[e]] = new_walkers[sort_indices[s]]
            else:
                # Clone small weight walker
                new_weights[sort_indices[s]] = 0.5 * wab
                new_walkers[sort_indices[s]] = new_walkers[sort_indices[e]]

            s += 1
            e -= 1
        else:
            # No more pair branching needed
            break

    return new_walkers, new_weights


def branching_dp0(walkers, weights, min_weight=0.1, max_weight=2.0):
    """
    Perform branching population control on walkers with dynamic programming

    Args:
        walkers (list): List of walker configurations
        weights (np.ndarray): Corresponding walker weights
        min_weight (float): Minimum weight threshold
        max_weight (float): Maximum weight threshold

    Returns:
        tuple: Updated walkers and weights after pair branching
    """
    assert len(walkers) == len(weights)

    # Create a structured array to track walker information
    walker_info = np.zeros(len(walkers), dtype=[
        ('weight', float),   # Absolute walker weight
        ('status', int),     # Walker status (0: die, 1: live, 2: cloned)
        ('org_idx', int),  # Original walker index
        ('final_idx', int)      # Final walker destination index
    ])

    # Initialize walker info
    walker_info['weight'] = np.abs(weights)
    walker_info['status'] = 1  # All walkers initially live
    walker_info['org_idx'] = np.arange(len(walkers))
    walker_info['final_idx'] = np.arange(len(walkers))

    # Sort walkers by absolute weight
    sort_indices = np.argsort(walker_info['weight'])
    sort_walker_info = walker_info[sort_indices]

    # Pair branching algorithm
    start, end = 0, len(sort_walker_info) - 1
    while start < end:
        # Check if pair needs branching
        if (sort_walker_info[start]['weight'] < min_weight or
            sort_walker_info[end]['weight'] > max_weight):

            # Compute total weight of the pair
            w_se = (sort_walker_info[start]['weight'] +
                   sort_walker_info[end]['weight'])

            # Randomly decide which walker to clone
            p = np.random.rand()
            if p < sort_walker_info[end]['weight'] / w_se:
                # Clone large weight walker
                sort_walker_info[end]['weight'] = 0.5 * w_se
                sort_walker_info[end]['status'] = 2  # cloned
                sort_walker_info[end]['final_idx'] = sort_walker_info[start]['org_idx']

                # Kill small weight walker
                sort_walker_info[start]['weight'] = 0.0
                sort_walker_info[start]['status'] = 0  # Die
                sort_walker_info[start]['final_idx'] = sort_walker_info[end]['org_idx']
            else:
                # Clone small weight walker
                sort_walker_info[start]['weight'] = 0.5 * w_se
                sort_walker_info[start]['status'] = 2  # cloned
                sort_walker_info[start]['final_idx'] = sort_walker_info[end]['org_idx']

                # Kill large weight walker
                sort_walker_info[end]['weight'] = 0.0
                sort_walker_info[end]['status'] = 0  # Die
                sort_walker_info[end]['final_idx'] = sort_walker_info[start]['org_idx']

            start += 1
            end -= 1
        else:
            # No more pair branching needed
            break

    # Reconstruct walkers and weights
    new_walkers = []
    new_weights = []

    for info in sort_walker_info:
        #if info['status'] > 0:  # Live or cloned walker
        for _ in range(info['status']):  # Live or cloned walker
            # Find the original walker
            original_walker = walkers[info['org_idx']]
            new_walkers.append(original_walker)
            new_weights.append(info['weight'])

    assert len(new_walkers) == len(new_weights)
    return new_walkers, np.array(new_weights)


#branching_dp = branching_dp_dynamics
#branching_dp = branching_dp_constant
branching_dp = branching_dp0


def branching_control(walkers, bound=2.0):
    """
    Perform branching (residual resampling) that preserves the total number of walkers.

    walkers class, which contains
      phiw    : np.ndarray
                Array of walker wavefunctions with shape [N, nao, no].
      weights : np.ndarray
                Array of walker weights with shape [N].
    bounds:
       a list of two floats: upper and lower bounds for the weights

    Returns: walkers with new phiw and weights
    """
    upper_bound = bound
    lower_bound = 1.0 / bound

    nwalkers = walkers.nwalkers
    total_weight = numpy.sum(walkers.weights)
    target = walkers.total_weight0 / nwalkers # target weight per walker

    indices_upper = numpy.where(walkers.weights > upper_bound)[0]
    indices_lower = numpy.where(walkers.weights < lower_bound)[0]
    indices_rest = numpy.where((arr >= lower_bound) & (arr <= upper_bound))[0]

    # For each walker, compute the number of copies (integer part)
    # and the fractional remainder.
    integer_copies = numpy.floor(walkers.weights / target).astype(int)
    residual = walkers.weights / target - integer_copies
    total_integer = numpy.sum(integer_copies)

    new_phi = []
    new_weights = []

    # First, add the integer number of copies for each walker
    count = 0
    for i in indices_upper: # range(nwalkers):
        for _ in range(integer_copies[i]):
            new_phi.append(numpy.copy(walkers.phiw[i]))
            new_weights.append(target)
            # if self.boson_phiw is not None
        count += integer_copies[i]
        if count > nwalkers: break

    # Calculate how many extra walkers we need to reach nwalkers.
    extra_needed = nwalkers - count

    if extra_needed > 0:
        # If the residuals sum to > 0, normalize them to get selection probabilities.
        if numpy.sum(residual) > 0:
            norm_residual = residual / numpy.sum(residual)
        else:
            norm_residual = numpy.ones(nwalkers) / nwalkers
        extra_indices = numpy.random.choice(numpy.arange(nwalkers), size=extra_needed, p=norm_residual)
        for i in extra_indices:
            new_phi.append(numpy.copy(walkers.phiw[i]))
            new_weights.append(target)
            # if self.boson_phiw is not None

    elif extra_needed < 0:
        # If too many walkers were generated (rare), randomly trim the list.
        new_phi = new_phi[:nwalkers]
        new_weights = new_weights[:nwalkers]

    return new_phi, numpy.array(new_weights)


def comb_resampling(phiw, weights):
    """
    Perform comb (systematic) resampling to control the walker population.

    Parameters:
        phiw    : np.ndarray of shape [N, ...]
                  Array of walker configurations/wavefunctions.
        weights : np.ndarray of shape [N]
                  Array of walker weights.

    Returns:
        new_phiw    : np.ndarray of shape [N, ...]
                      Array of resampled walker configurations.
        new_weights : np.ndarray of shape [N]
                      Array of uniform weights (total_weight / N).
    """
    N = len(weights)
    total_weight = np.sum(weights)
    norm_weights = weights / total_weight
    cumulative_sum = np.cumsum(norm_weights)

    # Generate N equally spaced markers with a random starting offset.
    offset = np.random.uniform(0, 1.0 / N)
    markers = offset + np.arange(N) / N

    new_phiw = []
    for marker in markers:
        # np.searchsorted finds the index where marker should be inserted
        # to maintain the order, which selects the walker.
        idx = np.searchsorted(cumulative_sum, marker)
        new_phiw.append(copy.copy(phiw[idx]))

    new_weights = np.full(N, total_weight / N)
    return new_phiw, new_weights


def stochastic_reconfiguration(phiw, weights):
    """
    Perform systematic resampling to preserve the total number of walkers.

    Parameters:
      phiw    : np.ndarray of shape [N, nao, nao]
      weights : np.ndarray of shape [N]

    Returns:
      new_phiw    : np.ndarray of shape [N, nao, nao]
      new_weights : np.ndarray of shape [N] (each is total_weight/N)
    """
    N = len(weights)
    total_weight = np.sum(weights)
    norm_weights = weights / total_weight  # Normalize weights so they sum to 1
    cumulative_sum = np.cumsum(norm_weights)

    # Generate N equally spaced positions in [0,1) with a random starting offset.
    start = np.random.uniform(0, 1/N)

    positions = start + np.arange(N) / N

    new_phiw = []
    new_weights = []

    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            new_phiw.append(copy.copy(phiw[j]))
            new_weights.append(total_weight / N)
            i += 1
        else:
            j += 1

    new_weights = np.array(new_weights)

    return new_phiw, new_weights


# method dict
control_func_dict = {
    "branching": branching_dp0,
    "reconfiguration": stochastic_reconfiguration,
    "comb": comb_resampling,
    #"hybrid": hybrid_control,
}


def population_control_factory(walkers, method="branching"):

    if method in control_func_dict:
        control_func = control_func_dict[method]

        # pack the walker WF (phiwa, phib, boson_phiw) in one list
        packed_walkers = walkers._pack_walkers()
        new_walkers, weights = control_func(packed_walkers, walkers.weights)

        # updte walker WF and weights
        walkers._unpack_walkers(new_walkers)
        walkers.weights = weights



if __name__ == '__main__':
    r"""
    """
    nwalkers = 500
    # generate a random walkers with weights to test the control methods
    from openms.qmc.generic_walkers import make_walkers
