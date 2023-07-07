#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np
import bisect
from libc.math cimport exp
from libc.math cimport log
# from libc.stdlib cimport rand, RAND_MAX, srand

DTYPEf = np.float64
DTYPEi = np.int64
ctypedef np.float64_t DTYPEf_t
ctypedef np.int64_t DTYPEi_t

def log_likelihood_of_events(np.ndarray[DTYPEf_t, ndim=1] base_rates,
                             np.ndarray[DTYPEf_t, ndim=3] impact_coefficients,
                             np.ndarray[DTYPEf_t, ndim=3] decay_coefficients,
                             int number_of_event_types,
                             int number_of_states,
                             np.ndarray[DTYPEf_t, ndim=1] times,
                             np.ndarray[DTYPEi_t, ndim=1] events,
                             np.ndarray[DTYPEi_t, ndim=1] states,
                             double time_start,
                             double time_end):
    """
    Computes the log-likelihood of events.
    :param parameters: [array] 1-D array of parameters (base rates, impact coefficients, decay coefficients)
    :param times:
    :param events:
    :param states:
    :param time_start:
    :param time_end:
    :return:
    """
    cdef int index_start
    cdef np.ndarray[DTYPEf_t, ndim=3] partial_sums = np.zeros((number_of_event_types, number_of_states, number_of_event_types), dtype=DTYPEf)
    cdef int n, event, state, e, e1, x, e2, index_end
    cdef double time, previous_time, intensity_of_the_event
    cdef double alpha, beta, ratio, time_increment, time_increment_2
    # events at and before this time are treated as an initial condition
    index_start = bisect.bisect_right(times, time_start)
    # saving the ratios of impact and decay coefficients will be useful
    cdef np.ndarray[DTYPEf_t, ndim=3] impact_decay_ratios = np.zeros((number_of_event_types, number_of_states, number_of_event_types), dtype=DTYPEf)
    for e1 in range(number_of_event_types):
        for x in range(number_of_states):
            for e2 in range(number_of_event_types):
                alpha = impact_coefficients[e1, x, e2]
                beta = decay_coefficients[e1, x, e2]
                impact_decay_ratios[e1, x, e2] = alpha / beta
    '''Initialise the partial sums S_{e'x'e} that will allow us to compute the intensity recursively;
    and initialise the log-likelihood taking into account the initial condition'''
    cdef double log_likelihood = 0
    for e in range(number_of_event_types):
        log_likelihood += base_rates[e]
    log_likelihood *= - (time_end - time_start)
    for n in range(index_start):
        time = times[n]
        event = events[n]
        state = states[n]
        time_increment = time_start - time
        time_increment_2 = time_end - time
        for e in range(number_of_event_types):
                beta = decay_coefficients[event, state, e]
                ratio = impact_decay_ratios[event, state, e]
                partial_sums[event, state, e] += exp(-beta * time_increment)
                log_likelihood -= ratio * (exp(-beta * time_increment)- exp(-beta * time_increment_2))
    # By doing so, multiplying the partial sums by the impact coefficients needs to be done only once
    for event in range(number_of_event_types):
        for state in range(number_of_states):
            for e in range(number_of_event_types):
                partial_sums[event, state, e] = partial_sums[event, state, e] * impact_coefficients[event, state, e]
    'Go through event times and update likelihood'
    previous_time = time_start
    index_end = times.shape[0]
    for n in range(index_start, index_end):
        time = times[n]
        event = events[n]
        state = states[n]
        'Update the partial sums: time-decay effect'
        time_increment = time - previous_time
        for e1 in range(number_of_event_types):
                for x in range(number_of_states):
                    for e2 in range(number_of_event_types):
                        beta = decay_coefficients[e1, x, e2]
                        partial_sums[e1, x, e2] *= exp(-beta * time_increment)
        'Update the first term of the log-likelihood (l_{+})'
        intensity_of_the_event = base_rates[event]
        for e in range(number_of_event_types):
                for x in range(number_of_states):
                    intensity_of_the_event += partial_sums[e, x, event]
        log_likelihood += log(intensity_of_the_event)
        'Update the partial sums: impact of the new event'
        for e in range(number_of_event_types):
                alpha = impact_coefficients[event, state, e]
                partial_sums[event, state, e] += alpha
        previous_time = time
        'Compute the second term of the likelihood (l_{-})'
        time_increment = time_end - time
        for e in range(number_of_event_types):
                beta = decay_coefficients[event, state, e]
                ratio = impact_decay_ratios[event, state, e]
                log_likelihood -= ratio * (1 - exp(-beta * time_increment))
    return log_likelihood

def log_likelihood_of_events_partial(int event_type,
                             double base_rate,
                             np.ndarray[DTYPEf_t, ndim=2] impact_coefficients,
                             np.ndarray[DTYPEf_t, ndim=2] decay_coefficients,
                             int number_of_event_types,
                             int number_of_states,
                             np.ndarray[DTYPEf_t, ndim=1] times,
                             np.ndarray[DTYPEi_t, ndim=1] events,
                             np.ndarray[DTYPEi_t, ndim=1] states,
                             double time_start,
                             double time_end):
    """
    Computes the log-likelihood associated to a single event type (the full log-likelihood is the sum of the partial log-likelihoods).
    :param parameters: [array] 1-D array of parameters (base rate, impact coefficients, decay coefficients)
    :param times:
    :param events:
    :param states:
    :param time_start:
    :param time_end:
    :return:
    """
    cdef int index_start
    cdef np.ndarray[DTYPEf_t, ndim=2] partial_sums = np.zeros((number_of_event_types, number_of_states), dtype=DTYPEf)
    cdef int n, event, state, e, e1, x, index_end
    cdef double time, previous_time, intensity_of_the_event
    cdef double alpha, beta, ratio, time_increment, time_increment_2
    # events at and before this time are treated as an initial condition
    index_start = bisect.bisect_right(times, time_start)
    # saving the ratios of impact and decay coefficients will be useful
    cdef np.ndarray[DTYPEf_t, ndim=2] impact_decay_ratios = np.zeros((number_of_event_types, number_of_states), dtype=DTYPEf)
    for e1 in range(number_of_event_types):
        for x in range(number_of_states):
            alpha = impact_coefficients[e1, x]
            beta = decay_coefficients[e1, x]
            impact_decay_ratios[e1, x] = alpha / beta
    '''Initialise the partial sums S_{e'x'} that will allow us to compute the intensity recursively;
    and initialise the log-likelihood taking into account the initial condition'''
    cdef double log_likelihood = 0
    log_likelihood += base_rate
    log_likelihood *= - (time_end - time_start)
    for n in range(index_start):
        time = times[n]
        event = events[n]
        state = states[n]
        time_increment = time_start - time
        time_increment_2 = time_end - time
        beta = decay_coefficients[event, state]
        ratio = impact_decay_ratios[event, state]
        partial_sums[event, state] += exp(-beta * time_increment)
        log_likelihood -= ratio * (exp(-beta * time_increment)- exp(-beta * time_increment_2))
    # By doing so, multiplying the partial sums by the impact coefficients needs to be done only once
    for event in range(number_of_event_types):
        for state in range(number_of_states):
            partial_sums[event, state] = partial_sums[event, state] * impact_coefficients[event, state]
    'Go through event times and update likelihood'
    previous_time = time_start
    index_end = times.shape[0]
    for n in range(index_start, index_end):
        time = times[n]
        event = events[n]
        state = states[n]
        'Update the partial sums: time-decay effect'
        time_increment = time - previous_time
        for e1 in range(number_of_event_types):
                for x in range(number_of_states):
                    beta = decay_coefficients[e1, x]
                    partial_sums[e1, x] *= exp(-beta * time_increment)
        'Update the first term of the log-likelihood (l_{+})'
        if event == event_type:
            intensity_of_the_event = base_rate
            for e in range(number_of_event_types):
                    for x in range(number_of_states):
                        intensity_of_the_event += partial_sums[e, x]
            log_likelihood += log(intensity_of_the_event)
        'Update the partial sums: impact of the new event'
        alpha = impact_coefficients[event, state]
        partial_sums[event, state] += alpha
        previous_time = time
        'Compute the second term of the likelihood (l_{-})'
        time_increment = time_end - time
        beta = decay_coefficients[event, state]
        ratio = impact_decay_ratios[event, state]
        log_likelihood -= ratio * (1 - exp(-beta * time_increment))
    return log_likelihood

def gradient(np.ndarray[DTYPEf_t, ndim=1] base_rates,
             np.ndarray[DTYPEf_t, ndim=3] impact_coefficients,
             np.ndarray[DTYPEf_t, ndim=3] decay_coefficients,
             int number_of_event_types,
             int number_of_states,
             np.ndarray[DTYPEf_t, ndim=1] times,
             np.ndarray[DTYPEi_t, ndim=1] events,
             np.ndarray[DTYPEi_t, ndim=1] states,
             double time_start,
             double time_end):
    """
    Computes the gradient of the log-likelihood of events.
    :param parameters:
    :param times:
    :param events:
    :param states:
    :param time_start:
    :param time_end:
    :return:
    """
    cdef int index_start
    cdef int n, event, state, e, e1, x, e2, index_end
    cdef double time, previous_time
    cdef double alpha, beta, ratio, time_increment, time_increment_2, sample_duration, a, b, c, decay, intensity_of_the_event
    # initialise the gradient vectors
    sample_duration = time_end - time_start
    cdef np.ndarray[DTYPEf_t, ndim=1] gradient_base_rates = np.zeros(number_of_event_types)
    for e in range(number_of_event_types):
        gradient_base_rates[e] = - sample_duration
    cdef np.ndarray[DTYPEf_t, ndim=3] gradient_impact_coefficients =\
        np.zeros((number_of_event_types, number_of_states, number_of_event_types))
    cdef np.ndarray[DTYPEf_t, ndim=3] gradient_decay_coefficients =\
        np.zeros((number_of_event_types, number_of_states, number_of_event_types))
    # events at and before this time are treated as an initial condition
    index_start = bisect.bisect_right(times, time_start)
    # compute the ratios impact/decay coefficients once as they will be used a lot
    cdef np.ndarray[DTYPEf_t, ndim=3] impact_decay_ratios = np.zeros((number_of_event_types, number_of_states, number_of_event_types), dtype=DTYPEf)
    for e1 in range(number_of_event_types):
        for x in range(number_of_states):
            for e2 in range(number_of_event_types):
                alpha = impact_coefficients[e1, x, e2]
                beta = decay_coefficients[e1, x, e2]
                impact_decay_ratios[e1, x, e2] = alpha / beta
    '''Initialise the partial sums S_{e'x'e} and S^{(1)}_{e'x'e}
    that will allow us to compute the intensity and the gradient recursively;
    and compute contribution of initial condition on gradient.'''
    cdef np.ndarray[DTYPEf_t, ndim=3] partial_sums = np.zeros((number_of_event_types, number_of_states, number_of_event_types), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=3] partial_sums_1 = np.zeros((number_of_event_types, number_of_states, number_of_event_types), dtype=DTYPEf)
    for n in range(index_start):
        time = times[n]
        event = events[n]
        state = states[n]
        time_increment = time_start - time
        time_increment_2 = time_end - time
        for e in range(number_of_event_types):
            beta = decay_coefficients[event, state, e]
            ratio = impact_decay_ratios[event, state, e]
            a = exp(- beta * time_increment)
            partial_sums[event, state, e] += a
            partial_sums_1[event, state, e] += a * time_increment
            b = exp(- beta * time_increment_2)
            gradient_impact_coefficients[event, state, e] -= (a - b) / beta
            gradient_decay_coefficients[event, state, e] -= ratio * (time_increment_2*b - time_increment*a)
            gradient_decay_coefficients[event, state, e] -= - ratio * (a - b) / beta
    # By doing so, we multiply by the impact coefficients only once
    for event in range(number_of_event_types):
        for state in range(number_of_states):
            for e in range(number_of_event_types):
                alpha = impact_coefficients[event, state, e]
                partial_sums[event, state, e] = partial_sums[event, state, e] * alpha
                partial_sums_1[event, state, e] = partial_sums_1[event, state, e] * alpha
    'Go through event times and update likelihood'
    previous_time = time_start
    index_end = times.shape[0]
    for n in range(index_start, index_end):
        time = times[n]
        event = events[n]
        state = states[n]
        'Update the partial sums: time-decay effect'
        time_increment = time - previous_time
        for e1 in range(number_of_event_types):
            for x in range(number_of_states):
                for e2 in range(number_of_event_types):
                    beta = decay_coefficients[e1, x, e2]
                    partial_sums_1[e1, x, e2] += time_increment * partial_sums[e1, x, e2]
                    decay = exp(-beta * time_increment)
                    partial_sums_1[e1, x, e2] *= decay
                    partial_sums[e1, x, e2] *= decay
        'Update the gradient'
        intensity_of_the_event = base_rates[event]
        for e in range(number_of_event_types):
            for x in range(number_of_states):
                intensity_of_the_event += partial_sums[e, x, event]
        gradient_base_rates[event] += 1 / intensity_of_the_event
        for e in range(number_of_event_types):
            for x in range(number_of_states):
                alpha = impact_coefficients[e, x, event]
                gradient_impact_coefficients[e, x, event] += (partial_sums[e, x, event] / alpha) / intensity_of_the_event
                gradient_decay_coefficients[e, x, event] -= partial_sums_1[e, x, event] / intensity_of_the_event
        'Update the partial sums: impact of the new event'
        for e in range(number_of_event_types):
            alpha = impact_coefficients[event, state, e]
            partial_sums[event, state, e] += alpha
        previous_time = time
        'Subtract gradient of second term of log-likelihood'
        time_increment = time_end - time
        for e in range(number_of_event_types):
            beta = decay_coefficients[event, state, e]
            ratio = impact_decay_ratios[event, state, e]
            c = 1 - exp(-beta * time_increment)
            gradient_impact_coefficients[event, state, e] -= c / beta
            gradient_decay_coefficients[event, state, e] -= ratio * time_increment * (1 - c)
            gradient_decay_coefficients[event, state, e] -= - ratio * c / beta
    'Return the result, i.e., the gradient'
    return gradient_base_rates, gradient_impact_coefficients, gradient_decay_coefficients

def gradient_partial(int event_type,
             double base_rate,
             np.ndarray[DTYPEf_t, ndim=2] impact_coefficients,
             np.ndarray[DTYPEf_t, ndim=2] decay_coefficients,
             int number_of_event_types,
             int number_of_states,
             np.ndarray[DTYPEf_t, ndim=1] times,
             np.ndarray[DTYPEi_t, ndim=1] events,
             np.ndarray[DTYPEi_t, ndim=1] states,
             double time_start,
             double time_end):
    """
    Computes the gradient of the log-likelihood of events, only with respect to the parameters that contribute to the intensity of the given event type.
    :param parameters:
    :param times:
    :param events:
    :param states:
    :param time_start:
    :param time_end:
    :return:
    """
    assert base_rate > 0.
    cdef int index_start
    cdef int n, event, state, e, e1, x, e2, index_end
    cdef double time, previous_time
    cdef double alpha, beta, ratio, time_increment, time_increment_2, sample_duration, a, b, c, decay, intensity_of_the_event
    # initialise the gradient vectors
    sample_duration = time_end - time_start
    cdef double gradient_base_rate
    gradient_base_rate = - sample_duration
    cdef np.ndarray[DTYPEf_t, ndim=2] gradient_impact_coefficients =\
        np.zeros((number_of_event_types, number_of_states))
    cdef np.ndarray[DTYPEf_t, ndim=2] gradient_decay_coefficients =\
        np.zeros((number_of_event_types, number_of_states))
    # events at and before this time are treated as an initial condition
    index_start = bisect.bisect_right(times, time_start)
    # compute the ratios impact/decay coefficients once as they will be used a lot
    cdef np.ndarray[DTYPEf_t, ndim=2] impact_decay_ratios = np.zeros((number_of_event_types, number_of_states), dtype=DTYPEf)
    for e1 in range(number_of_event_types):
        for x in range(number_of_states):
            alpha = impact_coefficients[e1, x]
            beta = decay_coefficients[e1, x]
            impact_decay_ratios[e1, x] = alpha / beta
    '''Initialise the partial sums S_{e'x'} and S^{(1)}_{e'x'}
    that will allow us to compute the intensity and the gradient recursively;
    and compute contribution of initial condition on gradient.'''
    cdef np.ndarray[DTYPEf_t, ndim=2] partial_sums = np.zeros((number_of_event_types, number_of_states), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] partial_sums_1 = np.zeros((number_of_event_types, number_of_states), dtype=DTYPEf)
    for n in range(index_start):
        time = times[n]
        event = events[n]
        state = states[n]
        time_increment = time_start - time
        time_increment_2 = time_end - time
        beta = decay_coefficients[event, state]
        ratio = impact_decay_ratios[event, state]
        a = exp(- beta * time_increment)
        partial_sums[event, state] += a
        partial_sums_1[event, state] += a * time_increment
        b = exp(- beta * time_increment_2)
        gradient_impact_coefficients[event, state] -= (a - b) / beta
        gradient_decay_coefficients[event, state] -= ratio * (time_increment_2*b - time_increment*a)
        gradient_decay_coefficients[event, state] -= - ratio * (a - b) / beta
    # By doing so, we multiply by the impact coefficients only once
    for event in range(number_of_event_types):
        for state in range(number_of_states):
            alpha = impact_coefficients[event, state]
            partial_sums[event, state] = partial_sums[event, state] * alpha
            partial_sums_1[event, state] = partial_sums_1[event, state] * alpha
    'Go through event times and update likelihood'
    previous_time = time_start
    index_end = times.shape[0]
    for n in range(index_start, index_end):
        time = times[n]
        event = events[n]
        state = states[n]
        'Update the partial sums: time-decay effect'
        time_increment = time - previous_time
        for e1 in range(number_of_event_types):
            for x in range(number_of_states):
                beta = decay_coefficients[e1, x]
                partial_sums_1[e1, x] += time_increment * partial_sums[e1, x]
                decay = exp(-beta * time_increment)
                partial_sums_1[e1, x] *= decay
                partial_sums[e1, x] *= decay
        'Update the gradient'
        if event == event_type:
            intensity_of_the_event = base_rate
            for e in range(number_of_event_types):
                for x in range(number_of_states):
                    intensity_of_the_event += partial_sums[e, x]
            gradient_base_rate += 1 / intensity_of_the_event
            for e in range(number_of_event_types):
                for x in range(number_of_states):
                    alpha = impact_coefficients[e, x]
                    if alpha > 0.:
                        gradient_impact_coefficients[e, x] += (partial_sums[e, x] / alpha) / intensity_of_the_event
                    gradient_decay_coefficients[e, x] -= partial_sums_1[e, x] / intensity_of_the_event
        'Update the partial sums: impact of the new event'
        alpha = impact_coefficients[event, state]
        partial_sums[event, state] += alpha
        previous_time = time
        'Subtract gradient of second term of log-likelihood'
        time_increment = time_end - time
        beta = decay_coefficients[event, state]
        ratio = impact_decay_ratios[event, state]
        c = 1 - exp(-beta * time_increment)
        gradient_impact_coefficients[event, state] -= c / beta
        gradient_decay_coefficients[event, state] -= ratio * time_increment * (1 - c)
        gradient_decay_coefficients[event, state] -= - ratio * c / beta
    'Return the result, i.e., the gradient'
    return gradient_base_rate, gradient_impact_coefficients, gradient_decay_coefficients

def simulate(int number_of_event_types,
             int number_of_states,
             np.ndarray[DTYPEf_t, ndim=1] base_rates,
             np.ndarray[DTYPEf_t, ndim=3] impact_coefficients,
             np.ndarray[DTYPEf_t, ndim=3] decay_coefficients,
             np.ndarray[DTYPEf_t, ndim=3] transition_probabilities,
             np.ndarray[DTYPEf_t, ndim=1] initial_condition_times,
             np.ndarray[DTYPEi_t, ndim=1] initial_condition_events,
             np.ndarray[DTYPEi_t, ndim=1] initial_condition_states,
             np.ndarray[DTYPEf_t, ndim=3] initial_partial_sums,
             int initial_state,
             double time_start,
             double time_end,
             int max_number_of_events):
    """
    Simulates a state-dependent Hawkes process with exponential kernels.
    :param number_of_event_types:
    :param number_of_states:
    :return:
    """
    '''Initialise the partial sums S_{e',x',e} that will allow us to compute the intensity recursively'''
    cdef int number_of_initial_events = initial_condition_times.shape[0]
    cdef np.ndarray[DTYPEf_t, ndim=3] partial_sums = np.zeros((number_of_event_types, number_of_states, number_of_event_types), dtype=DTYPEf)
    cdef double time, alpha, beta
    cdef int n, event, state, e, e1, e2, x
    for n in range(number_of_initial_events):
        time = initial_condition_times[n]
        event = initial_condition_events[n]
        state = initial_condition_states[n]
        for e in range(number_of_event_types):
            alpha = impact_coefficients[event, state, e]
            beta = decay_coefficients[event, state, e]
            partial_sums[event, state, e] += alpha * exp(-beta * (time_start - time))
    'Users can also pass directly the initial_partial_sums'
    for e1 in range(number_of_event_types):
        for x in range(number_of_states):
            for e2 in range(number_of_event_types):
                partial_sums[e1, x, e2] += initial_partial_sums[e1, x, e2]
    
    'Compute the initial intensities of events and the total intensity'
    cdef np.ndarray[DTYPEf_t, ndim=1] intensities = np.zeros(number_of_event_types, dtype=DTYPEf)
    cdef double intensity_max = 0
    for e2 in range(number_of_event_types):
        intensities[e2] = base_rates[e2]
        for e1 in range(number_of_event_types):
            for x in range(number_of_states):
                intensities[e2] += partial_sums[e1, x, e2]
        intensity_max += intensities[e2]

    'Set initial state'
    if number_of_initial_events > 0:
        # if the initial condition is not empty (there are events before time_start)
        state = initial_condition_states[number_of_initial_events-1]
        # the state at time_start is the state coordinate of the most recent mark
    else: # if no initial condition is given, use the given initial state
        state = initial_state

    'Simulate the state-dependent Hawkes process'
    cdef int max_size = number_of_initial_events + max_number_of_events
    cdef np.ndarray[DTYPEf_t, ndim=1] result_times = np.zeros(max_size, dtype=DTYPEf)
    cdef np.ndarray[DTYPEi_t, ndim=1] result_events = np.zeros(max_size, dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=1] result_states = np.zeros(max_size, dtype=DTYPEi)
    result_times[0:number_of_initial_events] = initial_condition_times
    result_events[0:number_of_initial_events] = initial_condition_events
    result_states[0:number_of_initial_events] = initial_condition_states
    time = time_start
    cdef double random_exponential, random_uniform, intensity_total
    cdef np.ndarray[DTYPEf_t, ndim=1] probabilities_state
    cdef double r
    n = number_of_initial_events
    while time < time_end and n < max_size:
        'Generate an exponential random variable with rate parameter intensity_max'
        random_exponential = np.random.exponential(1 / intensity_max)
        'Increase the time'
        time += random_exponential
        if time <= time_end:  # if we are not out of the considered time window
            'Update the partial sums at the current time using the recursive structure of the intensity'
            for e1 in range(number_of_event_types):
                for x in range(number_of_states):
                    for e2 in range(number_of_event_types):
                        beta = decay_coefficients[e1, x, e2]
                        partial_sums[e1, x, e2] *= exp(-beta * random_exponential)
            'Update the intensities of events and compute the total intensity'
            intensity_total = 0
            for e2 in range(number_of_event_types):
                intensities[e2] = base_rates[e2]
                for e1 in range(number_of_event_types):
                    for x in range(number_of_states):
                        intensities[e2] += partial_sums[e1, x, e2]
                intensity_total += intensities[e2]
            'Determine if this is an event time'
            random_uniform =  np.random.uniform(0, intensity_max)
            if random_uniform < intensity_total:  # then yes, it is an event time
                'Determine what event occurs'
                event = random_choice(intensities)
                'Determine the new state of the system'
                probabilities_state = transition_probabilities[state, event, :]
                state = random_choice(probabilities_state)
                'Update the result'
                result_times[n] = time  # add the event time to the result
                result_events[n] = event  # add the new event to the result
                result_states[n] = state  # add the new state to the result
                n += 1  # increment counter of number of events
                'Update the partial sums, the intensities of events and the total intensity'
                for e in range(number_of_event_types):  # only the partial sums S_{event, state, . } change
                    alpha = impact_coefficients[event, state, e]
                    partial_sums[event, state, e] += alpha
                    intensities[e] += alpha
                    intensity_total += alpha
            intensity_max = intensity_total  # the maximum total intensity until the next event
    return result_times[0:n], result_events[0:n], result_states[0:n]

def random_choice(np.ndarray[DTYPEf_t, ndim=1] weights):
    cdef double total, cumulative_sum, random_uniform
    cdef int result, dim, n, done
    dim = weights.shape[0]
    total = 0
    for n in range(dim):
        total += weights[n]
    random_uniform =  np.random.uniform(0, total)
    result = 0
    done = 0
    cumulative_sum = weights[result]
    if random_uniform <= cumulative_sum:
        done = 1
    while done == 0:
        result += 1
        cumulative_sum += weights[result]
        if random_uniform <= cumulative_sum:
            done = 1
    return result

def compute_events_residuals(np.ndarray[DTYPEf_t, ndim=1] base_rates,
                             np.ndarray[DTYPEf_t, ndim=3] impact_coefficients,
                             np.ndarray[DTYPEf_t, ndim=3] decay_coefficients,
                             int number_of_event_types,
                             int number_of_states,
                             np.ndarray[DTYPEf_t, ndim=1] times,
                             np.ndarray[DTYPEi_t, ndim=1] events,
                             np.ndarray[DTYPEi_t, ndim=1] states,
                             double time_start,
                             np.ndarray[DTYPEf_t, ndim=3] initial_partial_sums):
    
    'Find the start index'
    cdef int index_start = bisect.bisect_right(times, time_start)  # events at and before this time are treated as an initial condition
    
    'Initialise'
    cdef int length = times.shape[0]
    cdef np.ndarray[DTYPEf_t, ndim=2] residuals = np.zeros((number_of_event_types, length - index_start + 1), dtype=DTYPEf)
    # at most length-index_start residuals per event type, the +1 is to deal with boundary effect in main loop
    cdef np.ndarray[DTYPEi_t, ndim=1] residuals_lengths = np.zeros(number_of_event_types, dtype=DTYPEi)
    cdef np.ndarray[DTYPEf_t, ndim=1] previous_times = time_start*np.ones(number_of_event_types, dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=3] partial_sums = np.zeros((number_of_event_types, number_of_states, number_of_event_types), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=3] partial_sums_old = np.zeros((number_of_event_types, number_of_states, number_of_event_types), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=3] impact_decay_ratios = np.zeros((number_of_event_types, number_of_states, number_of_event_types), dtype=DTYPEf)
    cdef int e1, x, e2, n, e, event, state, i, pos
    cdef double alpha, beta, time, time_last

    'Compute ratios alpha/beta just once'
    for e1 in range(number_of_event_types):
        for x in range(number_of_states):
            for e2 in range(number_of_event_types):
                alpha = impact_coefficients[e1, x, e2]
                beta = decay_coefficients[e1, x, e2]
                impact_decay_ratios[e1, x, e2] = alpha / beta

    '''Initialise the partial sums S_{e',x',e} that will allow us to compute the residuals recursively.
    Note that, here, we work with (alpha_{e',x',e'}/beta_{e',x',e'})*S_{e',x',e'} instead of S_{e',x',e}'''
    for n in range(index_start):
        time = times[n]
        event = events[n]
        state = states[n]
        for e in range(number_of_event_types):
            beta = decay_coefficients[event, state, e]
            partial_sums[event, state, e] += exp(-beta * (time_start - time))
    # By doing so, multiplying the partial sums by the impact/decay coefficients needs to be done only once
    for event in range(number_of_event_types):
        for state in range(number_of_states):
            for e in range(number_of_event_types):
                partial_sums[event, state, e] = partial_sums[event, state, e] * impact_decay_ratios[event, state, e]
    'Users can also pass directly the initial_partial_sums'
    for e1 in range(number_of_event_types):
        for x in range(number_of_states):
            for e2 in range(number_of_event_types):
                partial_sums[e1, x, e2] += initial_partial_sums[e1, x, e2]
                partial_sums_old[e1, x, e2] = partial_sums[e1, x, e2]
    'Compute residuals'
    time_last = time_start
    for n in range(index_start, length):
        time = times[n]
        event = events[n]
        state = states[n]
        pos = residuals_lengths[event]
        'Contribution of the base rate'
        residuals[event, pos] += (time - previous_times[event])*base_rates[event]
        'Contribution of the constant terms to residuals of all event types'
        for e in range(number_of_event_types):
            if e != event:
                i = residuals_lengths[e]
                residuals[e, i] += impact_decay_ratios[event, state, e]
            if e == event:  # in this case, this event contributes to the next residual
                residuals[e, pos+1] += impact_decay_ratios[event, state, e]
        'Update partial sums up to current time but excluding the current time: decay effect'
        for e1 in range(number_of_event_types):
            for x in range(number_of_states):
                for e2 in range(number_of_event_types):
                    partial_sums[e1, x, e2] *= exp(-decay_coefficients[e1, x, e2] * (time - time_last))
        'Contribution of the partial sums'
        for e in range(number_of_event_types):
            for x in range(number_of_states):
                residuals[event, pos] += partial_sums_old[e, x, event] - partial_sums[e, x, event]
                # save new partial sums to compute the next residual for this event type
                partial_sums_old[e, x, event] = partial_sums[e, x, event]
        'Update partial sums: jump effect due to current event'
        for e in range(number_of_event_types):
            partial_sums[event, state, e] += impact_decay_ratios[event, state, e]
        'Update variables that keep track of current position'
        time_last = time
        previous_times[event] = time
        residuals_lengths[event] += 1
    'Return result'
    result = []
    for e in range(number_of_event_types):
        length = residuals_lengths[e]
        result.append(residuals[e, 0:length])
    return result

def compute_total_residuals(np.ndarray[DTYPEf_t, ndim=3] transition_probabilities,
                            np.ndarray[DTYPEf_t, ndim=1] base_rates,
                            np.ndarray[DTYPEf_t, ndim=3] impact_coefficients,
                            np.ndarray[DTYPEf_t, ndim=3] decay_coefficients,
                            int number_of_event_types,
                            int number_of_states,
                            np.ndarray[DTYPEf_t, ndim=1] times,
                            np.ndarray[DTYPEi_t, ndim=1] events,
                            np.ndarray[DTYPEi_t, ndim=1] states,
                            double time_start,
                            np.ndarray[DTYPEf_t, ndim=3] initial_partial_sums,
                            int initial_state):
    'Find the start index'
    cdef int index_start = bisect.bisect_right(times, time_start)  # events at and before this time are treated as an initial condition  
    'Initialise'
    cdef int length = times.shape[0]
    cdef np.ndarray[DTYPEf_t, ndim=3] residuals = np.zeros((number_of_event_types, number_of_states, length - index_start + 1), dtype=DTYPEf)
    # at most length-index_start residuals per event type, the +1 is to deal with boundary effect in main loop
    cdef np.ndarray[DTYPEi_t, ndim=2] residuals_lengths = np.zeros((number_of_event_types, number_of_states), dtype=DTYPEi)
    cdef np.ndarray[DTYPEf_t, ndim=3] partial_sums = np.zeros((number_of_event_types, number_of_states, number_of_event_types), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=3] partial_sums_old = np.zeros((number_of_event_types, number_of_states, number_of_event_types), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=3] impact_decay_ratios = np.zeros((number_of_event_types, number_of_states, number_of_event_types), dtype=DTYPEf)
    cdef int e1, x, e2, n, e, event, state, i, pos, previous_state, x2
    cdef double alpha, beta, time, time_last, phi
    'Compute ratios alpha/beta just once'
    for e1 in range(number_of_event_types):
        for x in range(number_of_states):
            for e2 in range(number_of_event_types):
                alpha = impact_coefficients[e1, x, e2]
                beta = decay_coefficients[e1, x, e2]
                impact_decay_ratios[e1, x, e2] = alpha / beta
    '''Initialise the partial sums S_{e',x',e} that will allow us to compute the residuals recursively.
    Note that, here, we work with (alpha_{e',x',e'}/beta_{e',x',e'})*S_{e',x',e'} instead of S_{e',x',e}'''
    for n in range(index_start):
        time = times[n]
        event = events[n]
        state = states[n]
        for e in range(number_of_event_types):
            beta = decay_coefficients[event, state, e]
            partial_sums[event, state, e] += exp(-beta * (time_start - time))
    # By doing so, multiplying the partial sums by the impact/decay coefficients needs to be done only once
    for event in range(number_of_event_types):
        for state in range(number_of_states):
            for e in range(number_of_event_types):
                partial_sums[event, state, e] = partial_sums[event, state, e] * impact_decay_ratios[event, state, e]
    'Users can also pass directly the initial_partial_sums'
    for e1 in range(number_of_event_types):
        for x in range(number_of_states):
            for e2 in range(number_of_event_types):
                partial_sums[e1, x, e2] += initial_partial_sums[e1, x, e2]
                partial_sums_old[e1, x, e2] = partial_sums[e1, x, e2]
    'Set initial state'
    if index_start > 0:
        # if the initial condition is not empty (there are events before time_start)
        previous_state = states[index_start-1]
        # the state at time_start is the state coordinate of the most recent mark
    else:  # if no initial condition is given, use the given initial state
        previous_state = initial_state
    'Compute residuals'
    time_last = time_start
    for n in range(index_start, length):
        time = times[n]
        event = events[n]
        state = states[n]
        'Update partial sums up to current time but excluding the current time: decay effect'
        for e1 in range(number_of_event_types):
            for x in range(number_of_states):
                for e2 in range(number_of_event_types):
                    partial_sums[e1, x, e2] *= exp(-decay_coefficients[e1, x, e2] * (time - time_last))
        'Compute contribution of [time_last, time) to residuals'
        for e in range(number_of_event_types):
            for x in range(number_of_states):
                phi = transition_probabilities[previous_state, e, x]
                pos = residuals_lengths[e, x]
                'Contribution of the base rate'
                residuals[e, x, pos] += (time - time_last)*base_rates[e]*phi
                'Contribution of the partial sums'
                for e2 in range(number_of_event_types):
                    for x2 in range(number_of_states):
                        residuals[e, x, pos] += phi*(partial_sums_old[e2, x2, e] - partial_sums[e2, x2, e])
                'Contribuion of the constant terms'
                if e != event:
                    residuals[e, x, pos] += impact_decay_ratios[event, state, e]*phi
                if e == event:  # in this case, this event contributes to the residual betwen time and next_time; hence the new current state must be used
                    phi = transition_probabilities[state, e, x]
                    if x != state:
                        residuals[e, x, pos] += impact_decay_ratios[event, state, e]*phi
                    if x == state:  # in this case, this event contributs to the next residual of the mark (e,x)
                        residuals[e, x, pos+1] += impact_decay_ratios[event, state, e]*phi
        'Save current partials sums for the computation of the residuals over the next time increment'
        for e1 in range(number_of_event_types):
            for x in range(number_of_states):
                for e2 in range(number_of_event_types):
                    partial_sums_old[e1, x, e2] = partial_sums[e1, x, e2]
        'Update partial sums: jump effect due to current event'
        for e in range(number_of_event_types):
            partial_sums[event, state, e] += impact_decay_ratios[event, state, e]
        'Update variables that keep track of current position'
        time_last = time
        previous_state = state
        residuals_lengths[event, state] += 1
    'Return result'
    result = []
    for e in range(number_of_event_types):
        for x in range(number_of_states):
            length = residuals_lengths[e, x]
            result.append(residuals[e, x, 0:length])
    return result
