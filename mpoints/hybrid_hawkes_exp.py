from typing import Optional, Sequence
import numpy as np
import math
import copy
import bisect
import scipy.optimize as opt
from . import hybrid_hawkes_exp_cython as cy


class HybridHawkesExp:
    """
    This class implements state-dependent Hawkes processes with exponential kernels, a subclass of hybrid marked point
    processes.
    The main features it provides include simulation and statistical inference (estimation).

    :type number_of_event_types: int
    :param number_of_event_types: number of different event types.
    :type number_of_states: int
    :param number_of_states: number of possible states.
    :type events_labels: list of strings
    :param events_labels: names of the different event types.
    :type states_labels: list of strings
    :param states_labels: names of the possible states.
    """

    def __init__(self, number_of_event_types, number_of_states, events_labels, states_labels):
        """
        Initialises an instance.

        :type number_of_event_types: int
        :param number_of_event_types: number of different event types.
        :type number_of_states: int
        :param number_of_states: number of possible states.
        :type events_labels: list of strings
        :param events_labels: names of the different event types.
        :type states_labels: list of strings
        :param states_labels: names of the possible states.
        """
        self.number_of_event_types = number_of_event_types
        self.number_of_states = number_of_states
        self.events_labels = events_labels
        self.states_labels = states_labels
        self.transition_probabilities = np.zeros(
            (number_of_states, number_of_event_types, number_of_states))
        self.base_rates = np.zeros(number_of_event_types)
        self.impact_coefficients = np.zeros(
            (number_of_event_types, number_of_states, number_of_event_types))
        self.decay_coefficients = np.zeros(
            (number_of_event_types, number_of_states, number_of_event_types))
        self.impact_decay_ratios = np.zeros(
            (number_of_event_types, number_of_states, number_of_event_types))

    def set_transition_probabilities(self, transition_probabilities):
        r"""
        Fixes the transition probabilities :math:`\phi` of the state-dependent Hawkes process.
        The are used to :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.simulate` and
        :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.compute_total_residuals`.

        :type transition_probabilities: 3D numpy array
        :param transition_probabilities: shape should be :math:`(d_x, d_e,d_x)` where :math:`d_e` and :math:`d_x`
                                         are the number of event types and states, respectively.
                                         The entry :math:`i, j, k` is the probability of going from state :math:`i`
                                         to state :math:`k` when an event of type :math:`j` occurs.
        :return:
        """
        'Raise ValueError if the given parameters do not have the right shape'
        if np.shape(transition_probabilities) != (self.number_of_states, self.number_of_event_types,
                                                  self.number_of_states):
            raise ValueError(
                'given transition probabilities have incorrect shape')
        self.transition_probabilities = copy.copy(transition_probabilities)

    def set_hawkes_parameters(self, base_rates, impact_coefficients, decay_coefficients):
        r"""
        Fixes the parameters :math:`(\nu, \alpha, \beta)` that define the intensities (arrival rates) of events.
        The are used in
        :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.simulate`,
        :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.compute_events_residuals`
        and :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.compute_total_residuals`.

        :type base_rates: 1D numpy array
        :param base_rates: one base rate :math:`\nu_e` per event type :math:`e`.
        :type impact_coefficients: 3D numpy array
        :param impact_coefficients: the alphas :math:`\alpha_{e'xe}`.
        :type decay_coefficients: 3D numpy array
        :param decay_coefficients: the betas :math:`\beta_{e'xe}`.
        :return:
        """
        'Raise ValueError if the given parameters do not have the right shape'
        if np.shape(base_rates) != (self.number_of_event_types,):
            raise ValueError('given base rates have incorrect shape')
        if np.shape(impact_coefficients) != (self.number_of_event_types, self.number_of_states,
                                             self.number_of_event_types):
            raise ValueError('given impact coefficients have incorrect shape')
        if np.shape(decay_coefficients) != (self.number_of_event_types, self.number_of_states,
                                            self.number_of_event_types):
            raise ValueError('given decay coefficients have incorrect shape')
        self.base_rates = copy.copy(base_rates)
        self.impact_coefficients = copy.copy(impact_coefficients)
        self.decay_coefficients = copy.copy(decay_coefficients)
        self.impact_decay_ratios = np.divide(
            impact_coefficients, decay_coefficients)

    @staticmethod
    def kernel_at_time(time, alpha, beta):
        r"""
        Evaluates the kernel of the model at the given time with the given parameters.

        :type time: float
        :param time: the positive time :math:`t`.
        :type alpha: float
        :param alpha: a non-negative :math:`\alpha`.
        :type beta: float
        :param beta: a positive :math:`\beta`.
        :rtype: float
        :return: :math:`\alpha\exp(-\beta t)`.
        """
        return alpha * np.exp(- np.multiply(time, beta))

    'Functions that estimate the model parameters'

    def estimate_transition_probabilities(self, events, states):
        r"""
        Estimates the transition probabilities :math:`\phi` of the state process from the data.
        This method returns the maximum likelihood estimate.
        One can prove that it coincides with the empirical transition probabilities.

        :type events: 1D array of int
        :param events: the sequence of event types, `events[n]` is the event type of the `n` th event.
        :type states: 1D array of int
        :param states: the sequence of states, `states[n]` is the new state of the system following the `n` th event.
        :rtype: 3D array
        :return: the estimated transition probabilities :math:`\phi`.
        """
        result = np.zeros(
            (self.number_of_states, self.number_of_event_types, self.number_of_states))
        count_of_states_events = np.zeros(
            (self.number_of_states, self.number_of_event_types))
        for n in range(1, len(events)):
            event = events[n]
            state_before = states[n - 1]
            state_after = states[n]
            count_of_states_events[state_before, event] += 1
            result[state_before, event, state_after] += 1
        for x1 in range(self.number_of_states):
            for e in range(self.number_of_event_types):
                size = count_of_states_events[x1, e]
                if size > 0:
                    for x2 in range(self.number_of_states):
                        result[x1, e, x2] /= size
                else:
                    message = 'Warning: Transition probabilities from state ' + \
                        str(x1)
                    message += ' when events of type ' + \
                        str(e) + ' occur cannot be estimated because'
                    message += ' events of this type never occur this state'
                    print(message)
        return result

    def estimate_hawkes_parameters(self, times, events, states, time_start, time_end, maximum_number_of_iterations=2000,
                                   method='TNC', parameters_lower_bound=10**(-6), parameters_upper_bound=None,
                                   given_guesses=[], number_of_random_guesses=1,
                                   min_decay_coefficient=0.5, max_decay_coefficient=100, parallel_estimation=True):
        r"""
        Estimates the parameters of the intensities (arrival rates) of events, i.e., :math:`(\nu, \alpha, \beta)`.
        Estimation if performed via maximum likelihood. This method uses the `scipy.minimize` library.

        :type times: 1D numpy array of float
        :param times: the times at which events occur.
        :type events: 1D numpy array of int
        :param events: the sequence of event types, `events[n]` is the event type of the `n` th event.
        :type states: 1D numpy array of int
        :param states: the sequence of states, `states[n]` is the new state of the system following the `n` th event.
        :type time_start: float
        :param time_start: the time at which we consider that the process started, prior times are treated as an
                           initial condition.
        :type time_end: float
        :param time_end: the time at which we stopped to record the process.
        :type maximum_number_of_iterations: int
        :param maximum_number_of_iterations:  will be passed to the `maxiter` argument in `scipy.minimize`.
                                              Depending on `method`, it is the maximum number of iterations or
                                              function evaluations.
        :type method: string
        :param method: the optimisation method used in `scipy.minimize`.
        :type parameters_lower_bound: float
        :param parameters_lower_bound: lower bound on all the parameters.
        :type parameters_upper_bound: float
        :param parameters_upper_bound: upper bound on all the parameters.
        :type given_guesses: list of 1D numpy array
        :param given_guesses: every member `x` is an initial guess on the parameters.
                              For every `x`, we attempt to maximise the likelihood starting from `x`.
                              One can go from `x` to :math:`(\nu, \alpha, \beta)` and vice versa using
                              :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.array_to_parameters`
                              and :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.parameters_to_array`.
                              We retain the solution that gives the highest likelihood.
        :type number_of_random_guesses: int
        :param number_of_random_guesses: the method can also generate random initial guesses.
        :type min_decay_coefficient: numpy array or float
        :param min_decay_coefficient: defines how a random guess is generated.
        :type max_decay_coefficient: numpy array of float
        :param max_decay_coefficient: a random guess on :math:`\beta_{e'xe}` is generated uniformly in the interval
                                      [`min_decay_coefficient[e',x,e]`, `max_decay_coefficient[e',x.e]`] but on a
                                      logarithmic scale.
        :type parallel_estimation: boolean
        :param parallel_estimation: the MLE problem can be decomposed into :math:`d_e` independent optimisation
                                    problems, where :math:`d_e` is the number of event types. When True, each problem
                                    is solved independently. In this case, the limit on the number of iterations
                                    or function evaluations is applied independently to each sub-problem.
        :rtype: scipy.optimize.OptimizerResult, 1D numpy array, string
        :return: The first object is the optimisation result and contains the maximum likelihood estimate along with
                 additional information on the optimisation routine. The second object contains the initial guess
                 that resulted in the highest likelihood after running the optimisation procedure.
                 The third object indicates the nature of this initial guess ('random' or 'given').
        """

        'Generate additional random guesses of the parameters'
        guesses = copy.copy(given_guesses)
        # if a scalar was given instead of a matrix
        if np.shape(min_decay_coefficient) == ():
            min_decay_coefficients = min_decay_coefficient * np.ones((self.number_of_event_types, self.number_of_states,
                                                                      self.number_of_event_types))
        # if a scalar was given instead of a matrix
        if np.shape(max_decay_coefficient) == ():
            max_decay_coefficients = max_decay_coefficient * np.ones((self.number_of_event_types, self.number_of_states,
                                                                      self.number_of_event_types))
        if number_of_random_guesses > 0:
            'Compute the average intensities'
            average_intensities = np.zeros(self.number_of_event_types)
            for n in range(len(times)):
                e = events[n]
                average_intensities[e] += 1
            average_intensities = np.divide(
                average_intensities, time_end - time_start)
            for n in range(number_of_random_guesses):
                'Base rates'
                guess_base_rates = np.zeros(self.number_of_event_types)
                for e in range(self.number_of_event_types):
                    guess_base_rates[e] = average_intensities[e] / 2
                'Decay coefficients'
                guess_decay_coefficients = np.zeros((self.number_of_event_types, self.number_of_states,
                                                     self.number_of_event_types))
                for e1 in range(self.number_of_event_types):
                    for x in range(self.number_of_states):
                        for e2 in range(self.number_of_event_types):
                            u_min = math.log10(
                                min_decay_coefficients[e1, x, e2])
                            u_max = math.log10(
                                max_decay_coefficients[e1, x, e2])
                            u = np.random.uniform(u_min, u_max)
                            beta = 10 ** u
                            guess_decay_coefficients[e1, x, e2] = beta
                'Impact coefficients'
                guess_impact_coefficients = np.zeros((self.number_of_event_types, self.number_of_states,
                                                      self.number_of_event_types))
                for e1 in range(self.number_of_event_types):
                    for x in range(self.number_of_states):
                        for e2 in range(self.number_of_event_types):
                            u = np.random.uniform(0, 1)
                            alpha = u * guess_decay_coefficients[e1, x, e2]
                            guess_impact_coefficients[e1, x, e2] = alpha
                'Save the random guess to the list of guesses'
                g = HybridHawkesExp.parameters_to_array(guess_base_rates, guess_impact_coefficients,
                                                        guess_decay_coefficients)
                guesses.append(g)

        'For each initial guess, apply the optimizer'
        if not parallel_estimation:
            optimal_results = []
            for g in guesses:
                dimension = self.number_of_event_types + 2 * \
                    self.number_of_states * self.number_of_event_types ** 2
                bounds = [(parameters_lower_bound,
                           parameters_upper_bound)] * dimension
                'Define the minus likelihood and gradient functions'
                def likelihood_minus(parameters):
                    result = - self.log_likelihood_of_events(parameters, times, events, states,
                                                             time_start, time_end)
                    return result

                def gradient_of_likelihood_minus(parameters):
                    result = - \
                        self.gradient(parameters, times, events,
                                      states, time_start, time_end)
                    return result
                o = opt.minimize(likelihood_minus, g, method=method,
                                 bounds=bounds, jac=gradient_of_likelihood_minus,
                                 options={'maxiter': maximum_number_of_iterations})
                optimal_results.append(o)
            'Look for the solution that gives the highest log-likelihood'
            index_of_best_result = 0
            log_likelihood_minus = optimal_results[0].fun
            for i in range(1, len(optimal_results)):
                current_log_likelihood_minus = optimal_results[i].fun
                if current_log_likelihood_minus < log_likelihood_minus:
                    index_of_best_result = i
                    log_likelihood_minus = current_log_likelihood_minus
            best_initial_guess = guesses[index_of_best_result]
            kind_of_best_initial_guess = ''
            if index_of_best_result < len(given_guesses):
                kind_of_best_initial_guess += 'given'
            elif index_of_best_result - len(given_guesses) < number_of_random_guesses:
                kind_of_best_initial_guess += 'random'
            'Return the OptimizeResult instance that gives the biggest likelihood'
            return optimal_results[index_of_best_result], best_initial_guess, kind_of_best_initial_guess
        else:
            dimension = 1 + 2 * self.number_of_states * self.number_of_event_types
            bounds = [(parameters_lower_bound,
                       parameters_upper_bound)] * dimension
            opt_nus = np.zeros(self.number_of_event_types)
            opt_alphas = np.zeros(
                (self.number_of_event_types, self.number_of_states, self.number_of_event_types))
            opt_betas = np.zeros(
                (self.number_of_event_types, self.number_of_states, self.number_of_event_types))
            best_guess_nu = np.zeros(self.number_of_event_types)
            best_guess_alphas =\
                np.zeros((self.number_of_event_types,
                         self.number_of_states, self.number_of_event_types))
            best_guess_betas = np.zeros(
                (self.number_of_event_types, self.number_of_states, self.number_of_event_types))
            success = True
            successes = []
            status = -999
            statuses = []
            message = 'Multiple messages because parallel estimation'
            messages = []
            fun = 0
            jacs = []
            hesss = []
            nfev = 0
            nit = 0
            kinds_of_best_initial_guesses = ''
            for e in range(self.number_of_event_types):
                'Define the minus likelihood and gradient functions'
                def likelihood_minus(parameters):
                    result = - self.log_likelihood_of_events_partial(e, parameters, times, events, states,
                                                                     time_start, time_end)
                    return result

                def gradient_of_likelihood_minus(parameters):
                    result = - \
                        self.gradient_partial(
                            e, parameters, times, events, states, time_start, time_end)
                    return result
                'For each initial guess, optimise likelihood'
                optimal_results = []
                for g in guesses:
                    guess_nus, guess_alphas, guess_betas = self.array_to_parameters(g, self.number_of_event_types,
                                                                                    self.number_of_states,
                                                                                    self.number_of_event_types)
                    g_partial = self.parameters_to_array(guess_nus[e:e+1],
                                                         guess_alphas[:,
                                                                      :, e:e+1],
                                                         guess_betas[:, :, e:e+1])
                    o = opt.minimize(likelihood_minus, g_partial, method=method,
                                     bounds=bounds, jac=gradient_of_likelihood_minus,
                                     options={'maxiter': maximum_number_of_iterations})
                    optimal_results.append(o)
                'Look for the solution that gives the highest log-likelihood'
                index_of_best_result = 0
                log_likelihood_minus = optimal_results[0].fun
                for i in range(1, len(optimal_results)):
                    current_log_likelihood_minus = optimal_results[i].fun
                    if current_log_likelihood_minus < log_likelihood_minus:
                        index_of_best_result = i
                        log_likelihood_minus = current_log_likelihood_minus
                best_initial_guess = guesses[index_of_best_result]
                kind_of_best_initial_guess = ''
                if index_of_best_result < len(given_guesses):
                    kind_of_best_initial_guess += 'given'
                elif index_of_best_result - len(given_guesses) < number_of_random_guesses:
                    kind_of_best_initial_guess += 'random'
                o = optimal_results[index_of_best_result]
                # Save the kind of best initial guess for this event type
                kinds_of_best_initial_guesses += kind_of_best_initial_guess + ' '
                # Save optimal parameters
                v, a, b = self.array_to_parameters(
                    o.x, self.number_of_event_types, self.number_of_states, 1)
                opt_nus[e:e+1] = v
                opt_alphas[:, :, e:e+1] = a
                opt_betas[:, :, e:e+1] = b
                # Save best initial guess
                v, a, b = self.array_to_parameters(best_initial_guess, self.number_of_event_types,
                                                   self.number_of_states, self.number_of_event_types)
                best_guess_nu[e] = v[e]
                best_guess_alphas[:, :, e] = a[:, :, e]
                best_guess_betas[:, :, e] = b[:, :, e]
                # Save optimiser information
                successes.append(o.success)
                statuses.append(o.status)
                messages.append(o.message)
                if success and not o.success:
                    success = False
                    status = o.status
                fun += o.fun
                jacs.append(o.jac)
                nfev += o.nfev
                nit += o.nit
            'Aggregate the Optimize Results into a single one'
            best_initial_guess = self.parameters_to_array(
                best_guess_nu, best_guess_alphas, best_guess_betas)
            o = opt.OptimizeResult()
            x = self.parameters_to_array(opt_nus, opt_alphas, opt_betas)
            o['x'] = x
            o['success'] = success
            o['successes'] = successes
            o['status'] = status
            o['statuses'] = statuses
            o['message'] = message
            o['messages'] = messages
            o['fun'] = fun
            o['jacs'] = jacs
            o['hesss'] = hesss
            o['nfev'] = nfev
            o['nit'] = nit
            return o, best_initial_guess, kinds_of_best_initial_guesses

    'Specification testing and simulation'

    def compute_events_residuals(self, times, events, states, time_start, initial_partial_sums=0):
        r"""
        Computes the events residuals :math:`r^e_n` defined by

        .. math::

            r^e_n := \int_{t^e_{n-1}}^{t^e_n} \lambda_e (t)dt,

        where :math:`t^e_n` is the time when the `n` th event of type `e` occurred.
        The methods wraps a C implementation that was obtained via Cython.

        :type times: 1D numpy array of float
        :param times: the times at which events occur.
        :type events: 1D numpy array of int
        :param events: the sequence of event types, `events[n]` is the event type of the `n` th event.
        :type states: 1D numpy array of int
        :param states: the sequence of states, `states[n]` is the new state of the system following the `n` th event.
        :type time_start: float
        :param time_start: the time at which we consider that the process started, prior times are treated as an
                           initial condition.
        :type initial_partial_sums: 3D numpy array
        :param initial_partial_sums: the initial condition can also be given implicitly via the partial sums
                                     :math:`S_{e',x,e}(-\infty, \mbox{time_start}]`.
        :rtype: list of 1D numpy arrays
        :return: the `e` th element of the list is the sequence :math:`(r^e_n)` corresponding to the event type `e`.
        """
        # Check if no initial partial sums if given
        s = np.zeros((self.number_of_event_types,
                     self.number_of_states, self.number_of_event_types))
        if len(np.shape(initial_partial_sums)) != 0:
            s = initial_partial_sums
            s = np.divide(s, self.decay_coefficients)
        return cy.compute_events_residuals(self.base_rates,
                                           self.impact_coefficients,
                                           self.decay_coefficients,
                                           self.number_of_event_types,
                                           self.number_of_states,
                                           times,
                                           events,
                                           states,
                                           time_start,
                                           s)

    def compute_total_residuals(self, times, events, states, time_start, initial_partial_sums=0,
                                initial_state=0):
        r"""

        Computes the total residuals :math:`r^{ex}_n` defined by

        .. math::

            r^{ex}_n := \int_{t^{ex}_{n-1}}^{t^{ex}_n}\phi_{e}(X(t),x) \lambda_e (t)dt,

        where :math:`t^{ex}_n` is the time when the `n` th event of type `e` after which the state is `x` occurred.
        The methods wraps a C implementation that was obtained via Cython.

        :type times: 1D numpy array of float
        :param times: the times at which events occur.
        :type events: 1D numpy array of int
        :param events: the sequence of event types, `events[n]` is the event type of the `n` th event.
        :type states: 1D numpy array of int
        :param states: the sequence of states, `states[n]` is the new state of the system following the `n` th event.
        :type time_start: float
        :param time_start: the time at which we consider that the process started, prior times are treated as an
                           initial condition.
        :type initial_partial_sums: 3D numpy array
        :param initial_partial_sums: the initial condition can also be given implicitly via the partial sums
                                     :math:`S_{e',x,e}(-\infty, \mbox{time_start}]`.
        :type initial_state: int
        :param initial_state: if there are no event times before `time_start`, this is used as the initial state.
        :rtype: list of 1D numpy arrays
        :return: the sequence :math:`(r^{ex}_n)` is the `x` + `e` * `number_of_states` th element in the list.
        """
        # Check if no initial partial sums if given
        s = np.zeros((self.number_of_event_types,
                     self.number_of_states, self.number_of_event_types))
        if len(np.shape(initial_partial_sums)) != 0:
            s = initial_partial_sums
            s = np.divide(s, self.decay_coefficients)
        return cy.compute_total_residuals(self.transition_probabilities,
                                          self.base_rates,
                                          self.impact_coefficients,
                                          self.decay_coefficients,
                                          self.number_of_event_types,
                                          self.number_of_states,
                                          times,
                                          events,
                                          states,
                                          time_start,
                                          s,
                                          initial_state)

    def simulate(self,
                 time_start: float,
                 time_end: float,
                 initial_condition_times: Optional[Sequence] = None,
                 initial_condition_events: Optional[Sequence] = None,
                 initial_condition_states: Optional[Sequence] = None,
                 initial_partial_sums: Optional[np.array] = None,
                 initial_state: int = 0,
                 max_number_of_events: int = 10**6):
        """
        Simulates a sample path of the state-dependent Hawkes process.
        The methods wraps a C implementation that was obtained via Cython.

        :type time_start: float
        :param time_start: time at which the simulation starts.
        :type time_end: float
        :param time_end: time at which the simulation ends.
        :type initial_condition_times: array
        :param initial_condition_times: times of events before and including `time_start`.
        :type initial_condition_events: array of int
        :param initial_condition_events: types of the events that occurred at `initial_condition_times`.
        :type initial_condition_states: array of int
        :param initial_condition_states: values of the state process just after the `initial_condition_times`.
        :type initial_partial_sums: 3D numpy array
        :param initial_partial_sums: the initial condition can also be given implicitly via the partial sums
                                     :math:`S_{e',x,e}(-\infty, \mbox{time_start}]`.
        :type initial_state: int
        :param initial_state: if there are no event times before `time_start`, this is used as the initial state.
        :type max_number_of_events: int
        :param max_number_of_events: the simulation stops when this number of events is reached
                                     (including the initial condition).
        :rtype: array, array of int, array of int
        :return: the times at which the events occur, their types and the values of the state process right after
                 each event. Note that these include the initial condition as well.
        """
        # Check if no initial partial sums if given
        if initial_partial_sums is None:
            s = np.zeros((self.number_of_event_types,
                         self.number_of_states, self.number_of_event_types))
        elif isinstance(initial_partial_sums, (int, float)):
            s = initial_partial_sums * np.ones((self.number_of_event_types,
                                                self.number_of_states,
                                                self.number_of_event_types), dtype=float)
        else:
            assert isinstance(initial_partial_sums, np.ndarray)
            assert initial_partial_sums.shape == (
                self.number_of_event_types, self.number_of_states, self.number_of_event_types)
            s = np.array(initial_partial_sums, dtype=float)
        # Convert the initial condition to np.arrays if required
        if initial_condition_times is None:
            initial_condition_times = np.array([], dtype=float)
        elif not isinstance(initial_condition_times, np.ndarray):
            initial_condition_times = np.asarray(
                initial_condition_times, dtype=float)
        if initial_condition_events is None:
            initial_condition_events = np.array([], dtype=int)
        elif not isinstance(initial_condition_events, np.ndarray):
            initial_condition_events = np.asarray(
                initial_condition_events, dtype=int)
        if initial_condition_states is None:
            initial_condition_states = np.array([], dtype=int)
        elif not isinstance(initial_condition_states, np.ndarray):
            initial_condition_states = np.asarray(
                initial_condition_states, dtype=int)
        return cy.simulate(
            self.number_of_event_types,
            self.number_of_states,
            self.base_rates,
            self.impact_coefficients,
            self.decay_coefficients,
            self.transition_probabilities,
            initial_condition_times,
            initial_condition_events,
            initial_condition_states,
            s,
            initial_state,
            time_start,
            time_end,
            max_number_of_events
        )

    'Likelihood and gradient'

    def log_likelihood_of_events(self, parameters, times, events, states, time_start, time_end):
        r"""
        Computes the log-likelihood of the observed times and event types under the assumption that they
        are the realisation of a state-dependent Hawkes process with the given parameters.
        The log-likelihood is given by

        .. math::

            l := \sum_{n : t_0 < t_n \leq T} \lambda_{e_n}(t_n) - \sum_e \int_{t_0}^T \lambda_e(t)dt,

        where :math:`(t_n)` and :math:`(e_n)` are the sequences of event times and event types, respectively.
        The method wraps a C implementation that was obtained via Cython.

        :type parameters: 1D numpy array
        :param parameters: the parameters :math:`(\nu, \alpha, \beta)` put into a single array.
                           Go from `parameters` to :math:`(\nu, \alpha, \beta)` and vice versa using
                           :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.array_to_parameters`
                           and :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.parameters_to_array`.
        :type times: 1D numpy array of float
        :param times: the times at which events occur.
        :type events: 1D numpy array of int
        :param events: the sequence of event types, `events[n]` is the event type of the `n` th event.
        :type states: 1D numpy array of int
        :param states: the sequence of states, `states[n]` is the new state of the system following the `n` th event.
        :type time_start: float
        :param time_start: :math:`t_0`, the time at which we consider that the process started, prior times are treated as an
                           initial condition.
        :type time_end: float
        :param time_end: :math:`T`, the time at which we stopped to record the process.
        :rtype: float
        :return: the log-likelihood :math:`l`.
        """
        number_of_event_types = self.number_of_event_types
        number_of_states = self.number_of_states
        base_rates, impact_coefficients, decay_coefficients = \
            HybridHawkesExp.array_to_parameters(
                parameters, self.number_of_event_types, self.number_of_states)
        return cy.log_likelihood_of_events(base_rates, impact_coefficients, decay_coefficients, number_of_event_types,
                                           number_of_states, times, events, states, float(time_start), float(time_end))

    def gradient(self, parameters, times, events, states, time_start, time_end):
        r"""
        Computes the gradient of the log-likelihood :math:`l` with respect to the
        parameters :math:`(\nu, \alpha, \beta)`.
        The method wraps a C implementation that was obtained via Cython.

        :type parameters: 1D numpy array
        :param parameters: the parameters :math:`(\nu, \alpha, \beta)` put into a single array.
                           Go from `parameters` to :math:`(\nu, \alpha, \beta)` and vice versa using
                           :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.array_to_parameters`
                           and :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.parameters_to_array`.
        :type times: 1D numpy array of float
        :param times: the times at which events occur.
        :type events: 1D numpy array of int
        :param events: the sequence of event types, `events[n]` is the event type of the `n` th event.
        :type states: 1D numpy array of int
        :param states: the sequence of states, `states[n]` is the new state of the system following the `n` th event.
        :type time_start: float
        :param time_start: :math:`t_0`, the time at which we consider that the process started, prior times are treated as an
                           initial condition.
        :type time_end: float
        :param time_end: :math:`T`, the time at which we stopped to record the process.
        :rtype: float
        :return: the gradient of the log-likelihood :math:`l`.
        """
        number_of_event_types = self.number_of_event_types
        number_of_states = self.number_of_states
        base_rates, impact_coefficients, decay_coefficients = \
            HybridHawkesExp.array_to_parameters(
                parameters, self.number_of_event_types, self.number_of_states)
        g_base_rates, g_impact_coefficients, g_decay_coefficients =\
            cy.gradient(base_rates, impact_coefficients, decay_coefficients, number_of_event_types,
                        number_of_states, times, events, states, float(time_start), float(time_end))
        return self.parameters_to_array(g_base_rates, g_impact_coefficients, g_decay_coefficients)

    def log_likelihood_of_events_partial(self, event_type, parameters, times, events, states, time_start, time_end):
        r"""
        Computes the log-likelihood of the arrival times of events of the given type under the assumption that they
        are the realisation of a state-dependent Hawkes process with the given parameters.
        These parameters include only those that govern :math:`\lambda_e`, the intensity of events of type `e`.
        For example, among the base rates :math:`\nu`, only :math:`\nu_e` should be contained in `parameters`.
        This partial log-likelihood is given by

        .. math::

            l_e := \sum_{n : t_0 < t^e_n \leq T} \lambda_{e}(t^e_n) - \int_{t_0}^T \lambda_e(t)dt,

        where :math:`(t^e_n)` are the arrival times of events of the given type `e`.
        The method wraps a C implementation that was obtained via Cython.

        :type event_type: int
        :param event_type: `e`, the event type for which we want to compute the partial log-likelihood :math:`l_e`.
        :type parameters: 1D numpy array
        :param parameters: only the parameters :math:`(\nu, \alpha, \beta)` that govern :math:`\lambda_e`  put into a single
                           array.
                           Go from `parameters` to :math:`(\nu, \alpha, \beta)` and vice versa using
                           :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.array_to_parameters`
                           and :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.parameters_to_array`.
        :type times: 1D numpy array of float
        :param times: the times at which events occur.
        :type events: 1D numpy array of int
        :param events: the sequence of event types, `events[n]` is the event type of the `n` th event.
        :type states: 1D numpy array of int
        :param states: the sequence of states, `states[n]` is the new state of the system following the `n` th event.
        :type time_start: float
        :param time_start: :math:`t_0`, the time at which we consider that the process started, prior times are treated as an
                           initial condition.
        :type time_end: float
        :param time_end: :math:`T`, the time at which we stopped to record the process.
        :rtype: float
        :return: the partial log-likelihood :math:`l_e`.
        """
        number_of_event_types = self.number_of_event_types
        number_of_states = self.number_of_states
        base_rate, impact_coefficients, decay_coefficients = \
            HybridHawkesExp.array_to_parameters(
                parameters, number_of_event_types, number_of_states, 1)
        return cy.log_likelihood_of_events_partial(event_type, float(base_rate[0]), impact_coefficients[:, :, 0],
                                                   decay_coefficients[:, :, 0],
                                                   number_of_event_types, number_of_states, times, events, states,
                                                   float(time_start), float(time_end))

    def gradient_partial(self, event_type, parameters, times, events, states, time_start, time_end):
        r"""
        Computes the gradient of the partial log-likelihood :math:`l_e` with respect to the
        parameters :math:`(\nu, \alpha, \beta)` that govern :math:`\lambda_e`, the intensity of events of type `e`.
        The method wraps a C implementation that was obtained via Cython.

        :type event_type: int
        :param event_type: `e`, the event type for which we want to compute the gradient of the partial
                            log-likelihood :math:`l_e`.
        :type parameters: 1D numpy array
        :param parameters: only the parameters :math:`(\nu, \alpha, \beta)` that govern :math:`\lambda_e`  put into a single
                           array.
                           Go from `parameters` to :math:`(\nu, \alpha, \beta)` and vice versa using
                           :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.array_to_parameters`
                           and :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.parameters_to_array`.
        :type times: 1D numpy array of float
        :param times: the times at which events occur.
        :type events: 1D numpy array of int
        :param events: the sequence of event types, `events[n]` is the event type of the `n` th event.
        :type states: 1D numpy array of int
        :param states: the sequence of states, `states[n]` is the new state of the system following the `n` th event.
        :type time_start: float
        :param time_start: :math:`t_0`, the time at which we consider that the process started, prior times are treated as an
                           initial condition.
        :type time_end: float
        :param time_end: :math:`T`, the time at which we stopped to record the process.
        :rtype: float
        :return: the gradient of the partial log-likelihood :math:`l_e`.
        """
        number_of_event_types = self.number_of_event_types
        number_of_states = self.number_of_states
        base_rate, impact_coefficients, decay_coefficients = \
            HybridHawkesExp.array_to_parameters(
                parameters, number_of_event_types, number_of_states, 1)
        g_base_rate, g_impact_coefficients, g_decay_coefficients = \
            cy.gradient_partial(event_type, float(base_rate[0]), impact_coefficients[:, :, 0], decay_coefficients[:, :, 0],
                                number_of_event_types,
                                number_of_states, times, events, states, float(time_start), float(time_end))
        a = np.zeros((number_of_event_types, number_of_states, 1))
        b = np.zeros((number_of_event_types, number_of_states, 1))
        a[:, :, 0] = g_impact_coefficients
        b[:, :, 0] = g_decay_coefficients
        return self.parameters_to_array([g_base_rate], a, b)

    'Miscellaneous tools'

    def intensities_of_events_at_times(self, compute_times, times, events, states):
        """
        Computes the intensities at the `compute_times` given a realisation of the state-dependent Hawkes process.

        :type compute_times: 1D numpy array of float
        :param compute_times: the times at which the intensities will be computed.
        :type times: 1D numpy array of float
        :param times: the times at which events occur.
        :type events: 1D numpy array of int
        :param events: the sequence of event types, `events[n]` is the event type of the `n` th event.
        :type states: 1D numpy array of int
        :param states: the sequence of states, `states[n]` is the new state of the system following the `n` th event.
        :rtype: 1D numpy array of float, 2D numpy array of float
        :return: the first array contains both the `compute_times` and the event times of the given realisation,
                 in increasing order. Note that the event times are put there twice:
                 the intensities are computed just before and right after the event times.
                 The second array gives the intensities at the times of the first array.
                 `array2[e,n]` is the intensity of events of type `e` at time `array1[n]`.
        """
        'Start/end time and first index of event times'
        # time at which we start to compute the intensity
        time_start = compute_times[0]
        time_end = compute_times[-1]
        next_event_time_index = bisect.bisect_right(
            times, time_start)  # first event time occurring in between
        next_event_time = time_end + 1  # in case there is no event after 'time_start'
        # i.e. there is an event time after
        if next_event_time_index < len(times):
            next_event_time = times[next_event_time_index]
        '''Initialise the partial sums S_{e',x',e} that will allow use to compute the intensity recursively'''
        partial_sums = np.zeros(
            (self.number_of_event_types, self.number_of_states, self.number_of_event_types))
        for n in range(next_event_time_index):
            time = times[n]
            event = events[n]
            state = states[n]
            for e in range(self.number_of_event_types):
                alpha = self.impact_coefficients[event, state, e]
                beta = self.decay_coefficients[event, state, e]
                partial_sums[event, state, e] += alpha * \
                    math.exp(-beta * (time_start - time))
        'Create aggregated sequence of times, events, and states (event times + compute times)'
        times_aggregated = []
        events_aggregated = []
        states_aggregated = []
        for t in compute_times:
            if t < next_event_time:
                times_aggregated.append(t)
                # -1 means that it is not an event time
                events_aggregated.append(-1)
                states_aggregated.append(-1)
            elif t > next_event_time:
                while next_event_time < t:
                    'We add it twice so that the intensity is computed just before and right after'
                    'This is to get nice looking figures when plotting the intensities'
                    times_aggregated.append(next_event_time)
                    times_aggregated.append(next_event_time)
                    event = events[next_event_time_index]
                    state = states[next_event_time_index]
                    events_aggregated.append(-1)
                    events_aggregated.append(event)
                    states_aggregated.append(-1)
                    states_aggregated.append(state)
                    next_event_time_index += 1
                    next_event_time = time_end + 1  # in case there is no event after 'time_start'
                    # i.e. there is an event time after
                    if next_event_time_index < len(times):
                        next_event_time = times[next_event_time_index]
                'Now add the time t'
                times_aggregated.append(t)
                events_aggregated.append(-1)
                states_aggregated.append(-1)
        'Compute the intensities at the aggregated times'
        number_of_times = len(times_aggregated)
        result_intensities = np.zeros(
            (self.number_of_event_types, number_of_times))
        intensities = self.intensities_of_events(partial_sums)
        for e in range(self.number_of_event_types):
            result_intensities[e, 0] = intensities[e]
        for n in range(1, number_of_times):
            time_increment = times_aggregated[n] - times_aggregated[n-1]
            event = events_aggregated[n]
            'Update partial sums: time decay effect'
            if time_increment > 0:
                for e1 in range(self.number_of_event_types):
                    for x in range(self.number_of_states):
                        for e2 in range(self.number_of_event_types):
                            beta = self.decay_coefficients[e1, x, e2]
                            partial_sums[e1, x,
                                         e2] *= math.exp(-beta * time_increment)
            'Update partial sums: impact of new event'
            if event >= 0:
                state = states_aggregated[n]
                for e2 in range(self.number_of_event_types):
                    alpha = self.impact_coefficients[event, state, e2]
                    partial_sums[event, state, e2] += alpha
            'Compute intensities and save'
            intensities = self.intensities_of_events(partial_sums)
            for e1 in range(self.number_of_event_types):
                result_intensities[e1, n] = intensities[e1]
        return times_aggregated, result_intensities

    def compute_partial_sums(self, times, events, states, time_end,
                             initial_partial_sums=None, time_initial_condition=None):
        r"""
        Computes the partial sums

        .. math::

            S_{e'xe}(-\infty,T] := \alpha_{e'xe} \sum_{n : t^{e'x}_n \leq T} \exp(-\beta_{e'xe}(T-t^{e'x}_n)),

        where `e` and `e'` are event types, `x` is a possible state, :math:`t^{e'x}_n` is the time when the `n` th
        event of type `e'` after which the state is `x` occurred.
        The collection of partial sums :math:`(S_{e'xe})` at time :math:`T` encodes the history of the
        state-dependent Hawkes process up to time :math:`T`. It can for example be used to simulate the process from
        time :math:`T`. In :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.simulate`, one can pass the partial
        sums as an initial condition instead of the entire history.

        :type times: 1D numpy array of float
        :param times: the times at which events occur.
        :type events: 1D numpy array of int
        :param events: the sequence of event types, `events[n]` is the event type of the `n` th event.
        :type states: 1D numpy array of int
        :param states: the sequence of states, `states[n]` is the new state of the system following the `n` th event.
        :type time_end: float
        :param time_end: :math:`T`, the time up to which the partial sums are computed.
        :type initial_partial_sums: 3D numpy array of float
        :param initial_partial_sums: an initial condition (events that occur before `times`) can be given implicitly
                                     as partial sums.
        :type time_initial_condition: float
        :param time_initial_condition: the time up to which the partial sums given as an initial condition were
                                       computed.
        :rtype: 3D numpy array of float
        :return: the partial sums, `array[e', x, e]` corresponds to :math:`S_{e'x'e}`.
        """
        'Compute contribution of the given events'
        partial_sums = np.zeros(
            (self.number_of_event_types, self.number_of_states, self.number_of_event_types))
        index_end = bisect.bisect_right(times, time_end)
        for n in range(index_end):
            time = times[n]
            event = events[n]
            state = states[n]
            partial_sums[event, state,
                         :] += np.exp(- self.decay_coefficients[event, state, :] * (time_end - time))
        partial_sums = np.multiply(partial_sums, self.impact_coefficients)
        'Add contribution of the given initial condition'
        if np.shape(initial_partial_sums) != () and time_initial_condition != None:
            partial_sums += np.multiply(np.exp(- self.decay_coefficients * (time_end - time_initial_condition)),
                                        initial_partial_sums)
        return partial_sums

    def intensity_of_event(self, event_type, partial_sums):
        r"""
        Computes the intensity of events of type `event_type`, given the partial sums :math:`S_{e'xe}`.

        :type event_type: int
        :param event_type: the event type for which the intensity is computed.
        :type partial_sums: 3D numpy array of float
        :param partial_sums: the partial sums :math:`S_{e',x',e}` at the considered time.
        :rtype: float
        :return: the value of :math:`\lambda_e` at the considered time.
        """
        result = self.base_rates[event_type]
        for e in range(self.number_of_event_types):
            for x in range(self.number_of_states):
                result += partial_sums[e, x, event_type]
        return result

    def intensities_of_events(self, partial_sums):
        r"""
        Computes the intensities, given the partial sums :math:`S_{e'xe}`.

        :type partial_sums: 3D numpy array of float
        :param partial_sums: the partial sums :math:`S_{e',x',e}` at the considered time.
        :rtype: 1D numpy array of float
        :return: the intensities, `array[e]` is the value of :math:`\lambda_e` at the considered time.
        """
        result = np.zeros(self.number_of_event_types)
        for e in range(self.number_of_event_types):
            result[e] = self.intensity_of_event(e, partial_sums)
        return result

    @staticmethod
    def parameters_to_array(base_rates, impact_coefficients, decay_coefficients):
        r"""
        Puts the model parameters :math:`(\nu, \alpha, \beta)` into a one dimensional array.

        :type base_rates: 1D numpy array
        :param base_rates: the collection :math:`(\nu_{e})`.
        :type impact_coefficients: 3D numpy array
        :param impact_coefficients: the collection :math:`(\alpha_{e'xe})`.
        :type decay_coefficients: 3D numpy array
        :param decay_coefficients: the collection :math:`(\beta_{e'xe})`.
        :rtype: 1D numpy array
        :return: the parameters put into a single 1D array.
        """
        s = np.shape(impact_coefficients)
        number_of_event_types_1 = s[0]
        number_of_states = s[1]
        number_of_event_types_2 = s[2]
        result =\
            np.zeros(number_of_event_types_2 + 2 * number_of_event_types_1 *
                     number_of_event_types_2 * number_of_states)
        for n in range(number_of_event_types_2):
            result[n] = base_rates[n]
        for i in range(number_of_event_types_1):
            for j in range(number_of_states):
                for k in range(number_of_event_types_2):
                    index = number_of_event_types_2 + j * \
                        number_of_event_types_1 * number_of_event_types_2
                    index += i * number_of_event_types_2 + k
                    result[index] = impact_coefficients[i, j, k]
        for i in range(number_of_event_types_1):
            for j in range(number_of_states):
                for k in range(number_of_event_types_2):
                    index = number_of_event_types_2
                    index += number_of_event_types_1 * number_of_event_types_2 * number_of_states
                    index += j * number_of_event_types_1 * number_of_event_types_2
                    index += i * number_of_event_types_2 + k
                    result[index] = decay_coefficients[i, j, k]
        return result

    @staticmethod
    def array_to_parameters(array, number_of_event_types_1, number_of_states, number_of_event_types_2=0):
        r"""
        Retrieves the parameters :math:`(\nu, \alpha, \beta)` from a 1D array.
        It is NOT assumed that the length of the 1st and 3rd dimensions of the arrays :math:`(\alpha_{e'xe})` and
        :math:`(\beta_{e'xe})` are equal. For instance, in
        :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.log_likelihood_of_events_partial`,
        only of subgroup of the parameters :math:`(\nu, \alpha, \beta)` are required.

        :type array: 1D numpy array
        :param array: an array containing the parameters :math:`(\nu, \alpha, \beta)`.
        :type number_of_event_types_1: int
        :param number_of_event_types_1: length of the first dimension of :math:`(\alpha_{e'xe})` and
                                        :math:`(\beta_{e'xe})`.
        :type number_of_states: int
        :param number_of_states: number of possible states in the state-dependent Hawkes process,
                                 length of the second dimension.
        :type number_of_event_types_2: int
        :param number_of_event_types_2: length of the third dimension of :math:`(\alpha_{e'xe})` and
                                        :math:`(\beta_{e'xe})`. It is also the length of :math:`\nu`.
                                        If set to zero, we assume that `array` contains ALL the parameters
                                        and not only a subgroup, meaning that
                                        1st and 3rd dimensions of the arrays :math:`(\alpha_{e'xe})` and
                                        :math:`(\beta_{e'xe})` are equal.
        :rtype: 1D numpy array, 3D numpy array, 3D numpy array
        :return: the parameters :math:`(\nu, \alpha, \beta)`.
        """
        if number_of_event_types_2 == 0:
            number_of_event_types_2 = number_of_event_types_1
        base_rates = np.zeros(number_of_event_types_2)
        for n in range(number_of_event_types_2):
            base_rates[n] = array[n]
        impact_coefficients = np.zeros(
            (number_of_event_types_1, number_of_states, number_of_event_types_2))
        for i in range(number_of_event_types_1):
            for j in range(number_of_states):
                for k in range(number_of_event_types_2):
                    index = number_of_event_types_2 + j * \
                        number_of_event_types_1 * number_of_event_types_2
                    index += i * number_of_event_types_2 + k
                    impact_coefficients[i, j, k] = array[index]
        decay_coefficients = np.zeros(
            (number_of_event_types_1, number_of_states, number_of_event_types_2))
        for i in range(number_of_event_types_1):
            for j in range(number_of_states):
                for k in range(number_of_event_types_2):
                    index = number_of_event_types_2
                    index += number_of_event_types_1 * number_of_event_types_2 * number_of_states
                    index += j * number_of_event_types_1 * number_of_event_types_2
                    index += i * number_of_event_types_2 + k
                    decay_coefficients[i, j, k] = array[index]
        return base_rates, impact_coefficients, decay_coefficients

    @staticmethod
    def transition_matrix_to_string(transition_probabilities):
        r"""
        Transforms the transition probabilities :math:`\phi` to a string that can be printed in the console or a
        text file.

        :type transition_probabilities: 3D numpy array
        :param transition_probabilities: the transition probabilities :math:`\phi`.
        :rtype: string
        :return: sequence of transition probability matrices :math:`(\phi_e)`, one per event type `e`.
        """
        number_of_states = len(transition_probabilities)
        number_of_events = len(transition_probabilities[0])
        result = ''
        for e in range(number_of_events):
            m = np.zeros((number_of_states, number_of_states))
            for x1 in range(number_of_states):
                for x2 in range(number_of_states):
                    m[x1][x2] = transition_probabilities[x1][e][x2]
            result += str(m)
            if e < number_of_events - 1:
                result += '\n'
        return result

    @staticmethod
    def impact_coefficients_to_string(impact_coefficients):
        r"""
        Transforms the impact coefficients :math:`\alpha` to a string that can be printed in the console or a text file.

        :type impact_coefficients: 3D numpy array
        :param impact_coefficients: the impact coefficients :math:`\alpha`.
        :rtype: string
        :return: sequence of matrices :math:`(\alpha_{\cdot x \cdot})`, one per possible state `x`.
        """
        return HybridHawkesExp.transition_matrix_to_string(impact_coefficients)

    @staticmethod
    def decay_coefficients_to_string(decay_coefficients):
        r"""
        Transforms the decay coefficients :math:`\beta` to a string that can be printed in the console or a text file.

        :type decay_coefficients: 3D numpy array
        :param decay_coefficients: the decay coefficients :math:`\beta`.
        :rtype: string
        :return: sequence of matrices :math:`(\beta_{\cdot x \cdot})`, one per possible state `x`.
        """
        return HybridHawkesExp.transition_matrix_to_string(decay_coefficients)

    @staticmethod
    def proportion_of_events_and_states(events, states, number_of_event_types, number_of_states):
        r"""
        Computes the empirical distribution of the events and states.

        :type events: 1D numpy array of int
        :param events: the sequence of event types, `events[n]` is the event type of the `n` th event.
        :type states: 1D numpy array of int
        :param states: the sequence of states, `states[n]` is the new state of the system following the `n` th event.
        :type number_of_event_types: int
        :param number_of_event_types: number of different event types.
        :type number_of_states: int
        :param number_of_states: number of possible states.
        :rtype: 2D numpy array of float
        :return: `array[e,x]` is the percentage of events of type `e` after which the state is `x`.
        """
        proportion_events_states = np.zeros(
            (number_of_event_types, number_of_states))
        size = len(events)
        for n in range(size):
            e = events[n]
            x = states[n]
            proportion_events_states[e, x] += 1
        proportion_events_states = np.divide(proportion_events_states, size)
        return proportion_events_states

    def generate_base_rates_labels(self):
        r"""
        Produces labels for the base rates :math:`\nu`.
        This uses the events labels of the model.

        :rtype: list of string
        :return: `list[e]` returns a label for :math:`\nu_e`.
        """
        labels = []
        for e in range(self.number_of_event_types):
            l = r'$\nu_{' + self.events_labels[e] + '}$'
            labels.append(l)
        return labels

    def generate_impact_coefficients_labels(self):
        r"""
        Produces labels for the impact coefficients :math:`\alpha`.
        This uses the events and states labels of the model.

        :rtype: list of list of list of string
        :return: `list[e'][x][e]` returns a label for :math:`\alpha_{e'xe}`.
        """
        labels = []
        for e1 in range(self.number_of_event_types):
            l1 = []
            for x in range(self.number_of_states):
                l2 = []
                for e2 in range(self.number_of_event_types):
                    s = r'$\alpha_{' + self.events_labels[e1]
                    s += r' \rightarrow ' + self.events_labels[e2] + '}('
                    s += self.states_labels[x] + ')$'
                    l2.append(s)
                l1.append(l2)
            labels.append(l1)
        return labels

    def generate_decay_coefficients_labels(self):
        r"""
        Produces labels for the decay coefficients :math:`\beta`.
        This uses the events and states labels of the model.

        :rtype: list of list of list of string
        :return: `list[e'][x][e]` returns a label for :math:`\beta_{e'xe}`.
        """
        labels = []
        for e1 in range(self.number_of_event_types):
            l1 = []
            for x in range(self.number_of_states):
                l2 = []
                for e2 in range(self.number_of_event_types):
                    s = r'$\beta_{' + self.events_labels[e1]
                    s += r' \rightarrow ' + self.events_labels[e2] + '}('
                    s += self.states_labels[x] + ')$'
                    l2.append(s)
                l1.append(l2)
            labels.append(l1)
        return labels

    def generate_product_labels(self):
        r"""
        Produces labels for all the couples of events types and possible states.
        This uses the events and states labels of the model.

        :rtype: list of strings
        :return: the label for the couple `(e, x)` is given by by `list[n]` where
                 :math:`n = x + e \times \mbox{number_of_event_types}`.
        """
        r = []
        for e in self.events_labels:
            for x in self.states_labels:
                r.append(e + ', ' + x)
        return r
