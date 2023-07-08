import unittest
import numpy as np
import scipy
from mpoints import hybrid_hawkes_exp


class HybridHawkesExpTest(unittest.TestCase):
    def setUp(self):
        number_of_event_types: int = np.random.randint(low=1, high=3)
        de = number_of_event_types
        number_of_states: int = np.random.randint(low=1, high=3)
        dx = number_of_states
        events_labels = [chr(65 + n) for n in range(number_of_event_types)]
        states_labels = [chr(48 + n) for n in range(number_of_states)]
        _phis = [np.eye(dx) + scipy.sparse.random(dx, dx,
                                                  density=.50).A for _ in range(de)]
        _phis = [_phi / 10*np.sum(_phi, axis=1, keepdims=True)
                 for _phi in _phis]
        _phis = [np.expand_dims(_phi, axis=1) for _phi in _phis]
        phis = np.concatenate(_phis, axis=1)
        nus = np.random.uniform(low=0., high=1., size=(de,))
        alphas = 10 * \
            scipy.sparse.random(de, dx*de, density=.50).A.reshape(de, dx, de)
        betas = np.ones((de, dx, de), dtype=float)
        self.number_of_event_types = number_of_event_types
        self.number_of_states = number_of_states
        self.events_labels = events_labels
        self.states_labels = states_labels
        self.nus = nus
        self.alphas = alphas
        self.betas = betas
        self.phis = phis

    def test_init_and_setters(self):
        hhe = hybrid_hawkes_exp.HybridHawkesExp(
            self.number_of_event_types,
            self.number_of_states,
            self.events_labels,
            self.states_labels,
        )
        self.assertEqual(self.number_of_event_types, hhe.number_of_event_types)
        self.assertEqual(self.number_of_states, hhe.number_of_states)
        self.assertEqual(self.phis.shape, hhe.transition_probabilities.shape)
        self.assertEqual(self.nus.shape, hhe.base_rates.shape)
        self.assertEqual(self.alphas.shape, hhe.impact_coefficients.shape)
        self.assertEqual(self.betas.shape, hhe.decay_coefficients.shape)

        hhe.set_transition_probabilities(self.phis)
        hhe.set_hawkes_parameters(self.nus, self.alphas, self.betas)
        self.assertTrue(np.allclose(self.phis, hhe.transition_probabilities))
        self.assertTrue(np.allclose(self.nus, hhe.base_rates))
        self.assertTrue(np.allclose(self.alphas, hhe.impact_coefficients))
        self.assertTrue(np.allclose(self.betas, hhe.decay_coefficients))

    def test_flatten_parameters(self):
        flat_params = hybrid_hawkes_exp.HybridHawkesExp.parameters_to_array(
            self.nus,
            self.alphas,
            self.betas
        )
        nus, alphas, betas = hybrid_hawkes_exp.HybridHawkesExp.array_to_parameters(
            flat_params,
            self.number_of_event_types,
            self.number_of_states,
        )
        self.assertTrue(
            np.allclose(nus, self.nus)
        )
        self.assertTrue(
            np.allclose(alphas, self.alphas)
        )
        self.assertTrue(
            np.allclose(betas, self.betas)
        )

    def _instantiate_hhe(self):
        hhe = hybrid_hawkes_exp.HybridHawkesExp(
            self.number_of_event_types,
            self.number_of_states,
            self.events_labels,
            self.states_labels,
        )
        return hhe

    def test_simulate(self):
        # Simulate
        hhe = self._instantiate_hhe()
        hhe.set_transition_probabilities(self.phis)
        hhe.set_hawkes_parameters(self.nus, self.alphas, self.betas)
        time_start = 0.
        time_end = 7200.
        times, events, states = hhe.simulate(
            time_start, time_end, max_number_of_events=1000000)
        self.assertTrue(
            np.all(states < self.number_of_states)
        )
        self.assertTrue(
            np.all(events < self.number_of_event_types)
        )
        self.assertTrue(np.all(np.diff(times) > 0.))

    @unittest.skip
    def test_simulate_and_estimate(self):
        # Simulate
        hhe = self._instantiate_hhe()
        hhe.set_transition_probabilities(self.phis)
        hhe.set_hawkes_parameters(self.nus, self.alphas, self.betas)
        time_start = 0.
        time_end = 7200.
        times, events, states = hhe.simulate(
            time_start, time_end, max_number_of_events=1000000)
        self.assertTrue(
            np.all(states < self.number_of_states)
        )
        self.assertTrue(
            np.all(events < self.number_of_event_types)
        )
        self.assertTrue(np.all(np.diff(times) > 0.))

        # Estimate
        estimator = self._instantiate_hhe()
        guess = estimator.parameters_to_array(
            self.nus, self.alphas, self.betas)
        guess *= np.random.uniform(low=.85, high=1.15, size=guess.shape)
        guess += np.random.uniform(low=0,  high=.001, size=guess.shape)
        opt_result, initial_guess, initial_guess_kind = estimator.estimate_hawkes_parameters(
            times, events, states, time_start, time_end,
            given_guesses=[guess],
        )
        mle_estimate = opt_result.x
        nus, alphas, betas = estimator.array_to_parameters(
            mle_estimate, estimator.number_of_event_types, estimator.number_of_states)
        print(f'Estimated nus:\n{nus}')
        print(f'True nus:\n{self.nus}')
        print(f'Estimated alphas:\n{alphas}')
        print(f'True alphas:\n{self.alphas}')
        print(f'Estimated betas:\n{betas}')
        print(f'True betas:\n{self.betas}')
        self.assertTrue(
            np.allclose(
                nus, self.nus, atol=1e-5, rtol=1e-4)
        )
        self.assertTrue(
            np.allclose(
                alphas, self.alphas, atol=1e-5, rtol=1e-4)
        )
        self.assertTrue(
            np.allclose(
                betas, self.betas, atol=1e-5, rtol=1e-4)
        )


if __name__ == '__main__':
    unittest.main()
