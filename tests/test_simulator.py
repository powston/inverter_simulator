import unittest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime, timedelta
from inverter_simulator.simulator import InverterSimulator
from inverter_simulator.battery import Battery

class TestInverterSimulator(unittest.TestCase):

    def setUp(self):
        self.mock_system = pd.DataFrame({
            'house_power': [1000, 2000, 3000],
            'solar_power': [2000, 3000, 4000],
            'buy_price': [20, 30, 40],
            'sell_price': [10, 20, 30],
            'rrp': [100, 200, 300],
            'forecast': [[100, 200, 300], [200, 300, 400], [300, 400, 500]]
        }, index=[datetime(2023, 1, 1) + timedelta(hours=i) for i in range(3)])
        
        self.mock_control_function = Mock(return_value=['auto', 'always auto'])
        self.simulator = InverterSimulator(self.mock_system, self.mock_control_function)

    def test_initialization(self):
        self.assertIsInstance(self.simulator.battery, Battery)
        self.assertEqual(self.simulator.battery.capacity, 10000)
        self.assertEqual(self.simulator.battery.charge_rate, 4600)
        self.assertEqual(self.simulator.grid_limit, 6000)

    def test_reset(self):
        self.simulator.battery.charge = 8000
        self.simulator.current_interval = self.simulator.system.index[-1]
        self.simulator.reset()
        self.assertEqual(self.simulator.current_interval, self.simulator.system.index[0])
        self.assertEqual(self.simulator.battery.charge, self.simulator.battery.capacity / 2)

    def test_is_done(self):
        self.assertFalse(self.simulator.is_done())
        self.simulator.current_interval = self.simulator.system.index[-1]
        self.assertTrue(self.simulator.is_done())

    @patch('inverter_simulator.simulator.InverterSimulator._create_state_dict')
    def test_get_state(self, mock_create_state_dict):
        mock_create_state_dict.return_value = {'test': 'state'}
        state = self.simulator.get_state()
        self.assertEqual(state, {'test': 'state'})
        mock_create_state_dict.assert_called_once()

    @patch('inverter_simulator.simulator.InverterSimulator._process_interval')
    def test_apply_action(self, mock_process_interval):
        initial_interval = self.simulator.current_interval
        self.simulator.apply_action('test_action')
        mock_process_interval.assert_called_once()
        self.assertEqual(self.simulator.current_interval, initial_interval + timedelta(minutes=self.simulator.interval))

    def test_process_interval(self):
        index = self.simulator.system.index[0]
        row = self.simulator.system.loc[index]
        self.simulator._process_interval(index, row, 'auto', 'always auto')
        self.assertEqual(len(self.simulator.solar_powers), 1)
        self.assertEqual(len(self.simulator.charges), 1)
        self.assertEqual(len(self.simulator.discharges), 1)
        self.assertEqual(len(self.simulator.battery_charges), 1)
        self.assertEqual(len(self.simulator.actions), 1)
        self.assertEqual(len(self.simulator.sim_costs), 1)

    def test_calculate_charge_discharge(self):
        test_cases = [
            ('charge', 1000, 1000, 0),
            ('discharge', -1000, 0, 1000),
            ('auto', 1000, 1000, 0),
            ('auto', -1000, 0, 1000),
            ('stopped', 1000, 0, 0),
        ]
        for action, pv_surplus, expected_charge, expected_discharge in test_cases:
            with self.subTest(action=action, pv_surplus=pv_surplus):
                charge, discharge = self.simulator._calculate_charge_discharge(action, pv_surplus)
                self.assertAlmostEqual(charge, expected_charge)
                self.assertAlmostEqual(discharge, expected_discharge)

    @patch('inverter_simulator.simulator.InverterSimulator._calculate_final_metrics')
    def test_run_simulation(self, mock_calculate_final_metrics):
        result = self.simulator.run_simulation()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        mock_calculate_final_metrics.assert_called_once()

    def test_calculate_final_metrics(self):
        self.simulator.run_simulation()
        expected_columns = [
            'solar_power', 'charge', 'discharge', 'action', 'battery_charge', 'battery_soc',
            'balance', 'Power from grid', 'Power to grid', 'Energy from grid', 'Energy to grid', 'sim_cost'
        ]
        for column in expected_columns:
            self.assertIn(column, self.simulator.system.columns)

    def test_integration(self):
        simulator = InverterSimulator(self.mock_system, self.mock_control_function)
        retail_usage, updated_system = simulator.run_simulation()
        self.assertIsInstance(retail_usage, float)
        self.assertIsInstance(updated_system, pd.DataFrame)
        self.assertEqual(len(updated_system), len(self.mock_system))
        expected_columns = ['solar_power', 'charge', 'discharge', 'action', 'battery_charge', 'battery_soc', 'balance', 'sim_cost']
        for column in expected_columns:
            self.assertIn(column, updated_system.columns)

if __name__ == '__main__':
    unittest.main()