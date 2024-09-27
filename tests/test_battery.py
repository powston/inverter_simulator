import unittest
from inverter_simulator.battery import Battery

class TestBattery(unittest.TestCase):
    def setUp(self):
        self.battery = Battery(capacity=10000, charge_rate=4600, initial_charge=5000)

    def test_initialization(self):
        self.assertEqual(self.battery.capacity, 10000)
        self.assertEqual(self.battery.charge_rate, 4600)
        self.assertEqual(self.battery.charge, 5000)
        self.assertEqual(self.battery.loss_rate, 5)

    def test_soc(self):
        self.assertEqual(self.battery.soc, 50)

    def test_charge_battery(self):
        charged = self.battery.charge_battery(1000)
        self.assertAlmostEqual(charged, 1000)
        self.assertAlmostEqual(self.battery.charge, 5079.17, places=2)

    def test_charge_battery_limit(self):
        charged = self.battery.charge_battery(6000)
        self.assertAlmostEqual(charged, 4600)
        self.assertAlmostEqual(self.battery.charge, 5364.17, places=2)

    def test_discharge_battery(self):
        discharged = self.battery.discharge_battery(1000)
        self.assertAlmostEqual(discharged, 1000)
        # 1000W over 5 mintes is 12th of 1000kW = 83.3Wh and 5% is lost
        self.assertAlmostEqual(self.battery.charge, 5000-(83.3333 * 1.05), places=2)

    def test_negative_discharge_battery(self):
        discharged = self.battery.discharge_battery(-1000)
        self.assertAlmostEqual(discharged, 0)
        # should do nothing
        self.assertAlmostEqual(self.battery.charge, 5000, places=2)

    def test_discharge_battery_limit(self):
        discharged = self.battery.discharge_battery(6000)
        self.assertAlmostEqual(discharged, 4600)
        self.assertAlmostEqual(self.battery.charge, 4597.5, places=2)

    def test_reset(self):
        self.battery.charge = 8000
        self.battery.reset()
        self.assertEqual(self.battery.charge, 5000)

if __name__ == '__main__':
    unittest.main()