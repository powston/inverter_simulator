import pandas as pd
from typing import Any, Tuple, Callable
import logging
from inverter_simulator.battery import Battery

logger = logging.getLogger(__name__)

class InverterSimulator:
    DEFAULT_INTERVAL = 5

    def __init__(self, system: pd.DataFrame, control_function: Callable, **kwargs: Any):
        self.system = system
        self.control_function = control_function
        self._init_parameters(kwargs)
        self._init_simulation_data()

    def _init_parameters(self, kwargs: dict) -> None:
        battery_capacity = kwargs.get('battery_capacity', 10000)
        charge_rate = kwargs.get('charge_rate', 4600)
        initial_charge = kwargs.get('battery_charge', battery_capacity / 2)
        battery_loss = kwargs.get('battery_loss', 5)
        self.battery = Battery(battery_capacity, charge_rate, initial_charge, battery_loss)

        self.grid_limit = kwargs.get('grid_limit', self._calculate_grid_limit())
        self.tariff = kwargs.get('tariff', '6900')
        self.network = kwargs.get('network', 'Energex')
        self.state = kwargs.get('state', 'QLD')
        self.max_ppv_power = kwargs.get('max_ppv_power', 5000)
        self.interval = kwargs.get('interval', 5)
        self.timezone_str = kwargs.get('timezone_str', 'Australia/Brisbane')
        self.latitude = kwargs.get('latitude', None)
        self.longitude = kwargs.get('longitude', None)
        if 'sim_cost' not in self.system.columns:
            self.system['sim_cost'] = 0.0
        self.algo_sim_usage = self.system['sim_cost'].sum()

    def _init_simulation_data(self) -> None:
        self.current_interval = self.system.index[0]
        self.solar_powers = []
        self.charges = []
        self.discharges = []
        self.battery_charges = []
        self.battery_socs = []
        self.actions = []
        self.reasons = []
        self.balances = []
        self.sim_costs = []
        self.power_from_grid = []
        self.power_to_grid = []

    def _calculate_grid_limit(self) -> int:
        return self.system['house_power'].max() * 2

    def reset(self) -> None:
        self._init_simulation_data()
        self.battery.reset()

    def is_done(self) -> bool:
        return self.current_interval == self.system.index[-1]

    def get_state(self) -> dict:
        row = self.system.loc[self.current_interval]
        return self._create_state_dict(row)

    def _create_state_dict(self, row: pd.Series) -> dict:
        return {
            'battery_charge': self.battery.charge,
            'battery_soc': self.battery.soc,
            'solar_power': row['solar_power'],
            'house_power': row['house_power'],
            'rrp': row['rrp'],
            'forecast': row['forecast'],
            'interval': self.interval,
            'current_interval': self.current_interval,
            'grid_limit': self.grid_limit,
            'tariff': self.tariff,
            'network': self.network,
            'state': self.state,
            'max_ppv_power': self.max_ppv_power,
            'timezone_str': self.timezone_str,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'spot_to_tariff': self.spot_to_tariff,
            'spot_to_feed_in_tariff': self.spot_to_feed_in_tariff,
            'get_daily_fee': self.get_daily_fee,
            'sim_cost': self.algo_sim_usage
        }

    def apply_action(self, inverter_action: str) -> None:
        row = self.system.loc[self.current_interval]
        self._process_interval(self.current_interval, row, inverter_action)
        self.current_interval += pd.Timedelta(minutes=self.interval)

    def _process_interval(self, index: pd.Timestamp, row: pd.Series, action: str, reason: str) -> None:
        house_power, solar_power, buy_price, sell_price = self._get_params(index, row)
        pv_surplus = max(0, solar_power - house_power)
        charge, discharge = self._calculate_charge_discharge(action, pv_surplus)
        self._update_simulation_data(action, reason, solar_power, charge, discharge, house_power, buy_price, sell_price)

    def _get_params(self, index: pd.Timestamp, row: pd.Series) -> Tuple[float, float, float]:
        return row['house_power'], row['solar_power'], row['buy_price'], row['sell_price']

    def _calculate_charge_discharge(self, action: str, pv_surplus: float) -> Tuple[float, float]:
        if action == 'charge':
            charge = self.battery.charge_battery(pv_surplus, self.interval)
            discharge = 0
        elif action == 'discharge':
            charge = 0
            discharge = self.battery.discharge_battery(-pv_surplus, self.interval)
        elif action == 'auto':
            if pv_surplus > 0:
                charge = self.battery.charge_battery(pv_surplus, self.interval)
                discharge = 0
            else:
                charge = 0
                discharge = self.battery.discharge_battery(-pv_surplus, self.interval)
        elif action == 'stopped':
            charge = discharge = 0
        elif action == 'export':
            charge = 0
            discharge = self.battery.discharge_battery(self.battery.charge_rate, self.interval)
        elif action == 'import':
            charge = self.battery.charge_battery(self.battery.charge_rate, self.interval)
            discharge = 0
        else:
            charge = discharge = 0
        return charge, discharge

    def _update_simulation_data(self, action: str, reason: str, solar_power: float, charge: float, discharge: float,
                                house_power: float, buy_price: float, sell_price: float) -> None:
        self.solar_powers.append(solar_power)
        self.charges.append(charge)
        self.discharges.append(discharge)
        self.battery_charges.append(self.battery.charge)
        self.battery_socs.append(self.battery.soc)
        
        self.actions.append(action)
        self.reasons.append(reason)

        balance = solar_power - house_power - charge + discharge
        self.grid_power = balance
        self.balances.append(balance)

        kwh_balance = balance / 12000
        if kwh_balance < 0:
            self.power_from_grid.append(-kwh_balance)
            self.power_to_grid.append(0)
            self.last_cost = buy_price * -kwh_balance
        else:
            self.power_from_grid.append(0)
            self.power_to_grid.append(kwh_balance)
            self.last_cost = -sell_price * kwh_balance
        self.sim_costs.append(self.last_cost)

    def run_simulation(self) -> Tuple[float, pd.DataFrame]:
        for index, row in self.system.iterrows():
            self._process_interval(index, row, *self.control_function(index, **row))
        self._calculate_final_metrics()
        return self.algo_sim_usage, self.system

    def _calculate_final_metrics(self) -> None:
        self.system['charge'] = self.charges
        self.system['discharge'] = self.discharges
        self.system['action'] = self.actions
        self.system['reason'] = self.reasons
        self.system['battery_charge'] = self.battery_charges
        self.system['battery_soc'] = self.battery_socs
        self.system['balance'] = self.balances
        self.system['sim_cost'] = self.sim_costs
        self.system['solar_power'] = self.solar_powers
        self.system['Power from grid'] = self.power_from_grid
        self.system['Power to grid'] = self.power_to_grid
        self.system['Energy from grid'] = self.system['Power from grid'] * self.interval / 60
        self.system['Energy to grid'] = self.system['Power to grid'] * self.interval  / 60
        self.algo_sim_usage = self.system['sim_cost'].sum()

def sim_inverter(system: pd.DataFrame, control_function: Callable, **kwargs: Any) -> Tuple[float, pd.DataFrame]:
    sim = InverterSimulator(system, control_function, **kwargs)
    return sim.run_simulation()