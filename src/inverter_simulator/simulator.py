import pandas as pd
from typing import Any, Tuple, Callable
import logging
from astral import LocationInfo
from astral.sun import sun
from inverter_simulator.battery import Battery
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

class InverterSimulator:
    DEFAULT_INTERVAL = 5

    def __init__(self, system: pd.DataFrame, control_function: Callable, **kwargs: Any):
        self.system = system
        # remove duplcates and keep the last value
        self.system = self.system[~self.system.index.duplicated(keep='last')]
        self.control_function = control_function
        self._init_parameters(kwargs)
        self._init_simulation_data()

    def _init_parameters(self, kwargs: dict) -> None:
        battery_capacity = kwargs.get('battery_capacity', 10000)
        charge_rate = kwargs.get('charge_rate', 4600)
        initial_charge = kwargs.get('battery_charge', battery_capacity / 2)
        battery_loss = kwargs.get('battery_loss', 5)
        self.min_soc = kwargs.get('min_soc', 10)
        self.battery = Battery(capacity=battery_capacity, charge_rate=charge_rate, min_soc=self.min_soc,
                               initial_charge=initial_charge, loss_rate=battery_loss)

        self.grid_limit = kwargs.get('grid_limit', self._calculate_grid_limit())
        self.tariff = kwargs.get('tariff', '6900')
        self.network = kwargs.get('network', 'energex')
        self.state = kwargs.get('state', 'QLD')
        self.max_ppv_power = kwargs.get('max_ppv_power', 5000)
        self.interval = kwargs.get('interval', self.DEFAULT_INTERVAL)
        self.timezone_str = kwargs.get('timezone_str', 'Australia/Brisbane')
        self.location = kwargs.get('location', 'Brisbane')
        self.latitude = kwargs.get('latitude', -27.4698)
        self.longitude = kwargs.get('longitude', 153.0251)
        self.daily_fee = kwargs.get('daily_fee', 1)
        self.spot_to_tariff = kwargs.get('spot_to_tariff', lambda x, y, z, a: a / 10)
        self.spot_to_feed_in_tariff = kwargs.get('spot_to_feed_in_tariff', lambda x: x / 10)
        if 'sim_cost' not in self.system.columns:
            self.system['sim_cost'] = 0.0
        self.algo_sim_usage = self.system['sim_cost'].sum()

    def _init_simulation_data(self) -> None:
        self.current_interval = self.system.index[0]
        self.grid_power = 0
        self.solar_powers = []
        self.charges = []
        self.battery_power = []
        self.discharges = []
        self.battery_charges = []
        self.battery_socs = []
        self.actions = []
        self.reasons = []
        self.balances = []
        self.sim_costs = []
        self.power_from_grid = []
        self.energy_from_grid = []
        self.energy_to_grid = []
        self.power_to_grid = []
        self.start_battery_soc = []
        self.feed_in_power_limitation = []
        self.solar_curtailed = []
        self.params = []

    def _calculate_grid_limit(self) -> int:
        return self.system['house_power'].max() * 2

    def reset(self) -> None:
        self._init_simulation_data()
        self.battery.reset()

    def is_done(self) -> bool:
        return self.current_interval == self.system.index[-1]

    def get_state(self) -> dict:
        row = self.system.loc[self.current_interval].copy()
        return self._create_state_dict(row)

    def _create_state_dict(self, row: pd.Series) -> dict:
        state_dict = row.to_dict()
        location = LocationInfo(name=self.location, region=self.state, timezone=self.timezone_str,
                                    latitude=self.latitude, longitude=self.longitude)
        # Calculate sunrise and sunset times
        s = sun(location.observer, date=self.current_interval.date())
        sunrise = s['sunrise']
        sunset = s['sunset']
        state_dict.update({
            'battery_charge': self.battery.charge,
            'battery_soc': self.battery.soc,
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
            'sunrise': sunrise.astimezone(ZoneInfo(self.timezone_str)),
            'sunset': sunset.astimezone(ZoneInfo(self.timezone_str)),
            'feed_in_power_limitation': row.get('feed_in_power_limitation', 0),
            'site_statistics': row.get('site_statistics', {}),
            'weather_data': row.get('weather_data', {}),
            'buy_forecast': row.get('buy_forecast', []),
            'sell_forecast': row.get('sell_forecast', []),
            'location': location,
            'spot_to_tariff': self.spot_to_tariff,
            'spot_to_feed_in_tariff': self.spot_to_feed_in_tariff,
            'sim_cost': self.algo_sim_usage
        })
        return state_dict

    def apply_action(self, inverter_action: str) -> None:
        row = self.system.loc[self.current_interval]
        if inverter_action is None:
            inverter_action = 'auto'
        if '-' in inverter_action:
            inverter_action, reason = inverter_action.split('-')
        self._process_interval(self.current_interval, row, inverter_action)
        self.current_interval += pd.Timedelta(minutes=self.interval)

    def _process_interval(self, index: pd.Timestamp, row: pd.Series, action: str, reason: str, params={}) -> None:
        house_power, solar_power, buy_price, sell_price, start_battery_soc = self._get_params(index, row)
        feed_in_power_limitation = params.get('feed_in_power_limitation', None)
        solar_curtailed = 0
        _balance = solar_power - house_power
        show_debug = False
        # if index == pd.Timestamp('2025-03-30 22:40:00+11:00'):
        #     show_debug = True
        #     print('Processing interval:', index, 'action:', action, 'reason:', reason, 'balance:', _balance,
        #           'house_power:', house_power, 'solar_power:', solar_power, 'buy_price:', buy_price,
        #           'sell_price:', sell_price, 'start_battery_soc:', start_battery_soc,)
        charge, discharge = self._calculate_charge_discharge(action, _balance, params=params, show_debug=show_debug)  # This is in Wh
        expected_grid_power = discharge - charge + solar_power - house_power  # This is in W
        # if index == pd.Timestamp('2025-03-30 22:40:00+11:00'):
        #     print('params:', params)
        #     print('expected_grid_power', expected_grid_power, 'feed_in_power_limitation', feed_in_power_limitation, 'balance:', _balance)
        #     print('charge', charge, 'discharge', discharge, 'solar_power',solar_power, 'house_power',house_power)
        if feed_in_power_limitation is not None and expected_grid_power > feed_in_power_limitation:
            curtail_needed = expected_grid_power + feed_in_power_limitation
            # if index == pd.Timestamp('2025-03-30 22:40:00+11:00'):
            #     logger.info(f'Curtail needed: {curtail_needed} solar_power: {solar_power}') 
            if curtail_needed > solar_power:
                solar_curtailed = solar_power
                solar_power = 0
            else:
                solar_curtailed = curtail_needed
                solar_power -= curtail_needed
        # if index == pd.Timestamp('2025-03-30 22:40:00+11:00'):
        #     logger.info(f'Processing interval: {index}, action: {action}, reason: {reason}, balance: {_balance}, house_power: {house_power}, solar_curtailed: {solar_curtailed} solar_power: {solar_power}, buy_price: {buy_price}, feed_in_power_limitation: {feed_in_power_limitation}, charge: {charge}, discharge: {discharge}')
        self._update_simulation_data(action, reason, solar_power, charge, discharge, house_power, buy_price, sell_price, start_battery_soc, feed_in_power_limitation, solar_curtailed, params)

    def _get_params(self, index: pd.Timestamp, row: pd.Series) -> Tuple[float, float, float]:
        if 'buy_price' not in row:
            raise ValueError('buy_price is not in the system dataframe')
            buy_price = self.spot_to_tariff(index, row['rrp'], self.tariff, self.network)
        else:
            buy_price = row['buy_price']
        if 'sell_price' not in row:
            sell_price = self.spot_to_feed_in_tariff(row['forecast'])
        else:
            sell_price = row['sell_price']
        start_battery_soc = 0
        if 'start_battery_soc' in row:
            start_battery_soc = row['start_battery_soc']
        return row['house_power'], row['solar_power'], buy_price, sell_price, start_battery_soc

    def _calculate_charge_discharge(self, action: str, balance: float, params={}, show_debug=False) -> Tuple[float, float]:
        feed_in_power_limitation = params.get('feed_in_power_limitation', None)
        optimal_charging = params.get('optimal_charging', None)
        optimal_discharging = params.get('optimal_discharging', None)
        if optimal_charging is not None:
            self.battery.charge_rate = min(optimal_charging, self.battery.max_charge_rate)
        if optimal_discharging is not None:
            self.battery.discharge_rate = min(optimal_discharging, self.battery.max_discharge_rate)
        if action is None:
            action = 'auto'
        action = str(action).lower()
        if '-' in action:
            action = action.split('-')[0]
        if action == 'charge':
            charge = self.battery.charge_battery(balance, self.interval)
            discharge = 0
        elif action == 'discharge':
            charge = 0
            # print('IMC: Discharge action:', action, 'balance:', balance, 'feed_in_power_limitation:',
            #       feed_in_power_limitation, '=', feed_in_power_limitation - balance)
            if feed_in_power_limitation:
                discharge = self.battery.discharge_battery(-balance, self.interval,
                                                           feed_in_power_limitation=feed_in_power_limitation - balance)
            else:
                discharge = self.battery.discharge_battery(-balance, self.interval)
        elif action == 'stopped':
            charge = discharge = 0
        elif action == 'export200':
            feed_in_power_limitation = 200
            if balance > 0:
                charge = self.battery.charge_battery(balance, self.interval,
                                                     feed_in_power_limitation=feed_in_power_limitation - balance)
                discharge = 0
            else:
                charge = 0
                discharge = self.battery.discharge_battery(-balance, self.interval)
        elif action == 'export':
            charge = 0
            if feed_in_power_limitation is not None:
                feed_in_power_limitation = feed_in_power_limitation - balance
                discharge = self.battery.discharge_battery(self.battery.discharge_rate, self.interval,
                                                        feed_in_power_limitation=feed_in_power_limitation - balance)
            else:
                discharge = self.battery.discharge_battery(self.battery.discharge_rate, self.interval)
        elif action == 'import':
            import_rate = self._get_import_rate(balance, show_debug=show_debug)
            if show_debug:
                print(f'Import rate: {import_rate}, balance: {balance}, grid_limit: {self.grid_limit}')
            charge = self.battery.charge_battery(import_rate, self.interval)
            discharge = 0
        else:
            if action != 'auto':
                print('Invalid action: using auto:', action)
            if balance > 0:
                charge = self.battery.charge_battery(balance, self.interval)
                discharge = 0
            else:
                charge = 0
                discharge = self.battery.discharge_battery(-balance, self.interval)

        assert charge >= 0, f'Charge is negative: {charge}'
        assert discharge >= 0, f'Discharge is negative: {discharge}'
        assert charge <= self.battery.max_charge_rate, f'Charge is greater than charge rate: {charge} v {self.max_charge_rate}'
        assert discharge <= self.battery.max_charge_rate, f'Discharge is greater than charge rate: {discharge} v {self.max_charge_rate}'
        return charge, discharge

    def _get_import_rate(self, balance: float, show_debug=False) -> float:
        import_rate = self.battery.charge_rate
        if show_debug:
            print(f'Balance: {balance}, grid_limit: {self.grid_limit}, battery charge rate: {self.battery.charge_rate}')
        if self.grid_limit:
            full_grid_charge = self.grid_limit + balance
            if full_grid_charge > self.battery.charge_rate:
                import_rate = self.battery.charge_rate
            elif full_grid_charge < 0:
                import_rate = 0
            else:
                import_rate = full_grid_charge
        if show_debug:
            print(f'Calculated import rate: {import_rate} for balance: {balance}')
        return import_rate if import_rate > 0 else 0

    def _update_simulation_data(self, action: str, reason: str, solar_power: float, charge: float, discharge: float, house_power: float,
                                buy_price: float, sell_price: float, start_battery_soc: float,
                                feed_in_power_limitation: float, solar_curtailed: float, params={}) -> None:
        self.solar_powers.append(solar_power)
        self.charges.append(charge)
        self.discharges.append(discharge)
        self.battery_charges.append(self.battery.charge)
        self.battery_power.append(discharge - charge)
        self.feed_in_power_limitation.append(feed_in_power_limitation)
        self.battery_socs.append(self.battery.soc)
        self.solar_curtailed.append(solar_curtailed)
        self.actions.append(action)
        self.reasons.append(reason)
        self.params.append(params)

        balance = solar_power - house_power - charge + discharge
        # print('balance', balance, 'solar_power', solar_power, 'house_power', house_power, 'charge', charge, 'discharge', discharge)
        self.grid_power = balance
        self.balances.append(balance)

        kwh_balance = balance * (self.interval / 60) / 1000
        if kwh_balance < 0:
            self.power_from_grid.append(-balance)
            self.energy_from_grid.append(-kwh_balance)
            self.power_to_grid.append(0)
            self.energy_to_grid.append(0)
            self.last_cost = buy_price * -kwh_balance
        else:
            self.power_from_grid.append(0)
            self.energy_from_grid.append(0)
            self.power_to_grid.append(balance)
            self.energy_to_grid.append(kwh_balance)
            self.last_cost = -sell_price * kwh_balance
        self.last_cost += self.daily_fee / (60 * 24 / self.interval)
        self.sim_costs.append(self.last_cost)

    def run_simulation(self) -> Tuple[float, pd.DataFrame]:
        for index, row in self.system.iterrows():
            self.current_interval = index
            params = self.get_state()
            params['past_power_from_grid'] = self.power_from_grid[:-12] if len(self.power_from_grid) > 12 else self.power_from_grid
            if 'interval_time' in params:
                del params['interval_time']
            if 'buy_forecast' not in params:
                params['buy_forecast'] = [self.spot_to_tariff(index, self.network, self.tariff, f) for f in row['forecast']]
            if 'sell_forecast' not in params:
                params['sell_forecast'] = [self.spot_to_feed_in_tariff(f) for f in row['forecast']]
            self._process_interval(index, row, *self.control_function(index, **params))
        self._calculate_final_metrics()
        return self.algo_sim_usage, self.system

    def _calculate_final_metrics(self) -> None:
        self.system['charge'] = self.charges
        self.system['discharge'] = self.discharges
        self.system['battery_power'] = self.battery_power
        self.system['action'] = self.actions
        self.system['reason'] = self.reasons
        self.system['battery_charge'] = self.battery_charges
        self.system['battery_soc'] = self.battery_socs
        self.system['balance'] = self.balances
        self.system['sim_grid'] = self.balances
        self.system['grid_power'] = self.balances
        self.system['sim_cost'] = self.sim_costs
        self.system['solar_power'] = self.solar_powers
        self.system['Power from grid'] = self.power_from_grid
        self.system['Power to grid'] = self.power_to_grid
        self.system['Energy from grid'] = self.energy_from_grid
        self.system['Energy to grid'] = self.energy_to_grid
        self.system['feed_in_power_limitation'] = self.feed_in_power_limitation
        self.system['solar_curtailed'] = self.solar_curtailed
        existing_columns = set(self.system.columns)
        new_columns = set()
        for param in self.params:
            for key, value in param.items():
                if key not in existing_columns and key not in new_columns:
                    new_columns.add(key)
        # Remove any columns with incorrect length
        for col in list(self.system.columns):
            if len(self.system[col]) != len(self.system.index):
                self.system.drop(columns=[col], inplace=True)
        # Add new columns with correct length
        new_cols_dict = {}
        for new_col in new_columns:
            values = [param.get(new_col, None) for param in self.params]
            if len(values) == len(self.system.index):
                new_cols_dict[new_col] = values
        if new_cols_dict:
            new_cols_df = pd.DataFrame(new_cols_dict, index=self.system.index)
            self.system = pd.concat([self.system, new_cols_df], axis=1)
        
        self.algo_sim_usage = self.system['sim_cost'].sum()


def sim_inverter(system: pd.DataFrame, control_function: Callable, **kwargs: Any) -> Tuple[float, pd.DataFrame]:
    sim = InverterSimulator(system, control_function, **kwargs)
    return sim.run_simulation()
