from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta, timezone  # noqa: F401
from zoneinfo import ZoneInfo
import numpy as np  # noqa: F401
import asyncio  # noqa: F401
from contextlib import suppress  # noqa: F401
from inverter_simulator.simulator import InverterSimulator
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import guarded_iter_unpack_sequence
from RestrictedPython import safe_builtins
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
import re
from astral import LocationInfo
from astral.sun import sun
from inverterintelligence.decision_logger import DecisionLogger
from pytrader.permutation_model import PermutationModel, find_best_five_minute_trades
from pytrader.aemo_retrieval import retrieve_forecasted_prices
from pytrader.battery.battery_activity import BatteryActivity
from unittest.mock import MagicMock
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def cicd_parse_script(script_lines):
    """
    Parses a script file for lines formatted as '# CICD: 'date', 'action'' and returns a list of tuples with date and action.
    """
    cicd_lines = [line for line in script_lines if line.startswith("# CICD: ")]
    cicd_data = []
    for line in cicd_lines:
        try:
            date_str, action = cicd_parse_line(line)
            cicd_data.append((date_str, action))
        except Exception as e:
            logger.error(f"Error parsing line '{line}': {e}")
    return cicd_data

def cicd_parse_line(line):
    """
    Parses a line formatted as '# CICD: 'date', 'action'' and returns the date and action.
    """
    if line.startswith("# CICD: "):
        line = line[9:]  # Remove the "# CICD: " prefix
        parts = line.split(",")
        if len(parts) == 2:
            date_str = parts[0].strip("'")
            action = parts[1].strip().strip("'") 
            return date_str, action
        else:
            return None
    else:
        return None

def classify_battery(state='NSW', battery_capacity=50, charge_rate=25, charge=25,
                     charge_efficiency=95, discharge_efficiency=95,
                     clairvoyant=False, cash=0, sink=0):
    possible_hours = [2, 3, 4]
    duration = battery_capacity / charge_rate

    # Round down to nearest classification
    classified_duration = max([h for h in possible_hours if h <= duration], default=2)

    base_unit = 25
    # Recalculate new capacity and proportionate charge
    new_capacity = int(classified_duration * base_unit)
    battery_soc = (charge / battery_capacity)
    new_charge = int(charge * battery_soc)
    logger.info(f"Classified battery: {state}, capacity: {new_capacity} kWh, charge: {new_charge} kWh")
    return MagicMock(state=state, battery_capacity=new_capacity, charge_rate=charge_rate, charge=base_unit,
                     charge_efficiency=charge_efficiency, discharge_efficiency=discharge_efficiency,
                     clairvoyant=clairvoyant, cash=cash, sink=sink)

# Global cache for options to avoid recomputation per simulation run
_OPTIONS_CACHE = {}

def build_options(half_hour_window=5, five_minute_window=12, battery_capacity=80, charge_rate=25,
                   charge_efficiency=95, discharge_efficiency=95, charge=40):
    """
    Build the battery options for the simulation.
    Caches options globally per run to avoid recomputation.
    """
    cache_key = str(f"{half_hour_window}_{five_minute_window}_{battery_capacity}_{charge_rate}_"
                   f"{charge_efficiency}_{discharge_efficiency}")
    if cache_key in _OPTIONS_CACHE:
        return _OPTIONS_CACHE[cache_key]

    battery = classify_battery(state='NSW', clairvoyant=False, battery_capacity=battery_capacity,
                               charge_rate=charge_rate, charge_efficiency=charge_efficiency,
                               discharge_efficiency=discharge_efficiency,
                               charge=charge, cash=0, sink=0)
    permutation_model = PermutationModel()
    half_hour_options = permutation_model.get_options(
            half_hour_window,
            capacity=battery.battery_capacity,
            in_discharge_efficiency=int(battery.discharge_efficiency),
            in_charge_efficiency=int(battery.charge_efficiency)
        )
    five_minute_options = permutation_model.get_five_minute_options(
            capacity=battery.battery_capacity,
            in_discharge_efficiency=int(battery.discharge_efficiency),
            in_charge_efficiency=int(battery.charge_efficiency),
            window=five_minute_window
        )
    _OPTIONS_CACHE[cache_key] = (half_hour_options, five_minute_options)
    return half_hour_options, five_minute_options

def get_battery_activity(interval_time: datetime, half_hour_options=None, five_min_options=None, state='NSW', battery_capacity=80, charge_rate=25,
        charge_efficiency=95, discharge_efficiency=95, charge=40, half_hour_window=5, five_minute_window=12,
        five_min_forecast=None, forecast=None):
    """
    Get the battery activity for a given time interval.
    :param interval_time: The time interval for the battery activity.
    :param half_hour_options: The half-hour options for the battery.
    :param five_min_options: The five-minute options for the battery.
    :param state: The state of the battery.
    :param battery_capacity: The capacity of the battery in kWh.
    :param charge_rate: The charge rate of the battery in kW.
    :param charge_efficiency: The charge efficiency of the battery in percentage.
    :param discharge_efficiency: The discharge efficiency of the battery in percentage.
    :param charge: The current charge of the battery in kWh.
    :param half_hour_window: The half-hour window for the battery options.
    :param five_minute_window: The five-minute window for the battery options.
    :return: A tuple containing the action and confidence of the battery activity."""

    battery = classify_battery(state='NSW', clairvoyant=False, battery_capacity=battery_capacity,
                               charge_rate=charge_rate, charge_efficiency=charge_efficiency,
                               discharge_efficiency=discharge_efficiency,
                               charge=charge, cash=0, sink=0)
    if five_min_options is None or half_hour_options is None:
        half_hour_options, five_min_options = build_options(
            half_hour_window=5,
            five_minute_window=12,
            battery_capacity=battery.battery_capacity,
            charge_rate=battery.charge_rate,
            charge_efficiency=battery.charge_efficiency,
            discharge_efficiency=battery.discharge_efficiency,
            charge=battery.charge
        )
    if five_min_forecast is None or forecast is None:
        five_min_forecast, forecast = retrieve_forecasted_prices(interval_time, battery, is_historic=False)
    five_minute_prices = [float(i) for i in five_min_forecast]
    half_hour_prices = [(float(i)) for i in forecast]
    max_permutation, confidence = find_best_five_minute_trades(
            five_min_options=five_min_options,
            half_hour_options=half_hour_options,
            five_min_prices=five_minute_prices,
            half_hour_prices=half_hour_prices,
            five_min_window=five_minute_window,
            half_hour_window=half_hour_window,
            charge=battery.charge  # type: ignore
        )
    action = BatteryActivity.get_action_by_cash(max_permutation["five_minute_permutation"][0])
    if action == BatteryActivity.HOLD:
        action = 'stopped'
    elif action == BatteryActivity.DISCHARGE:
        action = 'export'
    elif action == BatteryActivity.CHARGE:
        action = 'import'
    else:
        action = 'auto'
    return action, confidence

def guarded_unpack_sequence(seq, count):
    """
    Safely unpack a sequence with a fixed number of elements in restricted code.
    Raises ValueError if the sequence does not have exactly the expected number of elements.
    """
    if isinstance(seq, (list, tuple, set)) and len(seq) == count:
        return seq
    raise ValueError(f"Cannot unpack sequence: Expected {count} elements, got {len(seq)}")

def restricted_run_code(user_code, action_params, file_name=None):
    SAFE_AUGUMENTED_ASSIGNMENT_OPERATORS = (
        '+=', '-=', '*=', '/=', '%=', '**=',
        '<<=', '>>=', '|=', '^=', '&=', '//='
    )

    def custom_inplacevar(op, x, y):
        assert op in SAFE_AUGUMENTED_ASSIGNMENT_OPERATORS
        globs = {'x': x, 'y': y}
        exec(f'x {op} y', {}, globs)
        return globs['x']
    
    interval_time = action_params.get('interval_time', datetime.now())
    hour = action_params.get('interval_time', datetime.now()).hour
    decisions = DecisionLogger()
    restricted_globals = {"__builtins__": safe_builtins,
                          "__import__": lambda name, globals=None, locals=None, fromlist=(), level=0: __import__(name),
                          "getattr": lambda obj, attr: getattr(obj, attr),
                          "_getitem_": lambda obj, attr: obj[attr],
                          "min": min,
                          "max": max,
                          "sum": sum,
                          "mean": lambda x: sum(x) / len(x) if x else 0,
                          "math_log": math.log,
                          "sun": sun,
                          "all": all,
                          "any": any,
                          "list": list,
                          "range": range,
                          "sorted": sorted,
                          "enumerate": enumerate,
                          "timedelta": timedelta,
                          "datetime": datetime,
                          "timezone": timezone,
                          "ZoneInfo": ZoneInfo,
                          "suppress": suppress,
                          "next": next,
                          "interval_time": interval_time,
                          "hour": hour,
                          "inverters": {},
                          "np": np,
                          "log": logger.info if logger else lambda x: None,
                          "exit": lambda: None,
                          "quit": lambda: None,
                          "MagicMock": lambda: None,
                          "_inplacevar_": custom_inplacevar,
                          "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
                          "_unpack_sequence_": guarded_unpack_sequence,
                          "_getiter_": iter}
    try:
        for key, value in read_vars_from_lines(user_code.split("\n")).items():
            restricted_globals[key] = value
        action_params = process_params(action_params, restricted_globals)
        byte_code = compile_restricted(user_code, '<inline code>', 'exec')
        action_params['decisions'] = decisions
        exec(byte_code, restricted_globals, action_params)
        if decisions.has_decisions():
            action_params['reason'] = decisions.get_reason()
        action_params['decisions'] = decisions.to_dict()
    except Exception as e:
        # Extract line number, filename, offset, and text details for enhanced debugging
        logger.error(f"Error executing user code {file_name}: {e}", exc_info=True)
    return action_params


def process_params(params: dict, restricted_globals: dict) -> dict:  # noqa: C901
    interval_time = params['interval_time']
    weather_data = params.get('weather_data', {})
    forecast = params.get('forecast', [])
    battery_soc = params.get('battery_soc', 0.0)
    # Actually load them from the script
    BATTERY_SOC_NEEDED = restricted_globals.get('BATTERY_SOC_NEEDED', 80)
    BAD_SUN_DAY_KEEP_SOC = restricted_globals.get('BAD_SUN_DAY_KEEP_SOC', 20)
    GOOD_SUN_DAY = restricted_globals.get('GOOD_SUN_DAY', 20)
    GOOD_SUN_HOUR = restricted_globals.get('GOOD_SUN_HOUR', 20)

    current_hour = interval_time.hour
    hourly_gti_forecast = weather_data.get('hourly', {}).get('global_tilted_irradiance_instant', [-1] * 48)
    hours_until_midnight = 24 - current_hour
    gti_sum_tomorrow = sum(hourly_gti_forecast[hours_until_midnight:])
    soc_surplus = 0.0
    time_left = 0.0
    projected_deficit = 0.0
    night_reserve = BATTERY_SOC_NEEDED
    if gti_sum_tomorrow < (GOOD_SUN_DAY * 100):
        night_reserve += BAD_SUN_DAY_KEEP_SOC
    if forecast and current_hour > 16:
        morning_peak = max(forecast[:-6])
        if morning_peak > 400:
            night_reserve += 10
    elif forecast and current_hour < 6:
        morning_away = current_hour * 2
        morning_peak = max(forecast[morning_away:])
        if morning_peak > 400:
            night_reserve += 10
    night_reserve = min(night_reserve, 100)

    def get_first_above_threshold(hourly_gti_forecast, threshold):
        for i in range(len(hourly_gti_forecast)):
            if hourly_gti_forecast[i] > threshold:
                return i
        return -1

    def find_last_index(hourly_gti_forecast, threshold):
        for i in range(len(hourly_gti_forecast)):
            if hourly_gti_forecast[i] < threshold:
                return i
        return -1

    first_good_gti = get_first_above_threshold(hourly_gti_forecast, GOOD_SUN_HOUR * 10)
    last_good_gti = find_last_index(hourly_gti_forecast[:24], GOOD_SUN_HOUR * 10)
    is_solar_window_now = (interval_time.hour >= first_good_gti) and (interval_time.hour <= last_good_gti)
    sunrise_datetime = params.get('sunrise', None)
    if sunrise_datetime is None:
        sunrise_datetime = interval_time.replace(hour=6)
    sunrise_time = sunrise_datetime.time()
    sunset_datetime = params.get('sunset', None)
    if sunset_datetime is None:
        sunset_datetime = interval_time.replace(hour=18)
    sunset_time = sunset_datetime.time()
    is_daytime = sunrise_time < interval_time.time() < sunset_time
    if battery_soc and is_daytime:
        soc_surplus = battery_soc - night_reserve
        time_left = (((sunset_datetime - interval_time).total_seconds() - 1800) % 86400) / 3600
    elif battery_soc and not is_daytime:
        night_hours = (((sunrise_datetime - sunset_datetime).total_seconds()) % 86400) / 3600
        time_left = (((sunrise_datetime - interval_time).total_seconds() + 1800) % 86400) / 3600
        soc_surplus = (battery_soc - night_reserve)
        projected_deficit = battery_soc - night_reserve * (time_left / night_hours)
    gti_today = sum(hourly_gti_forecast[:hours_until_midnight])
    gti_past = sum(hourly_gti_forecast[:interval_time.hour])
    gti_to_2pm = sum(hourly_gti_forecast[:15])
    params.update({
        'current_hour': current_hour,
        'hourly_gti_forecast': hourly_gti_forecast,
        'hours_until_midnight': hours_until_midnight,
        'gti_today': gti_today,
        'gti_past': gti_past,
        'gti_to_2pm': gti_to_2pm,
        'gti_sum_tomorrow': gti_sum_tomorrow,
        'night_reserve': night_reserve,
        'first_good_gti': first_good_gti,
        'last_good_gti': last_good_gti,
        'is_solar_window_now': is_solar_window_now,
        'is_daytime': is_daytime,
        'soc_surplus': soc_surplus,
        'time_left': time_left,
        'projected_deficit': projected_deficit,
    })
    return params

def run_scripted_simulation(meter_data_df, script_content, filename, interval, battery_capacity, tariff, network,
                            charge_rate, max_ppv_power, daily_fee, spot_to_tariff, state,
                            latitude, longitude, timezone_str, **kwargs):
    default_action = kwargs.get('default_action', 'auto')
    def run_user_code(interval_time, **kwargs):
        try:
            location = LocationInfo(name='', region=state, timezone=timezone_str,
                                    latitude=latitude, longitude=longitude)
            # Calculate sunrise and sunset times
            s = sun(location.observer, date=interval_time.date())
            sunrise = s['sunrise']
            sunset = s['sunset']
            timezone = ZoneInfo(timezone_str)
            params = {'interval_time': interval_time,
                      'battery_capacity': battery_capacity,
                      'charge_rate': charge_rate,
                      'max_ppv_power': max_ppv_power,
                      'action': 'auto',
                      'reason': 'default: auto',
                      'latitude': latitude,
                      'longitude': longitude,
                      'grid_power': 0,
                      'export_limit_max': charge_rate,
                      'feed_in_power_limitation': charge_rate,
                      'solar': 'maximise',
                      'sunrise': sunrise.astimezone(timezone),
                      'sunset': sunset.astimezone(timezone),
                      'location': LocationInfo("Sydney", "Australia", timezone, latitude, longitude)}
            for key, val in kwargs.items():
                params[key] = val

            params = restricted_run_code(script_content, params, filename)
            return params['action'], params['reason'], params
        except Exception as e:
            logger.error(f"Error in user code {filename}: {e}", exc_info=True)
            return default_action, f"Error: {e}"

    sim = InverterSimulator(meter_data_df.copy(), run_user_code, interval=interval, battery_capacity=battery_capacity,
                            spot_to_tariff=spot_to_tariff, tariff=tariff, network=network,
                            charge_rate=charge_rate, max_ppv_power=max_ppv_power, daily_fee=daily_fee,
                            **kwargs)
    return sim.run_simulation()

def read_script_lines(filename):
    """
    Reads a script file and returns the content as lines of strings.
    """
    with open(filename, "r") as file:
        script_lines = file.readlines()

    return script_lines

def read_vars_from_script(filename):
    """
    Reads capitalized variables from a file and returns them as a dictionary.
    For example, lines like: SELL_PRE_DAWN = 50
    """
    lines = read_script_lines(filename)
    return read_vars_from_lines(lines)

def read_vars_from_lines(lines):
    pattern = r"^[A-Z_0-9]+\s*=\s*\d+(\.\d+)?"
    matches = [line.strip() for line in lines if re.match(pattern, line)]

    variables_dict = {
        line.split('=')[0].strip(): round(float(line.split('=')[1].split('#')[0].strip()))
        for line in matches
    }
    return variables_dict

ECONOMIST_COLORS = {
    "blue": "#6794a7",
    "red": "#cc5a71",
    "green": "#8a9b0f",
    "orange": "#fd8d3c",
    "purple": "#756bb1",
    "grey": "#bdbdbd",
    "cyan": "#80b1d3",
    "darkblue": "#084594",
    "darkgreen": "#006d2c",
    "yellow": "#ffd92f"
}

ACTION_COLORS = {
    'export': ECONOMIST_COLORS["red"],
    'import': ECONOMIST_COLORS["green"],
    'charge': ECONOMIST_COLORS["blue"],
    'auto': ECONOMIST_COLORS["grey"],
    'discharge': ECONOMIST_COLORS["purple"]
}

def plot(ret_df, title='Simulation run'):
    ret_df['cost'] = ret_df['sim_cost']
    interval = round((ret_df.index[1]-ret_df.index[0])/ pd.Timedelta(minutes=1))
    max_zoomed_rrp = 500

    if 'house_power' in ret_df.columns:
        ret_df['house_consumption'] = ret_df['house_power']
    if 'solar_power' in ret_df.columns:
        ret_df['ppv'] = ret_df['solar_power']
    if 'as_is_grid' not in ret_df.columns:
        if 'pgrid' in ret_df.columns:
            ret_df['as_is_grid'] = ret_df['pgrid']
        elif 'house_consumption' in ret_df.columns and 'ppv' in ret_df.columns:
            ret_df['as_is_grid'] = ret_df['house_consumption'] - ret_df['ppv']
        else:
            ret_df['as_is_grid'] = ret_df['Power fom grid'] - ret_df['Power to grid']
    # Negative draws from the grid (so get absolute value)
    ret_df['as_is_general_power'] = abs(ret_df['as_is_grid'].clip(upper=0))
    if 'as_is_general_kwh' not in ret_df.columns:
        ret_df['as_is_general_kwh'] = ret_df['as_is_general_power'] / 1000.0 * interval / 60
    # Positive draws from the grid
    ret_df['as_is_feed_in_power'] = ret_df['as_is_grid'].clip(lower=0)
    if 'as_is_feed_in_kwh' not in ret_df.columns:
        ret_df['as_is_feed_in_kwh'] = ret_df['as_is_feed_in_power'] / 1000.0 * interval / 60

    nb_bill = 0
    retail_bill = ret_df['cost'].sum() / 100
    # five plots one ontop of the other, the first one twice as high as the others
    fig, ax = plt.subplots(5, 1, figsize=(20, 15), sharex=True,
                       gridspec_kw={'height_ratios': [2, 1, 1, 1, 1]})
    plt.subplots_adjust(hspace=0.3)  # Better vertical spacing
    for a in ax:
        a.title.set_fontsize(14)
        a.tick_params(axis='x', labelsize=10)
        a.tick_params(axis='y', labelsize=10)
        a.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    # First show costs
    start_date, end_date = ret_df.index[0], ret_df.index[-1]
    ax[0].set_title(f'{title}: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')

    algo_label = 'Simulation Run $%.2f' % retail_bill
    ax[0].plot(ret_df['cost'].cumsum(), label=algo_label, color=ECONOMIST_COLORS["orange"])
    if 'billed_costs' in ret_df.columns and 'billed_earnings' in ret_df.columns:
        ret_df['retail_cost'] = (ret_df['billed_costs'] - ret_df['billed_earnings'])
        retail_label = 'Reported Bill'
        ax[0].plot(ret_df['retail_cost'].cumsum(), label='%s $%.2f' % (retail_label, ret_df['retail_cost'].sum() / 100), color=ECONOMIST_COLORS["grey"])
        # Highlight where cost is less than retail_cost (improvement)
        improvement = ret_df['cost'].cumsum() < ret_df['retail_cost'].cumsum()
        ax[0].fill_between(ret_df.index, ret_df['cost'].cumsum(), ret_df['retail_cost'].cumsum(),
                           where=improvement, color=ECONOMIST_COLORS["green"], alpha=0.2, label='Savings')
        # Highlight where cost is greater than retail_cost (worse)
        worse = ret_df['cost'].cumsum() > ret_df['retail_cost'].cumsum()
        ax[0].fill_between(ret_df.index, ret_df['cost'].cumsum(), ret_df['retail_cost'].cumsum(),
                           where=worse, color=ECONOMIST_COLORS["red"], alpha=0.2, label='Extra Cost')
        ax[0].legend()
    ax[0].xaxis.set_visible(False)
    ax[0].legend()
    # Set height of ax[0] to twice the height of ax[1]
    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0, box.width, box.height * 1.5])

    # show as bar chart Power to grid	Power from grid
    ax[1].set_title('Power to and from grid')
    ax[1].plot(ret_df.index, ret_df['Power from grid'], label='Power from grid (W)')
    ax[1].plot(ret_df.index, -ret_df['Power to grid'], label='Power to grid (W)')

    if 'general_kwh' in ret_df.columns and 'feed_in_kwh' in ret_df.columns:
        kwh_to_w = 1000 * 60 / interval
        ax[1].plot(ret_df.index, ret_df['general_kwh'] * kwh_to_w, label='Metered general (W)')
        ax[1].plot(ret_df.index, -ret_df['feed_in_kwh'] * kwh_to_w, label='Metered feed (W)')

    ax[1].legend()
    ax[1].xaxis.set_visible(False)
    # now SOC and actions
    ax[2].xaxis.set_visible(False)
    action_column = ''
    if 'battery_soc' in ret_df.columns:
        action_column = 'battery_soc'
        ax[2].set_title('Battery SOC')
        ax[2].plot(ret_df['battery_soc'], label='Simulated soc (%)', color=ECONOMIST_COLORS["orange"])
        ax[2].legend()
        if 'start_battery_soc' in ret_df.columns:
            ax[2].plot(ret_df['start_battery_soc'], label='Reported soc (%)', color=ECONOMIST_COLORS["grey"])
            ax[2].legend()
    elif 'battery_charge' in ret_df.columns:
        action_column = 'battery_charge'
        ax[2].set_title('Battery charge')
        ax[2].plot(ret_df['battery_charge'], label='Battery charge (Wh)', color=ECONOMIST_COLORS["green"])
        ax[2].legend()
        if 'battery_actual' in ret_df.columns:
            ax[2].plot(ret_df['battery_actual'], label='Battery actual (Wh)', color=ECONOMIST_COLORS["blue"])
            ax[2].legend()
    if 'action' in ret_df.columns:
        ax[2].set_title('Battery charge with actions')
        ret_df['change_action'] = ret_df['action'].shift(1) != ret_df['action']
        for row in ret_df[ret_df['change_action']].iterrows():
            color = ACTION_COLORS.get(row[1]['action'], 'black')
            if action_column in row[1] and row[1]['action']:
                ax[2].text(row[0], row[1][action_column], f"{row[1]['action'][0]}", color=color)
    ax[3].set_title('House and Solar Power')
    ax[3].plot(ret_df['house_consumption'], label='House Power (W)', color=ECONOMIST_COLORS["blue"])
    ax[3].fill_between(ret_df.index, 0, ret_df['house_consumption'], color=ECONOMIST_COLORS["blue"], alpha=0.15)
    ax[3].plot(ret_df['ppv'], color=ECONOMIST_COLORS["yellow"], label="Solar Power (W)")
    ax[3].fill_between(ret_df.index, 0, ret_df['ppv'], color=ECONOMIST_COLORS["yellow"], alpha=0.15)
    ax[3].legend()
    ax[3].xaxis.set_visible(False)

    # then subplot show rrp but only scale to 300
    if ret_df['rrp'].max() > 1000:
        ax[4].set_ylim(-10, max_zoomed_rrp)
    high_rrp = ret_df['rrp'] > 3000
    ax[4].fill_between(ret_df.index, 0, ret_df['rrp']/10,
                       where=high_rrp, color=ECONOMIST_COLORS["red"], alpha=0.3, label='Spike')

    ax[4].set_title('Wholesale market price')
    ax[4].plot(ret_df['rrp']/10, label='Spot price c/kWh', color=ECONOMIST_COLORS["darkgreen"])
    if 'buy_price' in ret_df.columns:
        ax[4].plot(ret_df['buy_price'], label='General price c/kWh', color=ECONOMIST_COLORS["darkblue"])
    ax[4].legend()
    return fig, ax
