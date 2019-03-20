import pandas as pd
import numpy as np
import datetime

maps = {}
maps["hod"] = [lambda x: x, 24]
maps["dow"] = [lambda x: x, 7]
maps["is_car"] = [lambda x: int(x), 2]
maps["des_char"] = [lambda x: np.digitize(x, [5, 10, 20, 40]), 5]
maps["per_char"] = [lambda x: np.digitize(x, [0.2, 0.4, 0.6, 0.8]), 5]
maps["price"] = [lambda x: np.digitize(x, [0.2, 0.4, 0.6, 0.8]), 5]
maps["curr_dur"] = [lambda x: np.digitize(x, [0.5, 1, 2, 4]), 5]


def load_elec_price_data(elec_price_data_file, time_step):
    """
    Load all electricity price data
    :param elec_price_data_file:
    :param time_step:
    :return:
    """
    df = pd.read_csv(elec_price_data_file)
    # df = df.set_index(pd.DatetimeIndex(df['timestamp']))
    # df = df[pd.date_range(start_time, start_time + datetime.timedelta(hours=episode_length * time_step)), time_step]
    # df - df.resample(str(time_step) + 'H').sum()
    return df


def load_charging_data(charging_data_file, num_stations, time_step):
    """
    Load all charging data
    :param charging_data_file:
    :param num_stations:
    :param time_step:
    :return:
    """
    df = pd.read_csv(charging_data_file)
    df = df[["Port ID", "Station Start Time (Local)", "Energy (kWh)", "Session Time (secs)"]]

    # Ports
    station_id = sorted([int(x) for x in df["Port ID"].unique()])

    # convert to datetimes and sort
    df['Station Start Time (Local)'] = df['Station Start Time (Local)'].apply(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    df['Station Start Time (Local)'] = df['Station Start Time (Local)'].apply(
        lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour, 15 * (dt.minute // 15)))
    df = df.sort_values(['Station Start Time (Local)'])

    # convert to hours
    df['Session Time (secs)'] = df['Session Time (secs)'] / 3600.0

    df = df.set_index(pd.DatetimeIndex(df['Station Start Time (Local)']))

    df = df[df["Port ID"].isin(station_id[0:num_stations])]

    return df

    # # iterate: split ports, create tuples
    # station_list = []
    # for this_id, _ in zip(station_id,range(num_stations)):
    #     df_id = df[df["Port ID"] == this_id]
    #     df_id = df_id.drop(columns=["Port ID"])
    #     station_list.append(list(df_id.itertuples(index=False, name=None)))
    #
    # return station_list


def sample_charging_data(charging_data, episode_length, time_step, random_state):
    max_start_datetime = charging_data.index[-1] - datetime.timedelta(
        hours=episode_length * time_step)
    valid_dates = charging_data.loc[charging_data.index[0]:max_start_datetime].index
    start_time = valid_dates[random_state.choice(range(len(valid_dates)))].to_pydatetime()

    df = charging_data.loc[start_time:start_time + datetime.timedelta(hours=episode_length * time_step)]
    # iterate: split ports, create tuples
    station_list = []
    station_id = charging_data["Port ID"].unique()
    for this_id in station_id:
        df_id = df[df["Port ID"] == this_id]
        df_id = df_id.drop(columns=["Port ID"])
        station_list.append(list(reversed(list(df_id.itertuples(index=False, name=None)))))

    return station_list


def sample_elec_price_data(elec_price_data, start_time, episode_length, time_step):
    pass


def one_hot(value, data):
    global maps
    output = [0]*maps[data][1]
    output[maps[data][0](value)] = 1
    return output


def featurize_s(s):
    hod = one_hot(s['time'].hour, "hod")
    dow = one_hot(s['time'].weekday(), "dow")
    price = one_hot(s['price'], 'price')
    is_car = []
    des_char = []
    # per_char = []
    per_missing = []
    curr_dur = []
    for stn in range(len(s['stations'])):
        stn_s = s['stations'][stn]
        is_car += one_hot(stn_s['is_car'], "is_car")
        des_char += one_hot(stn_s['des_char'], "des_char")
        per_m = float(stn_s['is_car']) * (1.0 - stn_s['per_char'])
        per_missing += one_hot(per_m, "per_char")
        # per_char += one_hot(stn_s['per_char'], "per_char")
        curr_dur += one_hot(stn_s['curr_dur'], "curr_dur")
    featurized = np.concatenate(
        (hod, dow, is_car, des_char, per_missing, curr_dur, price)
    )
    return featurized


def featurize_cont(s):
    # hod = one_hot(s['time'].hour, "hod")
    hod_single = [((s['time'].hour + s['time'].minute/60.0) / 13.0) - 1.0]
    hod_single2 = [abs(x) for x in hod_single]
    dow = one_hot(s['time'].weekday(), "dow")
    price = s['price']
    print(s)
    is_car = []
    des_char = []
    per_char = []
    curr_dur = []
    for stn in range(len(s['stations'])):
        is_car.append(float(s['stations'][stn]['is_car']))
        des_char.append(s['stations'][stn]['des_char'] / 20)
        per_char.append(s['stations'][stn]['per_char'])
        curr_dur.append(s['stations'][stn]['curr_dur'] / 4)
    per_missing = (1.0 - np.array(per_char)) * np.array(is_car)
    missing_charge = per_missing * np.array(des_char)
    featurized = np.concatenate(
        (dow, hod_single, hod_single2, price, is_car, des_char, missing_charge, per_missing, curr_dur)
    )
    return featurized


def scale_action(action, transformer_capacity):
    tot_charge_request = np.sum(action)
    if tot_charge_request > transformer_capacity:
        return action*transformer_capacity/tot_charge_request
    else:
        return action


   # # Called by take_action
    # # the three arguments are lists of the given values at each station
    # def reward_exp(self, energy_charged, percent_charged, charging_powers):
    #     charge_reward = sum(np.array(energy_charged) * (np.exp(percent_charged) - 1))  # sum [0, energy_charged*(e-1)] (~[0, 8.5])
    #
    #     elec_price = self.elec_price_data[self.get_current_state()['time'].to_pydatetime()]
    #     elec_cost = sum(np.array(energy_charged) * elec_price * (np.exp(1) - 1))  # sum [0, energy_charged*(e-1)] (~[0, 8.5])
    #
    #     pow_violation = max(np.sum(charging_powers) - self.transformer_capacity, 0) / self.transformer_capacity
    #     pow_penalty = np.exp(pow_violation * 12) - 1  # [0, e^(10*pow_violation) - 1]  (~[0, 20])
    #     # print(charge_reward, elec_cost, pow_penalty)
    #
    #     return self.reward_weights[0] * charge_reward - self.reward_weights[1] * elec_cost - self.reward_weights[2] * pow_penalty
