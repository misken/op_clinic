#!/usr/bin/env python
# coding: utf-8

import sys
import argparse
import math
import logging
from datetime import datetime
import csv

import pandas as pd
from numpy.random import default_rng
import simpy
from pathlib import Path


class Patient(object):
    def __init__(self, patient_id, appt_type):
        """

        Parameters
        ----------
        patient_id
        appt_type
        """
        self.patient_id = patient_id
        self.appt_type = appt_type

    def __str__(self):
        return self.patient_id


class ScheduledPatientGenerator(object):
    def __init__(self, env, clinic, mean_interappt_time, num_appts_per_block,
                 max_patients, stoptime, rg, enabled=True):

        self.env = env
        self.clinic = clinic
        self.mean_interappt_time = mean_interappt_time
        self.num_appts_per_block = num_appts_per_block
        self.stoptime = stoptime
        self.max_patients = max_patients
        self.rg = rg


        if enabled:
            # Start creating walk in patients
            env.process(self.run())

    def run(self):

        num_patients = 0
        # Generate first block of patients at time 0
        # Generate block of patients at this time
        for p in range(self.num_appts_per_block):
            # Generate new patient
            num_patients += 1
            # Create patient id which is just "s" (for scheduled) followed by number of such patients created.
            patient_id = f"s{num_patients}"
            patient = Patient(patient_id, "scheduled")

            logging.info(f"{self.env.now:.4f}:Patient {patient_id} created at {self.env.now:.4f}.")

            # Register a get_exam process for the new patient
            self.env.process(self.clinic.get_exam(patient))

        # Loop for generating patients
        while self.env.now < self.stoptime and num_patients < self.max_patients:
            # Generate next interarrival time
            iat = self.mean_interappt_time

            # This process will now yield to a 'timeout' event. This process will resume after iat time units.
            yield self.env.timeout(iat)

            # Generate block of patients at this time
            for p in range(self.num_appts_per_block):
                # Generate new patient
                num_patients += 1
                # Create patient id which is just "s" (for scheduled) followed by number of such patients created.
                patient_id = f"s{num_patients}"
                patient = Patient(patient_id, "scheduled")

                logging.info(f"{self.env.now:.4f}:Patient {patient_id} created at {self.env.now:.4f}.")

                # Register a get_exam process for the new patient
                self.env.process(self.clinic.get_exam(patient))


class WalkinPatientGenerator(object):
    def __init__(self, env, clinic, mean_interarrival_time, stoptime, rg, enabled=True):

        self.env = env
        self.clinic = clinic
        self.mean_interarrival_time = mean_interarrival_time
        self.stoptime = stoptime
        self.rg = rg


        if enabled:
            # Start creating walk in patients
            env.process(self.run())

    def run(self):

        num_patients = 0
        # Loop for generating patients
        while self.env.now < self.stoptime:
            # Generate next interarrival time
            iat = self.rg.exponential(self.mean_interarrival_time)

            # This process will now yield to a 'timeout' event. This process will resume after iat time units.
            yield self.env.timeout(iat)

            # Generate new patient
            num_patients += 1
            # Create patient id which is just "w" (for walkin) followed by number of such patients created.
            patient_id = f"w{num_patients}"
            patient = Patient(patient_id, "walk_in")

            logging.info(f"{self.env.now:.4f}:Patient {patient_id} created at {self.env.now:.4f}.")

            # Register a get_exam process for the new patient
            self.env.process(self.clinic.get_exam(patient))


class OutpatientClinic(object):
    def __init__(self, env, num_rooms_per_provider, num_med_techs, num_physicians,
                 vitals_time_mean, vitals_time_k,
                 exam_time_mean, exam_time_k,
                 post_exam_time_mean, post_exam_time_k,
                 room_turnover_time_mean, room_turnover_time_k,
                 rg
                 ):
        """
        Primary class that encapsulates clinic resources and patient flow logic.

        The detailed patient flow logic is now in in get_vaccinated() method of this class. Also,
        the run_clinic() function is now a run() method in this class. Patient objects are now
        passed in to some methods to enable modelling of multiple patient types.

        Parameters
        ----------
        env
        num_rooms_per_provider
        num_med_techs
        num_physicians

        rg
        """

        # Simulation environment and random number generator
        self.env = env
        self.rg = rg

        # Create list to hold timestamps dictionaries (one per patient)
        self.timestamps_list = []
        # Create lists to hold occupancy tuples (time, occ)
        # self.postvac_occupancy_list = [(0.0, 0.0)]
        # self.vac_occupancy_list = [(0.0, 0.0)]

        # Create SimPy resources
        self.exam_room = simpy.Resource(env, num_rooms_per_provider * num_physicians)
        self.med_tech = simpy.Resource(env, num_med_techs)
        self.physician = simpy.Resource(env, num_physicians)

        # Initialize the patient flow related attributes
        self.vitals_time_mean = vitals_time_mean
        self.vitals_time_k = vitals_time_k
        self.exam_time_mean = exam_time_mean
        self.exam_time_k = exam_time_k
        self.post_exam_time_mean = post_exam_time_mean
        self.post_exam_time_k = post_exam_time_k
        self.room_turnover_time_mean = room_turnover_time_mean
        self.room_turnover_time_k = room_turnover_time_k

    # Create process duration methods

    def vitals_check(self):
        stage_mean = self.vitals_time_mean / self.vitals_time_k
        yield self.env.timeout(self.rg.gamma(self.vitals_time_k, stage_mean))

    def exam(self):
        stage_mean = self.exam_time_mean / self.exam_time_k
        yield self.env.timeout(self.rg.gamma(self.exam_time_k, stage_mean))

    def post_exam(self):
        stage_mean = self.post_exam_time_mean / self.post_exam_time_k
        yield self.env.timeout(self.rg.gamma(self.post_exam_time_k, stage_mean))

    def room_turnover(self):
        stage_mean = self.room_turnover_time_mean / self.room_turnover_time_k
        yield self.env.timeout(self.rg.gamma(self.room_turnover_time_mean, stage_mean))

    def get_exam(self, patient):
        """
        Defines the sequence of steps traversed by patients.

        Parameters
        ----------
        patient : Patient object
        quiet : bool

        Returns
        -------
        None

        Also capture a bunch of timestamps to make it easy to compute various system
        performance measures such as patient waiting times, queue sizes and resource utilization.
        """
        # Patient arrives to clinic - note the arrival time
        arrival_ts = self.env.now

        # Request a med tech for vitals check at separate vitals check station
        # By using request() in a context manager, we'll automatically release the resource when done
        with self.med_tech.request() as request:
            request_med_tech_vitals_ts = self.env.now
            yield request
            got_med_tech_vitals_ts = self.env.now
            yield self.env.process(self.vitals_check())
            release_med_tech_vitals_ts = self.env.now

        # Request exam room - will keep through post exam care
        with self.exam_room.request() as room_request:
            request_exam_room_ts = self.env.now
            yield room_request
            got_exam_room_ts = self.env.now
            # In exam room, get a physician
            with self.physician.request() as physician_request:
                request_physician_ts = self.env.now
                yield physician_request
                got_physician_ts = self.env.now
                yield self.env.process(self.exam())
                release_physician_ts = self.env.now

            # Done with exam, get med tech for post exam care
            with self.med_tech.request() as med_tech_request:
                request_med_tech_post_ts = self.env.now
                yield med_tech_request
                got_med_tech_post_ts = self.env.now
                yield self.env.process(self.post_exam())
                # The patient is ready to exit but room still needs turnover
                exit_system_ts = self.env.now
                yield self.env.process(self.room_turnover())
                # Room turnover done, release the med tech
                release_med_tech_post_ts = self.env.now

            # Done with exam, post exam care and room turnover, release the room
            release_exam_room_ts = self.env.now

        # Create dictionary of timestamps
        timestamps = {'patient_id': patient.patient_id,
                      'appt_type': patient.appt_type,
                      'arrival_ts': arrival_ts,
                      'request_med_tech_vitals_ts': request_med_tech_vitals_ts,
                      'got_med_tech_vitals_ts': got_med_tech_vitals_ts,
                      'release_med_tech_vitals_ts': release_med_tech_vitals_ts,
                      'request_exam_room_ts': request_exam_room_ts,
                      'got_exam_room_ts': got_exam_room_ts,
                      'release_exam_room_ts': release_exam_room_ts,
                      'request_physician_ts': request_physician_ts,
                      'got_physician_ts': got_physician_ts,
                      'release_physician_ts': release_physician_ts,
                      'request_med_tech_post_ts': request_med_tech_post_ts,
                      'got_med_tech_post_ts': got_med_tech_post_ts,
                      'release_med_tech_post_ts': release_med_tech_post_ts,
                      'exit_system_ts': exit_system_ts}

        self.timestamps_list.append(timestamps)


def compute_durations(timestamp_df):
    """Compute time durations of interest from timestamps dataframe and append new cols to dataframe"""

    timestamp_df['init_wait_med_tech'] = \
        timestamp_df.loc[:, 'got_med_tech_vitals_ts'] - timestamp_df.loc[:, 'arrival_ts']
    timestamp_df['wait_for_physician'] = \
        timestamp_df.loc[:, 'got_physician_ts'] - timestamp_df.loc[:, 'release_med_tech_vitals_ts']
    timestamp_df['time_in_system'] = \
        timestamp_df.loc[:, 'exit_system_ts'] - timestamp_df.loc[:, 'arrival_ts']

    performance_measures = ['init_wait_med_tech', 'wait_for_physician', 'time_in_system']
    return timestamp_df, performance_measures


def simulate(arg_dict, rep_num):
    """

    Parameters
    ----------
    arg_dict : dict whose keys are the input args
    rep_num : int, simulation replication number

    Returns
    -------
    Nothing returned but numerous output files written to ``args_dict[output_path]``

    """

    # Create a random number generator for this replication
    seed = arg_dict['seed'] + rep_num - 1
    rg = default_rng(seed=seed)

    # Resource capacity levels
    num_rooms_per_provider = arg_dict['num_rooms_per_provider']
    num_med_techs = arg_dict['num_med_techs']
    num_physicians = arg_dict['num_physicians']

    # Initialize the patient flow related attributes
    patients_per_clinic_block = arg_dict['patients_per_clinic_block']
    patients_per_arrival_group = arg_dict['patients_per_arrival_group']
    clinic_length_minutes = arg_dict['clinic_length_minutes']
    num_arrival_groups = patients_per_clinic_block / patients_per_arrival_group
    mean_interarrival_time = clinic_length_minutes / num_arrival_groups

    vitals_time_mean = arg_dict['vitals_time_mean']
    vitals_time_k = arg_dict['vitals_time_k']

    exam_time_mean = arg_dict['exam_time_mean']
    exam_time_k = arg_dict['exam_time_k']

    post_exam_time_mean = arg_dict['post_exam_time_mean']
    post_exam_time_k = arg_dict['post_exam_time_k']

    room_turnover_time_mean = arg_dict['room_turnover_time_mean']
    room_turnover_time_k = arg_dict['room_turnover_time_k']

    # Other parameters
    scenario = arg_dict['scenario']

    # Run the simulation
    env = simpy.Environment()

    # Create a clinic to simulate
    clinic = OutpatientClinic(env, num_rooms_per_provider, num_med_techs, num_physicians,
                              vitals_time_mean, vitals_time_k,
                              exam_time_mean, exam_time_k,
                              post_exam_time_mean, post_exam_time_k,
                              room_turnover_time_mean, room_turnover_time_k,
                              rg
                              )

    # Initialize and register the patient arrival generators
    walkin_gen = WalkinPatientGenerator(env, clinic, mean_interarrival_time, clinic_length_minutes, rg, enabled=False)
    scheduled_gen = ScheduledPatientGenerator(env, clinic, mean_interarrival_time,
                                              patients_per_arrival_group,
                                              patients_per_clinic_block, clinic_length_minutes, rg, enabled=True)

    # Launch the simulation
    env.run()

    # Create output files and basic summary stats
    if len(arg_dict['output_path']) > 0:
        output_dir = Path.cwd() / arg_dict['output_path']
    else:
        output_dir = Path.cwd()

    # Create paths for the output logs
    clinic_patient_log_path = output_dir / f'clinic_patient_log_{scenario}_{rep_num}.csv'

    # Create patient log dataframe and add scenario and rep number cols
    clinic_patient_log_df = pd.DataFrame(clinic.timestamps_list)
    clinic_patient_log_df['scenario'] = scenario
    clinic_patient_log_df['rep_num'] = rep_num

    # Reorder cols to get scenario and rep_num first
    num_cols = len(clinic_patient_log_df.columns)
    new_col_order = [-2, -1]
    new_col_order.extend([_ for _ in range(0, num_cols - 2)])
    clinic_patient_log_df = clinic_patient_log_df.iloc[:, new_col_order]

    # Compute durations of interest for patient log
    clinic_patient_log_df, performance_measures = compute_durations(clinic_patient_log_df)

    # Export logs to csv
    clinic_patient_log_df.to_csv(clinic_patient_log_path, index=False)

    # Note simulation end time
    end_time = env.now
    print(f"Simulation replication {rep_num} ended at time {end_time}")



def process_sim_output(csvs_path, scenario, performance_measures):
    """

    Parameters
    ----------
    csvs_path : Path object for location of simulation output patient log csv files
    scenario : str

    Returns
    -------
    Dict of dicts

    Keys are:

    'patient_log_rep_stats' --> Contains dataframes from describe on group by rep num. Keys are perf measures.
    'patient_log_ci' -->        Contains dictionaries with overall stats and CIs. Keys are perf measures.
    """

    dest_path = Path(csvs_path) / Path(f"consolidated_clinic_patient_log_{scenario}.csv")

    sort_keys = ['scenario', 'rep_num']

    # Create empty dict to hold the DataFrames created as we read each csv file
    dfs = {}

    # Loop over all the csv files
    for csv_f in Path(csvs_path).glob('clinic_patient_log_*.csv'):
        # Split the filename off from csv extension. We'll use the filename
        # (without the extension) as the key in the dfs dict.
        fstem = csv_f.stem

        # Read the next csv file into a pandas DataFrame and add it to
        # the dfs dict.
        df = pd.read_csv(csv_f)
        dfs[fstem] = df

    # Use pandas concat method to combine the file specific DataFrames into
    # one big DataFrame.
    patient_log_df = pd.concat(dfs)

    # Since we didn't try to control the order in which the files were read,
    # we'll sort the final DataFrame in place by the specified sort keys.
    patient_log_df = patient_log_df.sort_values(sort_keys)

    # Export the final DataFrame to a csv file. Suppress the pandas index.
    patient_log_df.to_csv(dest_path, index=False)

    # Compute summary statistics for several performance measures
    patient_log_stats = summarize_patient_log(patient_log_df, scenario, performance_measures)

    # Now delete the individual replication files
    for csv_f in csvs_path.glob('clinic_patient_log_*.csv'):
        csv_f.unlink()

    return patient_log_stats


def summarize_patient_log(patient_log_df, scenario, performance_measures):
    """

    Parameters
    ----------
    patient_log_df : DataFrame created by process_sim_output
    scenario : str

    Returns
    -------
    Dict of dictionaries - See comments below
    """

    # Create empty dictionaries to hold computed results
    patient_log_rep_stats = {}  # Will store dataframes from describe on group by rep num. Keys are perf measures.
    patient_log_ci = {}  # Will store dictionaries with overall stats and CIs. Keys are perf measures.
    patient_log_stats = {}  # Container dict returned by this function containing the two previous dicts.

    # # Create list of performance measures for looping over
    # performance_measures = ['init_wait_med_tech', 'wait_for_physician', 'time_in_system', 'exit_system_ts']

    for pm in performance_measures:
        # Compute descriptive stats for each replication and store dataframe in dict
        patient_log_rep_stats[pm] = patient_log_df.groupby(['rep_num'])[pm].describe()
        # Compute across replication stats
        n_samples = patient_log_rep_stats[pm]['mean'].count()
        mean_mean = patient_log_rep_stats[pm]['mean'].mean()
        sd_mean = patient_log_rep_stats[pm]['mean'].std()
        ci_95_lower = mean_mean - 1.96 * sd_mean / math.sqrt(n_samples)
        ci_95_upper = mean_mean + 1.96 * sd_mean / math.sqrt(n_samples)
        mean_p75 = patient_log_rep_stats[pm][r'75%'].mean()
        mean_max = patient_log_rep_stats[pm]['max'].mean()
        # Store cross replication stats as dict in dict
        patient_log_ci[pm] = {'n_samples': n_samples, 'mean_mean': mean_mean, 'sd_mean': sd_mean,
                              'ci_95_lower': ci_95_lower, 'ci_95_upper': ci_95_upper,
                              'mean_p75': mean_p75, 'mean_max': mean_max}

    patient_log_stats['scenario'] = scenario
    patient_log_stats['patient_log_rep_stats'] = patient_log_rep_stats
    # Convert the final summary stats dict to a DataFrame
    patient_log_stats['patient_log_ci'] = pd.DataFrame(patient_log_ci)

    return patient_log_stats


def create_configs_from_inputs_csv(exp, scenarios_csv_file_path, config_path,
                                   run_script_path):
    """
    Create one simulation configuration file per scenario.

    Parameters
    ----------
    exp : str, experiment identifier
    scenarios_csv_file_path : str or Path, simulation scenario input csv file
    simulation_settings_path : str or Path, YAML file with simulation settings
    config_path : str or Path, destination for scenario specific config files
    run_script_path : str or Path, destination for shell scripts for running simulation scenarios
    update_check_rho : bool (Default=False), if True, recompute rho check values. Set to True if manual capacity levels set.

    Returns
    -------
    No return value
    """

    # Read scenarios file in DataFrame
    scenarios_df = pd.read_csv(scenarios_csv_file_path)

    global_vars = {}
    run_script_file_path = Path(run_script_path, f'{exp}_run.sh')
    with open(run_script_file_path, 'w') as bat_file:
        # Iterate over rows in scenarios file
        for row in scenarios_df.iterrows():
            scenario = int(row[1]['scenario'].tolist())

            global_vars['arrival_rate'] = row[1]['arrival_rate'].tolist()

            global_vars['mean_los_obs'] = row[1]['mean_los_obs'].tolist()
            global_vars['num_erlang_stages_obs'] = int(row[1]['num_erlang_stages_obs'])

            global_vars['mean_los_ldr'] = float(row[1]['mean_los_ldr'])
            global_vars['num_erlang_stages_ldr'] = int(row[1]['num_erlang_stages_ldr'])

            global_vars['mean_los_pp_noc'] = float(row[1]['mean_los_pp_noc'])
            global_vars['mean_los_pp_c'] = float(row[1]['mean_los_pp_c'])
            global_vars['num_erlang_stages_pp'] = int(row[1]['num_erlang_stages_pp'])

            global_vars['mean_los_csect'] = float(row[1]['mean_los_csect'])
            global_vars['num_erlang_stages_csect'] = int(row[1]['num_erlang_stages_csect'])

            global_vars['c_sect_prob'] = float(row[1]['c_sect_prob'])

            config = {}
            config['locations'] = settings['locations']
            cap_obs = int(row[1]['cap_obs'].tolist())
            cap_ldr = int(row[1]['cap_ldr'].tolist())
            cap_pp = int(row[1]['cap_pp'].tolist())
            config['locations'][1]['capacity'] = cap_obs
            config['locations'][2]['capacity'] = cap_ldr
            config['locations'][4]['capacity'] = cap_pp

            # Write scenario config file

            config['scenario'] = scenario
            config['run_settings'] = settings['run_settings']
            config['output'] = settings['output']
            config['random_number_streams'] = settings['random_number_streams']

            config['routes'] = settings['routes']
            config['global_vars'] = global_vars

            config_file_path = Path(config_path) / f'{exp}_scenario_{scenario}.yaml'

            with open(config_file_path, 'w', encoding='utf-8') as config_file:
                yaml.dump(config, config_file)

            run_line = f"obflow_sim {config_file_path} --loglevel=WARNING\n"
            bat_file.write(run_line)

        # Create output file processing line
        # output_proc_line = f'python obflow_stat.py {output_path_} {exp_suffix_} '
        # output_proc_line += f"--run_time {settings['run_settings']['run_time']} "
        # output_proc_line += f"--warmup_time {settings['run_settings']['warmup_time']} --include_inputs "
        # output_proc_line += f"--scenario_inputs_path {scenarios_csv_path_} --process_logs "
        # output_proc_line += f"--stop_log_path {settings['paths']['stop_logs']} "
        # output_proc_line += f"--occ_stats_path {settings['paths']['occ_stats']}"
        # bat_file.write(output_proc_line)


            # Rewrite scenarios input file with updated rho_checks
            scenarios_df.to_csv(scenarios_csv_file_path, index=False)

    print(f'Config files written to {Path(config_path)}')
    return run_script_file_path


def process_command_line():
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parser
    parser = argparse.ArgumentParser(prog='op_clinic',
                                     description='Run outpatient clinic simulation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add arguments
    parser.add_argument(
        "--config", type=str, default=None,
        help="Configuration file containing input parameter arguments and values"
    )

    parser.add_argument(
        "--scenario", type=str, default=datetime.now().strftime("%Y.%m.%d.%H.%M."),
        help="Appended to output filenames."
    )

    parser.add_argument("--patients_per_clinic_block", default=24, help="patients per clinic block",
                        type=int)

    parser.add_argument("--patients_per_arrival_group", default=2, help="patients per group at each appointment time",
                        type=int)

    parser.add_argument("--clinic_length_minutes", default=240,
                        help="length of clinic block in minutes (arrivals stop after this time)",
                        type=float)

    parser.add_argument("--num_med_techs", default=2, help="number of medical technicians",
                        type=int)

    parser.add_argument("--num_rooms_per_provider", default=2, help="number of exam rooms per exam provider",
                        type=int)

    parser.add_argument("--num_physicians", default=15, help="number of physicians",
                        type=int)

    parser.add_argument("--vitals_time_mean", default=6.0,
                        help="Mean time (mins) for taking vital signs",
                        type=float)

    parser.add_argument("--vitals_time_k", default=5,
                        help="Number of erlang stages for vitals time distribution",
                        type=int)

    parser.add_argument("--exam_time_mean", default=10.0,
                        help="Mean time (mins) for exam",
                        type=float)

    parser.add_argument("--exam_time_k", default=1,
                        help="Number of erlang stages for exam time distribution",
                        type=int)

    parser.add_argument("--post_exam_time_mean", default=2.0,
                        help="Mean time (mins) for post exam care",
                        type=float)

    parser.add_argument("--post_exam_time_k", default=2,
                        help="Number of erlang stages for post  exam time distribution",
                        type=int)

    parser.add_argument("--room_turnover_time_mean", default=2.5,
                        help="Mean time (mins) for room_turnover",
                        type=float)

    parser.add_argument("--room_turnover_time_k", default=7,
                        help="Number of erlang stages for room_turnover distribution",
                        type=int)

    parser.add_argument("--num_reps", default=1, help="number of simulation replications",
                        type=int)

    parser.add_argument("--seed", default=3, help="random number generator seed",
                        type=int)

    parser.add_argument("--log_level", default=logging.INFO, type=int,
                        help="Use valid values for logging package")

    parser.add_argument(
        "--output_path", type=str, default=".", help="location for output file writing")

    # do the parsing
    args = parser.parse_args()

    if args.config is not None:
        # Read inputs from config file
        with open(args.config, "r") as fin:
            args = parser.parse_args(fin.read().split())

    return args


def main():
    args = process_command_line()
    print(args)

    output_summary_path = Path(args.output_path) / Path("summary_stats.csv")

    # Quick setup of root logger
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    # Retrieve root logger (no logger name passed to ``getLogger()``) and update its level
    logger = logging.getLogger()
    logger.setLevel(args.log_level)

    # Simulation settings
    num_reps = args.num_reps
    scenario = args.scenario

    # Main simulation replication loop
    for i in range(1, num_reps + 1):
        simulate(vars(args), i)

    # Consolidate the patient logs and compute summary stats
    performance_measures = ['init_wait_med_tech', 'wait_for_physician', 'time_in_system', 'exit_system_ts']
    patient_log_stats = process_sim_output(Path(args.output_path), scenario, performance_measures)
    print(f"\nScenario: {scenario}")
    pd.set_option("display.precision", 3)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    #print(patient_log_stats['patient_log_rep_stats'])
    summary_stats = patient_log_stats['patient_log_ci']
    print(summary_stats)

    mean_waiti = summary_stats.loc['mean_mean', 'init_wait_med_tech']
    mean_waitp = summary_stats.loc['mean_mean', 'wait_for_physician']
    mean_time_in_system = summary_stats.loc['mean_mean', 'time_in_system']
    mean_eod = summary_stats.loc['mean_max', 'exit_system_ts']

    scenario_results = [scenario, mean_waiti, mean_waitp, mean_time_in_system, mean_eod]

    with open(output_summary_path, "a") as fout:
        writer = csv.writer(fout)
        writer.writerow(scenario_results)



if __name__ == '__main__':
    main()
